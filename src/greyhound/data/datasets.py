import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
import polars as pl
import pybigtools
import torch
from enformer_pytorch.data import FastaInterval, GenomeIntervalDataset
from loguru import logger
from pandera.typing import Series
from pydantic import BaseModel
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class FastGenomicRegion:
    """
    Lightweight genomic region class for internal use that skips validation.
    Much faster than GenomicRegion for trusted data during training.

    Performance comparison:
    - GenomicRegion: ~5-10 μs per creation (with validation)
    - FastGenomicRegion: ~0.1-0.5 μs per creation (no validation)

    Use this for trusted internal data where validation is unnecessary.
    Use GenomicRegion for user-facing APIs where validation is important.
    """

    chromosome: str
    start: int
    end: int
    strand: str = "+"

    def to_genomic_region(
        self, style: Literal["ucsc", "ensembl"] = "ucsc"
    ) -> "GenomicRegion":
        """Convert to full GenomicRegion with validation if needed."""
        return GenomicRegion(
            chromosome=self.chromosome,
            start=self.start,
            end=self.end,
            strand=self.strand,
            style=style,
        )


class GenomicRegion(BaseModel):
    chromosome: str
    start: int
    end: int
    strand: Literal["+", "-"] = "+"
    style: Literal["ucsc", "ensembl"] = "ucsc"

    def __model_post_init__(self):
        if self.start >= self.end:
            raise ValueError("Start position must be less than end position.")
        if self.chromosome.startswith("chr"):
            self.chromosome = self.chromosome[3:]

        match self.style:
            case "ucsc":
                self.chromosome = f"chr{self.chromosome.replace('chr', '')}"
            case "ensembl":
                self.chromosome = self.chromosome.replace("chr", "")
            case _:
                raise ValueError(f"Unsupported style: {self.style}")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.start}-{self.end}({self.strand})"

    @classmethod
    def from_str(
        cls, region_str: str, style: Literal["ucsc", "ensembl"] = "ucsc"
    ) -> "GenomicRegion":
        region_str = region_str.replace(",", "")
        chromosome, region = region_str.split(":")
        try:
            start_end, strand = region.split("(")
            strand = strand[:-1]
        except ValueError:
            start_end = region
            strand = "+"

        start, end = start_end.split("-")
        return cls(
            chromosome=chromosome,
            start=int(start),
            end=int(end),
            strand=strand,
            style=style,
        )

    @classmethod
    def from_tuple(cls, region_tuple: tuple) -> "GenomicRegion":
        return cls(
            chromosome=region_tuple[0],
            start=region_tuple[1],
            end=region_tuple[2],
            strand=region_tuple[3],
        )

    @classmethod
    def from_list(cls, region_list: list) -> "GenomicRegion":
        return cls(
            chromosome=region_list[0],
            start=region_list[1],
            end=region_list[2],
            strand=region_list[3],
        )

    @classmethod
    def from_dict(cls, region_dict: dict) -> "GenomicRegion":
        return cls(
            chromosome=region_dict["chromosome"],
            start=region_dict["start"],
            end=region_dict["end"],
            strand=region_dict["strand"],
        )

    @classmethod
    def from_named_tuple(cls, region_named_tuple: tuple) -> "GenomicRegion":
        return cls(
            chromosome=region_named_tuple.chromosome,
            start=region_named_tuple.start,
            end=region_named_tuple.end,
            strand=region_named_tuple.strand,
        )

    @classmethod
    def into(cls, gr: str | tuple | list | dict) -> "GenomicRegion":
        try:
            match type(gr):
                case "str":
                    return cls.from_str(gr)
                case "tuple":
                    return cls.from_tuple(gr)
                case "list":
                    return cls.from_list(gr)
                case "dict":
                    return cls.from_dict(gr)
                case "namedtuple":
                    return cls.from_named_tuple(gr)
                case _:
                    raise ValueError("Unsupported type")
        except ValueError as e:
            logger.error(e)
            raise e


class DataSources(pa.DataFrameModel):
    """
    DataFrame model for validating data sources.
    """

    sample_id: Series[str] = pa.Field(
        description="Unique identifier for the sample",
        coerce=True,
    )
    path: Series[str] = pa.Field(
        description="Path to the data source file",
        coerce=True,
    )
    scaling_factor: Series[float] = pa.Field(
        description="Scaling factor for the data source",
        coerce=True,
    )
    soft_clip: Series[float] = pa.Field(
        description="Soft clipping value for the data source",
        coerce=True,
    )
    hard_clip: Series[float] = pa.Field(
        description="Hard clipping value for the data source",
        coerce=True,
    )
    power_transform_exponent: Series[float] = pa.Field(
        description="Exponent for power transformation",
        coerce=True,
        default=1.0,
    )



class ChromatinDataset(Dataset):
    def __init__(
        self,
        genome_dataset: GenomeIntervalDataset,
        data: pd.DataFrame = None,
        bigwig_dir: str | Path = None,
        clip_soft: int = 32,
        clip_hard: int = 128,
        scale_factor: float = 2.0,
        power_transform_exponent: float = 1.0,
        cache_bigwig_handles: bool = False,
        num_workers: int = 4,
        scale_method: Literal["multiply", "divide"] = "multiply",
    ):
        self.genome_dataset = genome_dataset
        self.clip_soft = clip_soft
        self.clip_hard = clip_hard
        self.scale_factor = scale_factor
        self.power_transform_exponent = power_transform_exponent
        self.min_value = torch.finfo(torch.float16).min
        self.max_value = torch.finfo(torch.float16).max
        self.num_workers = num_workers

        # Thread-local data
        self._thread_local = threading.local()

        # Cache handles only if single-worker (safe)
        self.cache_bigwig_handles = cache_bigwig_handles and num_workers == 0
        if cache_bigwig_handles and num_workers > 0:
            logger.warning("Disabling cache_bigwig_handles because multiple workers are used.")

        # Load BigWig metadata
        if data is not None:
            self.data = data
            self.bigwig_files = self.data.path.tolist()
        else:
            if bigwig_dir is None:
                raise ValueError("Either 'data' or 'bigwig_dir' must be provided.")
            self.bigwig_dir = Path(bigwig_dir).resolve()
            self.bigwig_files = list(self.bigwig_dir.glob("*.bw")) + list(self.bigwig_dir.glob("*.bigWig"))
            self.data = None

        if not self.bigwig_files:
            raise FileNotFoundError("No BigWig files found.")

        # Scaling and clipping tensors
        if self.data is not None:
            self.scaling_factors = torch.from_numpy(self.data.scaling_factor.values)
            self.power_transform_exponent = torch.from_numpy(self.data.power_transform_exponent.values)
            self.soft_clip = torch.from_numpy(self.data.soft_clip.values)
            self.hard_clip = torch.from_numpy(self.data.hard_clip.values)
        else:
            n = len(self.bigwig_files)
            self.scaling_factors = torch.tensor(
                [self.scale_factor] * n
            )
            self.power_transform_exponent = torch.tensor(
                [self.power_transform_exponent] * n
            )
            self.soft_clip = torch.tensor([self.clip_soft] * n)
            self.hard_clip = torch.tensor([self.clip_hard] * n)

        if scale_method == "divide":
            self.scaling_factors = 1 / self.scaling_factors
    
    @classmethod
    def from_csv(
        cls,
        csv_file: str | Path,
        **kwargs) -> "ChromatinDataset":
        """
        Create a ChromatinDataset from a CSV file.

        Args:
            csv_file (str | Path): Path to the CSV file containing metadata.
            **kwargs: Additional parameters for ChromatinDataset initialization.

        Returns:
            ChromatinDataset: Initialized dataset instance.
        """
        df = pd.read_csv(csv_file)
        data = DataSources.validate(df)
        return cls(data=data, **kwargs)
    
    @property
    def params(self) -> dict[str, Any]:
        """
        Get the parameters of the dataset as a dictionary.
        
        Returns:
            dict[str, Any]: Dictionary containing dataset parameters.
        """
        return {
            'scaling_factors': self.scaling_factors,
            'soft_clip': self.soft_clip,
            'hard_clip': self.hard_clip,
            'power_transform_exponent': self.power_transform_exponent,
            'n_labels': self.n_labels,
            'n_samples': len(self),
            'region_length': self.genome_dataset.fasta.context_length,
            'bigwig_files': self.bigwig_files,
            'data' : self.data,
        }

    def __len__(self):
        return len(self.genome_dataset)

    def __getitem__(self, idx):
        self._ensure_fasta_initialized()
        row = self.genome_dataset.df[idx]

        coordinates = FastGenomicRegion(
            chromosome=row["column_1"].item(),
            start=row["column_2"].item(),
            end=row["column_3"].item(),
            strand="+",
        )

        targets = self._extract_data(coordinates)
        inputs, _, rc_augs = self.genome_dataset[idx]

        if rc_augs[0]:  # reverse complemented
            targets = torch.flip(targets, dims=[1])

        inputs = inputs.permute(1, 0)

        return {
            "input_ids": inputs,
            "label_ids": targets,
        }

    def _ensure_fasta_initialized(self):
        if not getattr(self._thread_local, "fasta_initialized", False):
            fasta = self.genome_dataset.fasta
            self.genome_dataset.fasta = FastaInterval(
                fasta_file=fasta.seqs.filename,
                context_length=fasta.context_length,
                return_seq_indices=fasta.return_seq_indices,
                shift_augs=fasta.shift_augs,
                rc_aug=fasta.rc_aug,
            )
            self._thread_local.fasta_initialized = True

    def _extract_data(self, coordinates):
        signals = self._extract_from_bigwig(coordinates)
        signal_array = np.array(signals, dtype=np.float32)
        tensor = torch.from_numpy(signal_array)
        tensor = torch.nan_to_num(tensor)
        tensor = (
            torch.nn.functional.avg_pool1d(tensor.unsqueeze(0), kernel_size=32, stride=32)
            .squeeze(0) * 32
        )
        return self._scale_inplace(
            tensor,
            self.scaling_factors,
            self.hard_clip,
            self.soft_clip,
            self.power_transform_exponent,
        )

    def _extract_from_bigwig(self, coordinates):
        handles = self._get_bigwig_handles()
        if handles is not None:
            return [
                handle.values(coordinates.chromosome, coordinates.start, coordinates.end)
                for handle in handles
            ]
        else:
            return [
                pybigtools.open(str(path)).values(
                    coordinates.chromosome, coordinates.start, coordinates.end
                )
                for path in self.bigwig_files
            ]

    def _get_bigwig_handles(self):
        if not hasattr(self._thread_local, "bigwig_handles"):
            if self.cache_bigwig_handles:
                self._thread_local.bigwig_handles = [
                    pybigtools.open(str(p)) for p in self.bigwig_files
                ]
            else:
                self._thread_local.bigwig_handles = None
        return self._thread_local.bigwig_handles

    def _scale_inplace(self, x, scaling_factors, hard_clip, soft_clip, power_exponent):
        scaling_factors = scaling_factors.unsqueeze(1)
        power_exponent = power_exponent.unsqueeze(1)
        soft_clip = soft_clip.unsqueeze(1)
        hard_clip = hard_clip.unsqueeze(1)

        x.mul_(scaling_factors)
        x.add_(1).pow_(power_exponent).sub_(1)

        mask = x > soft_clip
        if mask.any():
            excess = x - soft_clip + 1
            clipped = torch.sqrt(excess) + soft_clip - 1
            x = torch.where(mask, clipped, x)

        x.clamp_(min=-hard_clip, max=hard_clip)
        x.clamp_(min=self.min_value, max=self.max_value)
        return x

    def __del__(self):
        if hasattr(self._thread_local, "bigwig_handles"):
            for h in self._thread_local.bigwig_handles:
                try:
                    h.close()
                except Exception:
                    pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_thread_local", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_local = threading.local()

    @property
    def n_labels(self):
        return len(self.bigwig_files)

    @property
    def id2label(self):
        return {i: Path(f).stem for i, f in enumerate(self.bigwig_files)}

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}


def undo_squashed_scale(
    x, clip_soft=384, track_transform=3 / 4, track_scale=0.01, old_transform=True
):
    """
    Reverses the squashed scaling transformation applied to the output profiles.

    Args:
        x (torch.Tensor): The input tensor to be unsquashed.
        clip_soft (float, optional): The soft clipping value. Defaults to 384.
        track_transform (float, optional): The transformation factor. Defaults to 3/4.
        track_scale (float, optional): The scale factor. Defaults to 0.01.

    Returns:
        torch.Tensor: The unsquashed tensor.
    """
    x = x.clone()  # IMPORTANT BECAUSE OF IMPLACE OPERATIONS TO FOLLOW?

    if old_transform:
        x = x / track_scale
        unclip_mask = x > clip_soft
        x[unclip_mask] = (x[unclip_mask] - clip_soft) ** 2 + clip_soft
        x = x ** (1.0 / track_transform)
    else:
        unclip_mask = x > clip_soft
        x[unclip_mask] = (x[unclip_mask] - clip_soft + 1) ** 2 + clip_soft - 1
        x = (x + 1) ** (1.0 / track_transform) - 1
        x = x / track_scale
    return x


def train_filter(df: pl.DataFrame, test_fold: int, val_fold: int) -> pl.DataFrame:
    """
    Filter the DataFrame to only include rows where 'train' is True.
    """
    return df.filter(
        (pl.col("column_4") != f"fold{test_fold}")
        & (pl.col("column_4") != f"fold{val_fold}")
    )


def val_filter(df: pl.DataFrame, test_fold: int, val_fold: int) -> pl.DataFrame:
    """
    Filter the DataFrame to only include rows where 'val' is True.
    """
    return df.filter(pl.col("column_4") == f"fold{val_fold}").sample(fraction=0.5)


def test_filter(df: pl.DataFrame, test_fold: int, val_fold: int) -> pl.DataFrame:
    """
    Filter the DataFrame to only include rows where 'test' is True.
    """
    return df.filter(pl.col("column_4") == f"fold{test_fold}")


def toy_filter(df: pl.DataFrame, test_fold: int, val_fold: int) -> pl.DataFrame:
    """
    Filter the DataFrame to only include rows where 'toy' is True.
    """
    return df.sample(50, with_replacement=False)
