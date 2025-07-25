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
        cache_bigwig_handles: bool = True,
        num_workers: int = 4,
        scale_method: Literal["multiply", "divide"] = "multiply",
    ) -> None:
        """
        Initializes the ChromatinDataset.
        Args:
            genome_dataset (GenomeIntervalDataset): The genome dataset containing genomic regions.
            bigwig_dir (Union[str, Path]): Directory containing BigWig files.
            clip_soft (int, optional): Soft clipping value. Defaults to 32.
            clip_hard (int, optional): Hard clipping value. Defaults to 128.
            scale_factor (float, optional): Scaling factor for the signal. Defaults to 2.0.
            power_transform_exponent (float, optional): Exponent for power transformation. Defaults to 1.0.
            cache_bigwig_handles (bool, optional): Whether to cache BigWig file handles. Defaults to True.
            num_workers (int, optional): Number of worker threads for BigWig extraction. Defaults to 4.
        """
        self.genome_dataset = genome_dataset
        self.clip_soft = clip_soft
        self.clip_hard = clip_hard
        self.scale_factor = scale_factor
        self.power_transform_exponent = power_transform_exponent
        self.cache_bigwig_handles = cache_bigwig_handles
        self.num_workers = num_workers
        self.min_value = torch.finfo(torch.float16).min
        self.max_value = torch.finfo(torch.float16).max

        # Thread-local storage for BigWig handles to avoid conflicts in multiprocessing
        self._thread_local = threading.local()

        if data is not None:
            try:
                self.data = DataSources.validate(data)
            except pa.errors.SchemaError as e:
                logger.error(f"Data validation failed: {e}")
                raise e

            self.bigwig_files = self.data.path.tolist()

        else:
            if bigwig_dir is None:
                raise ValueError("Either 'data' or 'bigwig_dir' must be provided.")
            self.data = None
            self.bigwig_dir = Path(bigwig_dir)
            if not self.bigwig_dir.exists():
                raise FileNotFoundError(
                    f"BigWig directory {self.bigwig_dir} does not exist."
                )
            if not self.bigwig_dir.is_dir():
                raise NotADirectoryError(
                    f"BigWig directory {self.bigwig_dir} is not a directory."
                )
            self.bigwig_dir = self.bigwig_dir.resolve()
            self.bigwig_files = list(self.bigwig_dir.glob("*.bw")) + list(
                self.bigwig_dir.glob("*.bigWig")
            )

        if not self.bigwig_files:
            raise FileNotFoundError(f"No BigWig files found in {self.bigwig_dir}.")

        # Apply scaling transformation in-place for memory efficiency
        if self.data is not None:
            self.scaling_factors = torch.from_numpy(self.data.scaling_factor.values)

            if self.scale_method == "divide":
                self.scaling_factors = 1 / self.scaling_factors

            self.power_transform_exponent = torch.from_numpy(
                self.data.power_transform_exponent.values
            )
            self.soft_clip = torch.from_numpy(self.data.soft_clip.values)
            self.hard_clip = torch.from_numpy(self.data.hard_clip.values)
        else:
            self.scaling_factors = torch.tensor(
                [self.scale_factor] * len(self.bigwig_files),
            )
            if self.scale_method == "divide":
                self.scaling_factors = 1 / self.scaling_factors

            self.power_transform_exponent = torch.tensor(
                [self.power_transform_exponent] * len(self.bigwig_files),
            )
            self.soft_clip = torch.tensor(
                [self.clip_soft] * len(self.bigwig_files),
            )
            self.hard_clip = torch.tensor(
                [self.clip_hard] * len(self.bigwig_files),
            )

        # Pre-allocate common tensors to avoid repeated allocation
        self._setup_tensor_cache()

        # Pre-format chromosome names to avoid repeated string operations
        self._preprocess_chromosome_names()

    @classmethod
    def from_csv(cls, csv_file: str | Path, **kwargs) -> "ChromatinDataset":
        """
        Create a ChromatinDataset from a CSV file containing data sources.
        Args:
            csv_file (str | Path): Path to the CSV file.
            **kwargs: Additional arguments for ChromatinDataset initialization.
        Returns:
            ChromatinDataset: An instance of ChromatinDataset.
        """
        df = pd.read_csv(csv_file)
        return cls(data=df, **kwargs)

    def _preprocess_chromosome_names(self):
        """
        Pre-process chromosome names in the dataset to avoid repeated string operations
        during __getitem__ calls. This can provide significant speedup.
        """
        if hasattr(self.genome_dataset.df, "with_columns"):
            # For polars DataFrame - ensure chromosome names are UCSC format
            try:
                self.genome_dataset.df = self.genome_dataset.df.with_columns(
                    [
                        pl.when(pl.col("column_1").str.starts_with("chr"))
                        .then(pl.col("column_1"))
                        .otherwise(pl.concat_str([pl.lit("chr"), pl.col("column_1")]))
                        .alias("column_1")
                    ]
                )
            except Exception as e:
                logger.error(
                    "Failed to preprocess chromosome names in polars DataFrame."
                )
                raise ValueError(
                    "Ensure 'column_1' contains chromosome names in UCSC format."
                ) from e

    def _setup_tensor_cache(self):
        """Setup tensor cache and thread-local BigWig handles."""
        pass  # Will be implemented if needed

    def _get_bigwig_handles(self):
        """Get thread-local BigWig handles for efficient access."""
        if not hasattr(self._thread_local, "bigwig_handles"):
            if self.cache_bigwig_handles:
                self._thread_local.bigwig_handles = [
                    pybigtools.open(str(bigwig_file))
                    for bigwig_file in self.bigwig_files
                ]
            else:
                self._thread_local.bigwig_handles = None
        return self._thread_local.bigwig_handles

    def __len__(self):
        return len(self.genome_dataset)

    @property
    def params(self) -> dict[str, Any]:
        """
        Returns the parameters of the dataset as a dictionary.
        """
        return {
            "clip_soft": self.clip_soft,
            "clip_hard": self.clip_hard,
            "scale_factor": self.scale_factor,
            "power_transform_exponent": self.power_transform_exponent,
            "bigwigs": str(self.bigwig_files),
            "fasta_file": str(self.genome_dataset.fasta.seqs.filename),
            "context_length": self.genome_dataset.fasta.context_length,
        }

    def _reinit_fasta_reader(self):
        """
        Re-initializes the FastaInterval reader only if needed.

        This is necessary because pyfaidx and torch multiprocessing are not compatible.
        Optimized to avoid unnecessary reinitialization.
        """
        if (
            not hasattr(self._thread_local, "fasta_initialized")
            or not self._thread_local.fasta_initialized
        ):
            self.genome_dataset.fasta = FastaInterval(
                fasta_file=self.genome_dataset.fasta.seqs.filename,
                context_length=self.genome_dataset.fasta.context_length,
                return_seq_indices=self.genome_dataset.fasta.return_seq_indices,
                shift_augs=self.genome_dataset.fasta.shift_augs,
                rc_aug=self.genome_dataset.fasta.rc_aug,
            )
            self._thread_local.fasta_initialized = True

    def _scale(
        self,
        x: torch.Tensor,
        hard_clip: int = 128,
        soft_clip: int = 32,
        scale_factor: float = 2.0,
        power_transform_exponent: float = 1,
    ) -> torch.Tensor:
        """
        Applies squashed scaling transformation to input profiles.
        Forward transformation corresponding to undo_squashed_scale.

        Args:
            x (torch.Tensor): Input tensor to transform
            scale (bool): Apply scaling (default: True)

        Returns:
            torch.Tensor: Transformed tensor
        """

        # Transform order: scale → power → clip
        x.mul_(scale_factor)

        # Power transform with offset
        x.add_(1).pow_(power_transform_exponent).sub_(1)

        # Soft clipping with offset handling
        mask = x > soft_clip
        if mask.any():
            excess = x[mask] - soft_clip + 1
            x[mask] = torch.sqrt(excess) + soft_clip - 1

        # Hard clipping
        x.clamp_(min=-hard_clip, max=hard_clip)
        # Ensure values are within the specified range
        x.clamp_(min=self.min_value, max=self.max_value)
        return x

    def _scale_inplace(
        self,
        x: torch.Tensor,
        scaling_factors: torch.Tensor,
        hard_clip: torch.Tensor,
        soft_clip: torch.Tensor,
        power_transform_exponent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized in-place scaling transformation to reduce memory allocations.

        Args:
            x (torch.Tensor): Input tensor to transform
            scaling_factors (torch.Tensor): Scaling factors for each element
            hard_clip (torch.Tensor): Hard clipping values for each element
            soft_clip (torch.Tensor): Soft clipping values for each element
            power_transform_exponent (torch.Tensor): Power transform exponents for each element
        Returns:
            torch.Tensor: Transformed tensor (same object, modified in-place)
        """
        # Transform order: scale → power → clip (all in-place)
        # Reshape scaling parameters for broadcasting across tracks (dim 0)
        scaling_factors = scaling_factors.unsqueeze(1)  # (num_tracks, 1)
        power_transform_exponent = power_transform_exponent.unsqueeze(
            1
        )  # (num_tracks, 1)
        soft_clip = soft_clip.unsqueeze(1)  # (num_tracks, 1)
        hard_clip = hard_clip.unsqueeze(1)  # (num_tracks, 1)

        x.mul_(scaling_factors)

        # Power transform with offset (in-place)
        x.add_(1).pow_(power_transform_exponent).sub_(1)

        # Soft clipping with offset handling (in-place where possible)
        mask = x > soft_clip
        if mask.any():
            # Handle soft clipping with proper broadcasting
            excess = x - soft_clip + 1
            clipped_values = torch.sqrt(excess) + soft_clip - 1
            x = torch.where(mask, clipped_values, x)

        # Hard clipping and range enforcement (in-place)
        x.clamp_(min=-hard_clip, max=hard_clip)
        x.clamp_(min=self.min_value, max=self.max_value)

        return x

    def _extract_from_bigwig(
        self, coordinates: FastGenomicRegion | GenomicRegion
    ) -> list:
        """
        Extract BigWig data using cached handles for better performance.

        Args:
            coordinates: Object with chromosome, start, end attributes

        Returns:
            list: List of extracted values from each BigWig file
        """
        bigwig_handles = self._get_bigwig_handles()

        if bigwig_handles is not None:
            # Use cached handles for much faster access
            return [
                handle.values(
                    coordinates.chromosome, coordinates.start, coordinates.end
                )
                for handle in bigwig_handles
            ]
        else:
            # Fallback to opening files each time (slower but lower memory)
            return [
                pybigtools.open(str(bigwig_file)).values(
                    coordinates.chromosome,
                    coordinates.start,
                    coordinates.end,
                )
                for bigwig_file in self.bigwig_files
            ]

    def _extract_data(
        self, coordinates: FastGenomicRegion | GenomicRegion
    ) -> torch.Tensor:
        """
        Extracts data from BigWig files for the given genomic coordinates.
        Optimized for performance with minimal tensor operations.

        Args:
            coordinates (GenomicRegion): The genomic coordinates to extract data for.

        Returns:
            torch.Tensor: A tensor containing the extracted data with shape (num_tracks, sequence_length)
                        suitable for Borzoi model training.
        """
        # Extract all BigWig data using optimized method
        signal = self._extract_from_bigwig(coordinates)

        # Single vectorized conversion to tensor with optimized dtype
        signal_array = np.array(signal, dtype=np.float32)
        signal_tensor = torch.from_numpy(signal_array)

        # Replace NaN values efficiently
        signal_tensor = torch.nan_to_num(signal_tensor)

        # Perform binning operation more efficiently
        signal_tensor_binned = (
            torch.nn.functional.avg_pool1d(
                signal_tensor.unsqueeze(0), kernel_size=32, stride=32
            ).squeeze(0)
            * 32
        )
        return self._scale_inplace(
            signal_tensor_binned,
            self.scaling_factors,
            self.hard_clip,
            self.soft_clip,
            self.power_transform_exponent,
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Gets the item at the given index with maximum performance optimization.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: Dictionary containing 'input_ids' and 'label_ids' tensors.
        """
        # Only reinitialize FASTA reader if needed (per-thread, not per-call)
        self._reinit_fasta_reader()

        # Get coordinates directly from DataFrame for efficiency
        row = self.genome_dataset.df[idx]

        # Use lightweight coordinate class - no validation overhead!
        coordinates = FastGenomicRegion(
            chromosome=row["column_1"].item(),
            start=row["column_2"].item(),
            end=row["column_3"].item(),
            strand="+",
        )

        # Extract data and inputs efficiently
        targets = self._extract_data(coordinates)
        inputs, _, rc_augs = self.genome_dataset[idx]

        # Handle reverse complement efficiently
        if rc_augs[0]:  # If input is reverse complemented
            targets = torch.flip(targets, dims=[1])

        # Transpose inputs once (avoid multiple tensor operations)
        inputs = inputs.permute(1, 0)  # Change to (C, N, L)

        return {
            "input_ids": inputs,
            "label_ids": targets,
        }

    @property
    def n_labels(self) -> int:
        """
        Returns the number of labels (tracks) in the dataset.

        Returns:
            int: The number of tracks.
        """
        return len(self.bigwig_files)

    @property
    def id2label(self) -> dict[int, str]:
        """
        Returns a mapping from track indices to track names.

        Returns:
            Dict[int, str]: A dictionary mapping track indices to track names.
        """
        return {
            i: Path(bigwig_file).stem for i, bigwig_file in enumerate(self.bigwig_files)
        }

    @property
    def label2id(self) -> dict[str, int]:
        """
        Returns a mapping from track names to track indices.

        Returns:
            Dict[str, int]: A dictionary mapping track names to track indices.
        """
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
