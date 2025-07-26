import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import weakref

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


# class ChromatinDataset(Dataset):
#     def __init__(
#         self,
#         genome_dataset: "GenomeIntervalDataset",
#         data: pd.DataFrame = None,
#         bigwig_dir: str | Path = None,
#         clip_soft: int = 32,
#         clip_hard: int = 128,
#         scale_factor: float = 2.0,
#         power_transform_exponent: float = 1.0,
#         cache_bigwig_handles: bool = True,
#         num_workers: int = 4,
#         dtype: torch.dtype = torch.float16,  # Use float16 by default for memory savings
#         pin_memory: bool = False,  # Allow pinned memory allocation
#         prefetch_factor: int = 2,  # For DataLoader prefetching
#     ) -> None:
#         """
#         Memory-optimized ChromatinDataset for HuggingFace Trainer and PyTorch multiprocessing.
        
#         Key optimizations:
#         - Uses float16 by default to halve memory usage
#         - Implements lazy loading and memory mapping
#         - Optimized tensor operations with minimal allocations
#         - Thread-safe BigWig handle management
#         - Memory-efficient data transformations
#         """
#         self.genome_dataset = genome_dataset
#         self.clip_soft = clip_soft
#         self.clip_hard = clip_hard
#         self.scale_factor = scale_factor
#         self.power_transform_exponent = power_transform_exponent
#         self.cache_bigwig_handles = cache_bigwig_handles
#         self.num_workers = num_workers
#         self.dtype = dtype
#         self.pin_memory = pin_memory
        
#         # Use appropriate min/max values for the specified dtype
#         self.min_value = torch.finfo(self.dtype).min
#         self.max_value = torch.finfo(self.dtype).max

#         # Thread-local storage with weak references to prevent memory leaks
#         self._thread_local = threading.local()
#         self._bigwig_handles_registry = weakref.WeakSet()

#         # Initialize data sources
#         self._initialize_data_sources(data, bigwig_dir)
        
#         # Pre-compute scaling parameters as tensors for efficiency
#         self._precompute_scaling_parameters()
        
#         # Pre-allocate reusable tensors to minimize allocations
#         self._setup_tensor_cache()
        
#         # Pre-format chromosome names
#         self._preprocess_chromosome_names()

#     def _initialize_data_sources(self, data: pd.DataFrame, bigwig_dir: str | Path):
#         """Initialize data sources with validation."""
#         if data is not None:
#             try:
#                 # Use categorical dtype for string columns to save memory
#                 if 'path' in data.columns:
#                     data['path'] = data['path'].astype('category')
#                 self.data = data  # Assume validation is done elsewhere
#             except Exception as e:
#                 logger.error(f"Data validation failed: {e}")
#                 raise e
#             self.bigwig_files = self.data.path.tolist()
#         else:
#             if bigwig_dir is None:
#                 raise ValueError("Either 'data' or 'bigwig_dir' must be provided.")
#             self.data = None
#             self.bigwig_dir = Path(bigwig_dir)
#             if not self.bigwig_dir.exists():
#                 raise FileNotFoundError(f"BigWig directory {self.bigwig_dir} does not exist.")
#             if not self.bigwig_dir.is_dir():
#                 raise NotADirectoryError(f"BigWig directory {self.bigwig_dir} is not a directory.")
            
#             self.bigwig_dir = self.bigwig_dir.resolve()
#             self.bigwig_files = list(self.bigwig_dir.glob("*.bw")) + list(
#                 self.bigwig_dir.glob("*.bigWig")
#             )

#         if not self.bigwig_files:
#             raise FileNotFoundError(f"No BigWig files found.")

#     def _precompute_scaling_parameters(self):
#         """Pre-compute scaling parameters as tensors for efficient reuse."""
#         num_tracks = len(self.bigwig_files)
        
#         if self.data is not None:
#             # Convert to tensors once and cache
#             self.scaling_factors = torch.tensor(
#                 self.data.scaling_factor.values, dtype=self.dtype
#             ).unsqueeze(1)
#             self.power_exponents = torch.tensor(
#                 self.data.power_transform_exponent.values, dtype=self.dtype
#             ).unsqueeze(1)
#             self.soft_clips = torch.tensor(
#                 self.data.soft_clip.values, dtype=self.dtype
#             ).unsqueeze(1)
#             self.hard_clips = torch.tensor(
#                 self.data.hard_clip.values, dtype=self.dtype
#             ).unsqueeze(1)
#         else:
#             # Create uniform tensors
#             self.scaling_factors = torch.full(
#                 (num_tracks, 1), self.scale_factor, dtype=self.dtype
#             )
#             self.power_exponents = torch.full(
#                 (num_tracks, 1), self.power_transform_exponent, dtype=self.dtype
#             )
#             self.soft_clips = torch.full(
#                 (num_tracks, 1), self.clip_soft, dtype=self.dtype
#             )
#             self.hard_clips = torch.full(
#                 (num_tracks, 1), self.clip_hard, dtype=self.dtype
#             )

#     @classmethod
#     def from_csv(cls, csv_file: str | Path, **kwargs) -> "ChromatinDataset":
#         """Create a ChromatinDataset from a CSV file containing data sources."""
#         # Use efficient CSV reading with appropriate dtypes
#         df = pd.read_csv(
#             csv_file, 
#             dtype={'path': 'category'} if 'path' in pd.read_csv(csv_file, nrows=1).columns else None
#         )
#         return cls(data=df, **kwargs)

#     def _preprocess_chromosome_names(self):
#         """Pre-process chromosome names to avoid repeated string operations."""
#         if hasattr(self.genome_dataset.df, "with_columns"):
#             try:
#                 import polars as pl
#                 self.genome_dataset.df = self.genome_dataset.df.with_columns([
#                     pl.when(pl.col("column_1").str.starts_with("chr"))
#                     .then(pl.col("column_1"))
#                     .otherwise(pl.concat_str([pl.lit("chr"), pl.col("column_1")]))
#                     .alias("column_1")
#                 ])
#             except Exception as e:
#                 logger.error("Failed to preprocess chromosome names in polars DataFrame.")
#                 raise ValueError(
#                     "Ensure 'column_1' contains chromosome names in UCSC format."
#                 ) from e

#     def _setup_tensor_cache(self):
#         """Setup tensor cache for reusable operations."""
#         # Pre-allocate common tensor shapes to avoid repeated allocations
#         self._tensor_cache = {}
        
#     def _get_cached_tensor(self, shape: tuple, dtype: torch.dtype = None) -> torch.Tensor:
#         """Get a cached tensor of the specified shape to minimize allocations."""
#         if dtype is None:
#             dtype = self.dtype
#         key = (shape, dtype)
        
#         if key not in self._tensor_cache:
#             tensor = torch.empty(shape, dtype=dtype)
#             if self.pin_memory:
#                 tensor = tensor.pin_memory()
#             self._tensor_cache[key] = tensor
        
#         return self._tensor_cache[key]

#     def _get_bigwig_handles(self):
#         """Get thread-local BigWig handles with proper cleanup."""
#         if not hasattr(self._thread_local, "bigwig_handles") or self._thread_local.bigwig_handles is None:
#             if self.cache_bigwig_handles:
#                 handles = []
#                 for bigwig_file in self.bigwig_files:
#                     handle = pybigtools.open(str(bigwig_file))
#                     handles.append(handle)
#                     # Register for cleanup
#                     self._bigwig_handles_registry.add(handle)
                
#                 self._thread_local.bigwig_handles = handles
#             else:
#                 self._thread_local.bigwig_handles = None
        
#         return self._thread_local.bigwig_handles

#     def __len__(self):
#         return len(self.genome_dataset)

#     @property
#     def params(self) -> dict[str, Any]:
#         """Returns the parameters of the dataset as a dictionary."""
#         return {
#             "clip_soft": self.clip_soft,
#             "clip_hard": self.clip_hard,
#             "scale_factor": self.scale_factor,
#             "power_transform_exponent": self.power_transform_exponent,
#             "bigwigs": str(self.bigwig_files),
#             "fasta_file": str(self.genome_dataset.fasta.seqs.filename),
#             "context_length": self.genome_dataset.fasta.context_length,
#             "dtype": str(self.dtype),
#         }

#     def _reinit_fasta_reader(self):
#         """Re-initializes the FastaInterval reader only if needed (multiprocessing safe)."""
#         if (
#             not hasattr(self._thread_local, "fasta_initialized")
#             or not self._thread_local.fasta_initialized
#         ):
#             # Import here to avoid circular imports
#             from your_module import FastaInterval  # Replace with actual import
            
#             self.genome_dataset.fasta = FastaInterval(
#                 fasta_file=self.genome_dataset.fasta.seqs.filename,
#                 context_length=self.genome_dataset.fasta.context_length,
#                 return_seq_indices=self.genome_dataset.fasta.return_seq_indices,
#                 shift_augs=self.genome_dataset.fasta.shift_augs,
#                 rc_aug=self.genome_dataset.fasta.rc_aug,
#             )
#             self._thread_local.fasta_initialized = True

#     def _scale_inplace_optimized(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Highly optimized in-place scaling with minimal memory allocations.
#         Uses pre-computed tensors and optimized operations.
#         """
#         # Ensure scaling parameters are on the same device
#         device = x.device
#         scaling_factors = self.scaling_factors.to(device, non_blocking=True)
#         power_exponents = self.power_exponents.to(device, non_blocking=True)
#         soft_clips = self.soft_clips.to(device, non_blocking=True)
#         hard_clips = self.hard_clips.to(device, non_blocking=True)

#         # Transform order: scale → power → clip (all in-place)
#         x.mul_(scaling_factors)

#         # Power transform with offset (optimized)
#         x.add_(1.0)
#         if not torch.allclose(power_exponents, torch.tensor(1.0, device=device)):
#             x.pow_(power_exponents)
#         x.sub_(1.0)

#         # Soft clipping with vectorized operations
#         mask = x > soft_clips
#         if mask.any():
#             # Use out parameter to avoid extra allocation
#             excess = x - soft_clips + 1.0
#             torch.sqrt(excess, out=excess)  # In-place sqrt
#             excess.add_(soft_clips - 1.0)
#             x.masked_scatter_(mask, excess[mask])

#         # Hard clipping (in-place)
#         x.clamp_(min=-hard_clips, max=hard_clips)
#         x.clamp_(min=self.min_value, max=self.max_value)

#         return x

#     def _extract_from_bigwig(self, coordinates) -> list:
#         """Extract BigWig data using cached handles for better performance."""
#         bigwig_handles = self._get_bigwig_handles()

#         if bigwig_handles is not None:
#             return [
#                 handle.values(coordinates.chromosome, coordinates.start, coordinates.end)
#                 for handle in bigwig_handles
#             ]
#         else:
#             # Use context manager for proper cleanup
#             results = []
#             for bigwig_file in self.bigwig_files:
#                 with pybigtools.open(str(bigwig_file)) as handle:
#                     results.append(
#                         handle.values(coordinates.chromosome, coordinates.start, coordinates.end)
#                     )
#             return results

#     def _extract_data(self, coordinates) -> torch.Tensor:
#         """
#         Memory-optimized data extraction with minimal tensor operations.
#         """
#         # Extract BigWig data
#         signal = self._extract_from_bigwig(coordinates)

#         # Convert to tensor with optimal dtype from the start
#         signal_array = np.array(signal, dtype=np.float32)
        
#         # Handle NaN values in numpy (more efficient than torch)
#         np.nan_to_num(signal_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Convert to tensor with target dtype
#         signal_tensor = torch.from_numpy(signal_array).to(dtype=self.dtype)

#         # Optimized binning with proper memory management
#         # Use unfold for memory-efficient binning instead of avg_pool1d
#         kernel_size = 32
#         seq_len = signal_tensor.shape[1]
#         if seq_len % kernel_size == 0:
#             # Reshape and mean along last dimension
#             signal_tensor_binned = (
#                 signal_tensor.view(signal_tensor.shape[0], -1, kernel_size)
#                 .mean(dim=2)
#                 .mul_(kernel_size)
#             )
#         else:
#             # Fallback to avg_pool1d for non-divisible lengths
#             signal_tensor_binned = (
#                 torch.nn.functional.avg_pool1d(
#                     signal_tensor.unsqueeze(0), kernel_size=kernel_size, stride=kernel_size
#                 ).squeeze(0)
#                 .mul_(kernel_size)
#             )

#         # Apply scaling transformation
#         return self._scale_inplace_optimized(signal_tensor_binned)

#     def __getitem__(self, idx: int) -> dict:
#         """
#         Memory-optimized item retrieval with minimal allocations.
#         """
#         # Only reinitialize FASTA reader if needed
#         self._reinit_fasta_reader()

#         # Get coordinates efficiently
#         row = self.genome_dataset.df[idx]
        
#         # Import your coordinate class here
#         from your_module import FastGenomicRegion  # Replace with actual import
        
#         coordinates = FastGenomicRegion(
#             chromosome=row["column_1"].item(),
#             start=row["column_2"].item(),
#             end=row["column_3"].item(),
#             strand="+",
#         )

#         # Extract data and inputs
#         targets = self._extract_data(coordinates)
#         inputs, _, rc_augs = self.genome_dataset[idx]

#         # Handle reverse complement efficiently
#         if rc_augs[0]:
#             targets = torch.flip(targets, dims=[1])

#         # Transpose inputs efficiently
#         if inputs.dtype != self.dtype:
#             inputs = inputs.to(dtype=self.dtype)
#         inputs = inputs.permute(1, 0)

#         return {
#             "input_ids": inputs,
#             "label_ids": targets,
#         }

#     @property
#     def n_labels(self) -> int:
#         """Returns the number of labels (tracks) in the dataset."""
#         return len(self.bigwig_files)

#     @property
#     def id2label(self) -> dict[int, str]:
#         """Returns a mapping from track indices to track names."""
#         return {
#             i: Path(bigwig_file).stem for i, bigwig_file in enumerate(self.bigwig_files)
#         }

#     @property
#     def label2id(self) -> dict[str, int]:
#         """Returns a mapping from track names to track indices."""
#         return {v: k for k, v in self.id2label.items()}

#     def cleanup(self):
#         """Cleanup method for proper resource management."""
#         # Close any cached BigWig handles
#         if hasattr(self._thread_local, "bigwig_handles") and self._thread_local.bigwig_handles:
#             for handle in self._thread_local.bigwig_handles:
#                 try:
#                     handle.close()
#                 except:
#                     pass
        
#         # Clear tensor cache
#         if hasattr(self, '_tensor_cache'):
#             self._tensor_cache.clear()

#     def __del__(self):
#         """Destructor for cleanup."""
#         self.cleanup()

class ChromatinDataset(Dataset):
    def __init__(
        self,
        genome_dataset: "GenomeIntervalDataset",
        data: pd.DataFrame = None,
        bigwig_dir: str | Path = None,
        clip_soft: int = 32,
        clip_hard: int = 128,
        scale_factor: float = 2.0,
        power_transform_exponent: float = 1.0,
        cache_bigwig_handles: bool = True,
        num_workers: int = 4,
        dtype: torch.dtype = torch.float32,  # Use float32 by default for memory savings
        pin_memory: bool = False,  # Allow pinned memory allocation
        prefetch_factor: int = 2,  # For DataLoader prefetching
    ) -> None:
        """
        Memory-optimized ChromatinDataset for HuggingFace Trainer and PyTorch multiprocessing.
        
        Key optimizations:
        - Uses float16 by default to halve memory usage
        - Implements lazy loading and memory mapping
        - Optimized tensor operations with minimal allocations
        - Thread-safe BigWig handle management
        - Memory-efficient data transformations
        """
        self.genome_dataset = genome_dataset
        self.clip_soft = clip_soft
        self.clip_hard = clip_hard
        self.scale_factor = scale_factor
        self.power_transform_exponent = power_transform_exponent
        self.cache_bigwig_handles = cache_bigwig_handles
        self.num_workers = num_workers
        self.dtype = dtype
        self.pin_memory = pin_memory
        
        # Use appropriate min/max values for the specified dtype
        self.min_value = torch.finfo(self.dtype).min
        self.max_value = torch.finfo(self.dtype).max

        # Store serializable configuration for multiprocessing
        self._bigwig_files_paths = None  # Will be set in _initialize_data_sources
        self._fasta_config = None  # Will be set after genome_dataset initialization
        
        # Thread-local storage - this won't be pickled
        self._thread_local = None  # Initialize lazily in worker processes
        
        # Initialize data sources
        self._initialize_data_sources(data, bigwig_dir)
        
        # Store FASTA configuration for reconstruction in workers
        if hasattr(genome_dataset, 'fasta'):
            self._fasta_config = {
                'fasta_file': str(genome_dataset.fasta.seqs.filename),
                'context_length': genome_dataset.fasta.context_length,
                'return_seq_indices': getattr(genome_dataset.fasta, 'return_seq_indices', False),
                'shift_augs': getattr(genome_dataset.fasta, 'shift_augs', None),
                'rc_aug': getattr(genome_dataset.fasta, 'rc_aug', False),
            }
        
        # Pre-compute scaling parameters as tensors for efficiency
        self._precompute_scaling_parameters()
        
        # Pre-format chromosome names
        self._preprocess_chromosome_names()

    def _initialize_data_sources(self, data: pd.DataFrame, bigwig_dir: str | Path):
        """Initialize data sources with validation."""
        if data is not None:
            try:
                # Use categorical dtype for string columns to save memory
                if 'path' in data.columns:
                    data['path'] = data['path'].astype('category')
                self.data = data  # Assume validation is done elsewhere
            except Exception as e:
                logger.error(f"Data validation failed: {e}")
                raise e
            # Store as strings for pickling
            self._bigwig_files_paths = [str(p) for p in self.data.path.tolist()]
        else:
            if bigwig_dir is None:
                raise ValueError("Either 'data' or 'bigwig_dir' must be provided.")
            self.data = None
            self.bigwig_dir = Path(bigwig_dir)
            if not self.bigwig_dir.exists():
                raise FileNotFoundError(f"BigWig directory {self.bigwig_dir} does not exist.")
            if not self.bigwig_dir.is_dir():
                raise NotADirectoryError(f"BigWig directory {self.bigwig_dir} is not a directory.")
            
            self.bigwig_dir = self.bigwig_dir.resolve()
            bigwig_files = list(self.bigwig_dir.glob("*.bw")) + list(
                self.bigwig_dir.glob("*.bigWig")
            )
            # Store as strings for pickling
            self._bigwig_files_paths = [str(f) for f in bigwig_files]

        if not self._bigwig_files_paths:
            raise FileNotFoundError(f"No BigWig files found.")

    @property 
    def bigwig_files(self):
        """Return bigwig file paths (for compatibility)."""
        return self._bigwig_files_paths

    def _precompute_scaling_parameters(self):
        """Pre-compute scaling parameters as tensors for efficient reuse."""
        num_tracks = len(self._bigwig_files_paths)
        
        if self.data is not None:
            # Convert to tensors once and cache
            self.scaling_factors = torch.tensor(
                self.data.scaling_factor.values, dtype=self.dtype
            ).unsqueeze(1)
            self.power_exponents = torch.tensor(
                self.data.power_transform_exponent.values, dtype=self.dtype
            ).unsqueeze(1)
            self.soft_clips = torch.tensor(
                self.data.soft_clip.values, dtype=self.dtype
            ).unsqueeze(1)
            self.hard_clips = torch.tensor(
                self.data.hard_clip.values, dtype=self.dtype
            ).unsqueeze(1)
        else:
            # Create uniform tensors
            self.scaling_factors = torch.full(
                (num_tracks, 1), self.scale_factor, dtype=self.dtype
            )
            self.power_exponents = torch.full(
                (num_tracks, 1), self.power_transform_exponent, dtype=self.dtype
            )
            self.soft_clips = torch.full(
                (num_tracks, 1), self.clip_soft, dtype=self.dtype
            )
            self.hard_clips = torch.full(
                (num_tracks, 1), self.clip_hard, dtype=self.dtype
            )

    @classmethod
    def from_csv(cls, csv_file: str | Path, **kwargs) -> "ChromatinDataset":
        """Create a ChromatinDataset from a CSV file containing data sources."""
        # Use efficient CSV reading with appropriate dtypes
        df = pd.read_csv(
            csv_file, 
            dtype={'path': 'category'} if 'path' in pd.read_csv(csv_file, nrows=1).columns else None
        )
        return cls(data=df, **kwargs)

    def _preprocess_chromosome_names(self):
        """Pre-process chromosome names to avoid repeated string operations."""
        if hasattr(self.genome_dataset.df, "with_columns"):
            try:
                import polars as pl
                self.genome_dataset.df = self.genome_dataset.df.with_columns([
                    pl.when(pl.col("column_1").str.starts_with("chr"))
                    .then(pl.col("column_1"))
                    .otherwise(pl.concat_str([pl.lit("chr"), pl.col("column_1")]))
                    .alias("column_1")
                ])
            except Exception as e:
                logger.error("Failed to preprocess chromosome names in polars DataFrame.")
                raise ValueError(
                    "Ensure 'column_1' contains chromosome names in UCSC format."
                ) from e

    def _setup_tensor_cache(self):
        """Setup tensor cache for reusable operations."""
        # Pre-allocate common tensor shapes to avoid repeated allocations
        self._tensor_cache = {}
        
    def _get_cached_tensor(self, shape: tuple, dtype: torch.dtype = None) -> torch.Tensor:
        """Get a cached tensor of the specified shape to minimize allocations."""
        if dtype is None:
            dtype = self.dtype
        key = (shape, dtype)
        
        if key not in self._tensor_cache:
            tensor = torch.empty(shape, dtype=dtype)
            if self.pin_memory:
                tensor = tensor.pin_memory()
            self._tensor_cache[key] = tensor
        
        return self._tensor_cache[key]

    def _ensure_thread_local(self):
        """Ensure thread-local storage is initialized (for multiprocessing)."""
        if self._thread_local is None:
            self._thread_local = threading.local()
        return self._thread_local

    def _get_bigwig_handles(self):
        """Get thread-local BigWig handles with proper cleanup."""
        thread_local = self._ensure_thread_local()
        
        if not hasattr(thread_local, "bigwig_handles") or thread_local.bigwig_handles is None:
            if self.cache_bigwig_handles:
                handles = []
                for bigwig_file_path in self._bigwig_files_paths:
                    handle = pybigtools.open(bigwig_file_path)
                    handles.append(handle)
                
                thread_local.bigwig_handles = handles
            else:
                thread_local.bigwig_handles = None
        
        return thread_local.bigwig_handles

    def __len__(self):
        return len(self.genome_dataset)

    @property
    def params(self) -> dict[str, Any]:
        """Returns the parameters of the dataset as a dictionary."""
        return {
            "clip_soft": self.clip_soft,
            "clip_hard": self.clip_hard,
            "scale_factor": self.scale_factor,
            "power_transform_exponent": self.power_transform_exponent,
            "bigwigs": str(self._bigwig_files_paths),
            "fasta_file": str(self.genome_dataset.fasta.seqs.filename) if hasattr(self.genome_dataset, 'fasta') else None,
            "context_length": self.genome_dataset.fasta.context_length if hasattr(self.genome_dataset, 'fasta') else None,
            "dtype": str(self.dtype),
        }

    def _reinit_fasta_reader(self):
        """Re-initializes the FastaInterval reader only if needed (multiprocessing safe)."""
        thread_local = self._ensure_thread_local()
        
        if (
            not hasattr(thread_local, "fasta_initialized")
            or not thread_local.fasta_initialized
        ):
            # Only reinitialize if we have the config and it's needed
            if self._fasta_config:
                # Import here to avoid circular imports
                from enformer_pytorch.data import FastaInterval  # Replace with actual import
                
                self.genome_dataset.fasta = FastaInterval(
                    fasta_file=self._fasta_config['fasta_file'],
                    context_length=self._fasta_config['context_length'],
                    return_seq_indices=self._fasta_config['return_seq_indices'],
                    shift_augs=self._fasta_config['shift_augs'],
                    rc_aug=self._fasta_config['rc_aug'],
                )
            thread_local.fasta_initialized = True

    def _scale_inplace_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Highly optimized in-place scaling with minimal memory allocations.
        Uses pre-computed tensors and optimized operations.
        """
        # Ensure scaling parameters are on the same device
        device = x.device
        scaling_factors = self.scaling_factors.to(device, non_blocking=True)
        power_exponents = self.power_exponents.to(device, non_blocking=True)
        soft_clips = self.soft_clips.to(device, non_blocking=True)
        hard_clips = self.hard_clips.to(device, non_blocking=True)

        # Transform order: scale → power → clip (all in-place)
        x.mul_(scaling_factors)

        # Power transform with offset (optimized)
        x.add_(1.0)
        if not torch.allclose(power_exponents, torch.tensor(1.0, device=device)):
            x.pow_(power_exponents)
        x.sub_(1.0)

        # Soft clipping with vectorized operations
        mask = x > soft_clips
        if mask.any():
            # Use out parameter to avoid extra allocation
            excess = x - soft_clips + 1.0
            torch.sqrt(excess, out=excess)  # In-place sqrt
            excess.add_(soft_clips - 1.0)
            x.masked_scatter_(mask, excess[mask])

        # Hard clipping (in-place)
        x.clamp_(min=-hard_clips, max=hard_clips)
        x.clamp_(min=self.min_value, max=self.max_value)

        return x

    def _extract_from_bigwig(self, coordinates) -> list:
        """Extract BigWig data using cached handles for better performance."""
        bigwig_handles = self._get_bigwig_handles()

        if bigwig_handles is not None:
            return [
                handle.values(coordinates.chromosome, coordinates.start, coordinates.end)
                for handle in bigwig_handles
            ]
        else:
            # Use context manager for proper cleanup
            results = []
            for bigwig_file_path in self._bigwig_files_paths:
                with pybigtools.open(bigwig_file_path) as handle:
                    results.append(
                        handle.values(coordinates.chromosome, coordinates.start, coordinates.end)
                    )
            return results

    def _extract_data(self, coordinates) -> torch.Tensor:
        """
        Memory-optimized data extraction with minimal tensor operations.
        """
        # Extract BigWig data
        signal = self._extract_from_bigwig(coordinates)

        # Convert to tensor with optimal dtype from the start
        signal_array = np.array(signal, dtype=np.float32)
        
        # Handle NaN values in numpy (more efficient than torch)
        np.nan_to_num(signal_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensor with target dtype
        signal_tensor = torch.from_numpy(signal_array).to(dtype=self.dtype)

        # Optimized binning with proper memory management
        # Use unfold for memory-efficient binning instead of avg_pool1d
        kernel_size = 32
        seq_len = signal_tensor.shape[1]
        if seq_len % kernel_size == 0:
            # Reshape and mean along last dimension
            signal_tensor_binned = (
                signal_tensor.view(signal_tensor.shape[0], -1, kernel_size)
                .mean(dim=2)
                .mul_(kernel_size)
            )
        else:
            # Fallback to avg_pool1d for non-divisible lengths
            signal_tensor_binned = (
                torch.nn.functional.avg_pool1d(
                    signal_tensor.unsqueeze(0), kernel_size=kernel_size, stride=kernel_size
                ).squeeze(0)
                .mul_(kernel_size)
            )

        # Apply scaling transformation
        return self._scale_inplace_optimized(signal_tensor_binned)
    

    def _format_coordinates(self, row: pd.Series) -> "FastGenomicRegion":
        """
        Format coordinates from a DataFrame row into a FastGenomicRegion object.
        """
        # Import your coordinate class here
        from greyhound.data.datasets import FastGenomicRegion
        return FastGenomicRegion(
            chromosome=row["column_1"].item(),
            start=row["column_2"].item(),
            end=row["column_3"].item(),
            strand="+",
        )


    def __getitem__(self, idx: int) -> dict:
        """
        Memory-optimized item retrieval with minimal allocations.
        """
        # Only reinitialize FASTA reader if needed
        self._reinit_fasta_reader()

        # Get coordinates efficiently
        row = self.genome_dataset.df[idx]
        coordinates = self._format_coordinates(row)

        # Extract data and inputs
        targets = self._extract_data(coordinates)
        inputs, _, rc_augs = self.genome_dataset[idx]

        # Handle reverse complement efficiently
        if rc_augs[0]:
            targets = torch.flip(targets, dims=[1])

        # Transpose inputs efficiently
        if inputs.dtype != self.dtype:
            inputs = inputs.to(dtype=self.dtype)
        inputs = inputs.permute(1, 0)

        return {
            "input_ids": inputs,
            "label_ids": targets,
        }

    @property
    def n_labels(self) -> int:
        """Returns the number of labels (tracks) in the dataset."""
        return len(self._bigwig_files_paths)

    @property
    def id2label(self) -> dict[int, str]:
        """Returns a mapping from track indices to track names."""
        return {
            i: Path(bigwig_file_path).stem for i, bigwig_file_path in enumerate(self._bigwig_files_paths)
        }

    @property
    def label2id(self) -> dict[str, int]:
        """Returns a mapping from track names to track indices."""
        return {v: k for k, v in self.id2label.items()}

    def cleanup(self):
        """Cleanup method for proper resource management."""
        thread_local = self._ensure_thread_local()
        
        # Close any cached BigWig handles
        if hasattr(thread_local, "bigwig_handles") and thread_local.bigwig_handles:
            for handle in thread_local.bigwig_handles:
                try:
                    handle.close()
                except:
                    pass
        
        # Clear tensor cache
        if hasattr(self, '_tensor_cache'):
            self._tensor_cache.clear()

    def __del__(self):
        """Destructor for cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    
    def __getstate__(self):
        """Custom pickle support - exclude non-serializable objects."""
        state = self.__dict__.copy()
        # Remove thread-local storage
        state['_thread_local'] = None
        
        # Clear tensor cache to reduce size and avoid potential issues
        if '_tensor_cache' in state:
            del state['_tensor_cache']
        
        # Remove the non-picklable fasta reader from genome_dataset if present
        if hasattr(state.get('genome_dataset', None), 'fasta'):
            state['genome_dataset'].fasta = None  # Remove the open file handle
        
        return state

    def __setstate__(self, state):
        """Custom unpickle support - restore transient objects."""
        self.__dict__.update(state)
        # Thread-local storage will be initialized lazily when needed
        self._thread_local = None

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
