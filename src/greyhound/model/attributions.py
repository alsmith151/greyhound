from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger

from greyhound.data import GenomicRegion


def gradient_x_input(
    model: Callable, X: torch.Tensor, target: int, bins: list[int] = None
) -> torch.Tensor:
    """
    Computes Gradient x Input attributions for a single input and target class.

    Args:
        model (torch.nn.Module): model that outputs shape (1, C, B)
        X (torch.Tensor): input tensor of shape (1, 4, L), one-hot encoded
        target (int): the target class index C
        bins (List[int], optional): list of bin indices to consider for attribution

    Returns:
        torch.Tensor: attributions of shape (1, 4, L)
    """
    X = X.requires_grad_(True)
    model.zero_grad()

    with torch.amp.autocast("cuda"):
        out = model(X)  # shape: (1, C, B)
        # If output is (1, C, B), sum over bins to get a scalar for the target class
        if bins is not None:
            out_target = out[0, target, bins].sum()  # scalar
        else:
            out_target = out[0, target, :].sum()  # scalar

        grads = torch.autograd.grad(
            outputs=out_target,
            inputs=X,
            grad_outputs=torch.ones_like(out_target),
            retain_graph=False,
            create_graph=False,
        )[
            0
        ]  # shape: (1, 4, L)

    grad_x_input = grads * X  # (1, 4, L)
    return grad_x_input


class GenomicConverter:
    """
    A class for converting LongBoi model outputs and attributions to genomic formats,
    particularly BEDGRAPH format, and handling genomic coordinate transformations.
    """

    def __init__(self, genome_interval_dataset, default_bin_size: int = 32):
        """
        Initialize the converter with a genome interval dataset.

        Args:
            genome_interval_dataset: The GenomeIntervalDataset object
            default_bin_size (int): Default bin size for binned operations
        """
        self.gid = genome_interval_dataset
        self.default_bin_size = default_bin_size

    def predictions_to_bedgraph(
        self, tensor: torch.Tensor, index: int, bin_size: int = None
    ) -> pd.DataFrame:
        """
        Convert a tensor output from LongBoi to a DataFrame suitable for BEDGRAPH format.

        Args:
            tensor (torch.Tensor): The output tensor from LongBoi, shape (1, N, L).
            chrom (str): The chromosome name.
            index (int): The index of the sequence of the genome interval dataset used for prediction.
            bin_size (int, optional): The bin size. Uses default_bin_size if not provided.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Chromosome', 'Start', 'End', 'Score'.
        """
        if bin_size is None:
            bin_size = self.default_bin_size

        # Ensure tensor is on CPU and convert to numpy
        tensor = tensor.cpu().detach().numpy().squeeze()  # shape (1, N, L) -> (N, L)

        # Get the start and end positions from the genome interval dataset
        chrom = self.gid.df[index]["column_1"].item()
        roi_start = self.gid.df[index]["column_2"].item()
        roi_end = self.gid.df[index]["column_3"].item()

        bins = np.arange(roi_start, roi_end, bin_size)

        # Ensure tensor matches the number of bins
        if tensor.ndim > 1:
            # If tensor is 2D (N, L), we need to reshape or select appropriate data
            # For predictions, we typically want to flatten or select a specific track
            tensor = tensor.flatten()

        # Ensure the tensor length matches the number of bins
        if len(tensor) != len(bins):
            # If tensor is longer, truncate to match bins
            if len(tensor) > len(bins):
                tensor = tensor[:len(bins)]
            # If tensor is shorter, this indicates a mismatch in expected data
            else:
                raise ValueError(
                    f"Tensor length ({len(tensor)}) is shorter than number of bins ({len(bins)}). "
                    f"Expected tensor to have at least {len(bins)} elements."
                )

        df = pd.DataFrame(
            {
                "Chromosome": np.repeat(chrom, bins.size),
                "Start": bins,
                "End": bins + bin_size,
                "Score": tensor,
            }
        )

        return df

    def attributions_to_bedgraph(
        self,
        attributions: torch.Tensor,
        chrom: str,
        index: int,
    ) -> pd.DataFrame:
        """
        Convert attributions from LongBoi to a DataFrame suitable for BEDGRAPH format.

        Args:
            attributions (torch.Tensor): The attributions tensor, shape (1, L).
                                       Need to have combined the attributions for all basepairs.
            chrom (str): The chromosome name.
            index (int): The index of the sequence of the genome interval dataset used for prediction.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Chromosome', 'Start', 'End', 'Score'.
        """
        # Ensure attributions are on CPU and convert to numpy
        attributions = attributions.cpu().numpy().squeeze()  # shape (1, L) -> (L,)
        if attributions.ndim != 1:
            logger.error(
                "Attributions should be a 1D tensor after combining basepair attributions."
            )
            raise ValueError(
                "Attributions should be a 1D tensor after combining basepair attributions."
            )

        # Get the start and end positions from the genome interval dataset
        roi_start = self.gid.df[index]["column_2"].item()
        roi_end = self.gid.df[index]["column_3"].item()
        context_length = self.gid.fasta.context_length

        # Calculate the actual start and end positions
        # The datasets given are just the region of interest, so we need to adjust for the context length
        supplied_region_length = roi_end - roi_start
        flanked_region_length = (context_length - supplied_region_length) // 2

        # Adjust the start and end positions to include the context length
        roi_start_actual = roi_start - flanked_region_length
        roi_end_actual = roi_end + flanked_region_length

        # Create bins for the BEDGRAPH format
        bins = np.arange(roi_start_actual, roi_end_actual, 1)
        df = pd.DataFrame(
            {
                "Chromosome": np.repeat(chrom, bins.size),
                "Start": bins,
                "End": bins + 1,
                "Score": attributions,
            }
        )
        return df

    def find_region_in_bin_space(
        self,
        region: GenomicRegion,
        bin_size: int = None,
    ) -> dict[int, list[int]]:
        """
        Find the indices of the bins that overlap with a given genomic region.

        Args:
            region (GenomicRegion): The genomic region to find.
            bin_size (int, optional): The size of the bins. Uses default_bin_size if not provided.

        Returns:
            Dict[int, List[int]]: A dictionary mapping the index of the overlapping region
                                to a list of bin indices.
        """
        if bin_size is None:
            bin_size = self.default_bin_size

        # Find regions in the genome interval dataset that overlap with the given region
        overlapping_regions = self.gid.df.with_row_index().filter(
            (pl.col("column_1") == region.chromosome)
            & (pl.col("column_2") < region.end)  # region start < query end
            & (pl.col("column_3") > region.start)  # region end > query start
        )

        # Get the indices of the overlapping regions
        overlapping_indices = overlapping_regions["index"].to_list()

        # For the overlapping regions, find the bins that overlap with the given region
        bin_indices = defaultdict(list)
        for index in overlapping_indices:
            roi_start = self.gid.df[index]["column_2"].item()
            roi_end = self.gid.df[index]["column_3"].item()

            # Where does the supplied region start and end in the interval?
            start_in_interval = max(region.start, roi_start)
            end_in_interval = min(region.end, roi_end)
            # Calculate the start and end bin indices
            start_bin = (start_in_interval - roi_start) // bin_size
            end_bin = (end_in_interval - roi_start) // bin_size
            # Add the bin indices to the dictionary
            for bin_index in range(start_bin, end_bin + 1):
                bin_indices[index].append(bin_index)
        # If no overlapping regions found, return an empty dictionary
        if not bin_indices:
            logger.warning(
                f"No overlapping regions found for {region.chromosome}:{region.start}-{region.end}"
            )
            return {}

        return bin_indices

    def convert_attribution_region_to_genomic_coordinates(
        self,
        chrom: str,
        start: int,
        end: int,
        index: int,
    ) -> GenomicRegion:
        """
        Convert a region defined by start and end positions in the genome interval dataset
        to genomic coordinates.

        Args:
            chrom (str): The chromosome name.
            start (int): The start position of the region in the genome interval dataset.
            end (int): The end position of the region in the genome interval dataset.
            index (int): The index of the sequence in the genome interval dataset.

        Returns:
            GenomicRegion: A GenomicRegion object with the converted coordinates.
        """
        roi_start = self.gid.df[index]["column_2"].item()
        roi_end = self.gid.df[index]["column_3"].item()

        # Adjust the start and end positions to include the context length
        context_length = self.gid.fasta.context_length
        flanked_region_length = (context_length - (roi_end - roi_start)) // 2
        adjusted_start = roi_start - flanked_region_length + start
        adjusted_end = roi_start - flanked_region_length + end

        return GenomicRegion(chrom=chrom, start=adjusted_start, end=adjusted_end)

    def get_context_adjusted_coordinates(self, index: int) -> tuple[int, int]:
        """
        Get the context-adjusted start and end coordinates for a given index.

        Args:
            index (int): The index of the sequence in the genome interval dataset.

        Returns:
            tuple[int, int]: A tuple containing (adjusted_start, adjusted_end)
        """
        roi_start = self.gid.df[index]["column_2"].item()
        roi_end = self.gid.df[index]["column_3"].item()
        context_length = self.gid.fasta.context_length

        supplied_region_length = roi_end - roi_start
        flanked_region_length = (context_length - supplied_region_length) // 2

        adjusted_start = roi_start - flanked_region_length
        adjusted_end = roi_end + flanked_region_length

        return adjusted_start, adjusted_end
