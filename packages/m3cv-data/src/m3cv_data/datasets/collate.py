"""Custom collate functions for batching medical imaging data."""

from __future__ import annotations

import torch


def patient_collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard collate function for patient data batches.

    Takes a list of (volume, label) tuples and returns batched tensors.

    Args:
        batch: List of (volume, label) tuples where volumes have shape
            (C, Z, Y, X) and labels are integers.

    Returns:
        Tuple of (volumes, labels) where volumes has shape (B, C, Z, Y, X)
        and labels has shape (B,).

    Raises:
        ValueError: If volumes in the batch have inconsistent shapes.
    """
    volumes, labels = zip(*batch, strict=True)

    # Verify all volumes have the same shape
    shapes = [v.shape for v in volumes]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"All volumes in batch must have the same shape. Got shapes: {shapes}"
        )

    # Stack into batched tensors
    batched_volumes = torch.stack(volumes, dim=0)
    batched_labels = torch.tensor(labels, dtype=torch.long)

    return batched_volumes, batched_labels
