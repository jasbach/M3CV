#!/usr/bin/env python3
"""Demo training script for ResNet3D using HDF5 data from m3cv-dataprep.

This script demonstrates:
1. Loading volumetric data from HDF5 files packed by m3cv-prep
2. Building a ResNet3D model with optional tabular fusion
3. Running a simple training loop to verify gradients flow

Usage:
    uv run python packages/m3cv-models/examples/demo_training.py --data path/to/packed.h5
    uv run python packages/m3cv-models/examples/demo_training.py --synthetic

"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from m3cv_models import FusionConfig, FusionPoint, ResNet3DBuilder


def load_hdf5_data(
    hdf5_path: Path,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load volumetric data from HDF5 file(s).

    Args:
        hdf5_path: Path to HDF5 file or directory containing HDF5 files from m3cv-prep.
        max_samples: Maximum number of samples to load (None = load all).

    Returns:
        Tuple of (volumes, labels) tensors.
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py required for HDF5 loading. Install with: pip install h5py"
        )

    # Check if path is a directory or file
    if hdf5_path.is_dir():
        # Load all .h5 files from directory
        h5_files = sorted(hdf5_path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No .h5 files found in directory: {hdf5_path}")
        print(f"Found {len(h5_files)} HDF5 file(s) in {hdf5_path}")
    else:
        # Single file
        h5_files = [hdf5_path]

    all_volumes = []
    for h5_file in h5_files:
        volumes = _load_single_hdf5(h5_file)
        all_volumes.extend(volumes)

        # Check if we've hit the max sample limit
        if max_samples is not None and len(all_volumes) >= max_samples:
            all_volumes = all_volumes[:max_samples]
            break

    if not all_volumes:
        raise ValueError(f"No volumetric data found in {hdf5_path}")

    # Check if all volumes have the same shape
    shapes = [v.shape for v in all_volumes]
    unique_shapes = list(set(shapes))

    if len(unique_shapes) > 1:
        print(f"[Warning] Volumes have different shapes: {unique_shapes}")
        print("Finding common bounding box and cropping/padding...")

        # Find minimum dimensions that can fit all volumes
        target_shape = _get_common_shape(all_volumes)
        print(f"Target shape: {target_shape}")

        # Crop/pad all volumes to target shape
        all_volumes = [_crop_or_pad_volume(v, target_shape) for v in all_volumes]

    # Stack all volumes and create synthetic labels
    volumes_arr = np.stack(all_volumes, axis=0)
    volumes_tensor = torch.from_numpy(volumes_arr).float()

    # Create synthetic binary labels (randomly assigned for demo purposes)
    labels = torch.randint(0, 2, (len(all_volumes),))

    return volumes_tensor, labels


def _get_common_shape(volumes: list[np.ndarray]) -> tuple[int, ...]:
    """Find a common shape for all volumes (minimum along each dimension).

    Args:
        volumes: List of volume arrays with shape [C, D, H, W].

    Returns:
        Common shape tuple (C, D, H, W).
    """
    # Use minimum dimensions to avoid excessive memory for demo
    min_shape = volumes[0].shape
    for v in volumes[1:]:
        min_shape = tuple(min(a, b) for a, b in zip(min_shape, v.shape, strict=False))
    return min_shape


def _crop_or_pad_volume(
    volume: np.ndarray, target_shape: tuple[int, ...]
) -> np.ndarray:
    """Crop or pad a volume to match target shape.

    Args:
        volume: Input volume with shape [C, D, H, W].
        target_shape: Target shape (C, D, H, W).

    Returns:
        Volume cropped/padded to target shape.
    """
    # For simplicity, center crop if larger, zero-pad if smaller
    result = np.zeros(target_shape, dtype=volume.dtype)

    # Calculate crop/pad offsets for each dimension
    slices_src = []
    slices_dst = []

    for src_size, tgt_size in zip(volume.shape, target_shape, strict=False):
        if src_size >= tgt_size:
            # Crop: take center portion
            start = (src_size - tgt_size) // 2
            slices_src.append(slice(start, start + tgt_size))
            slices_dst.append(slice(None))
        else:
            # Pad: place in center
            start = (tgt_size - src_size) // 2
            slices_src.append(slice(None))
            slices_dst.append(slice(start, start + src_size))

    result[tuple(slices_dst)] = volume[tuple(slices_src)]
    return result


def _load_single_hdf5(hdf5_path: Path) -> list[np.ndarray]:
    """Load volumes from a single HDF5 file.

    Each HDF5 file from m3cv-prep contains one patient with ct/dose at root level.

    Args:
        hdf5_path: Path to HDF5 file.

    Returns:
        List of volume arrays (each with shape [C, D, H, W]).
        Note: Returns a list with one element per file (one patient per file).
    """
    with h5py.File(hdf5_path, "r") as f:
        # m3cv-prep saves ct and dose as direct datasets at root level
        channels = []

        if "ct" in f:
            ct = np.array(f["ct"])
            channels.append(ct)

        if "dose" in f:
            dose = np.array(f["dose"])
            channels.append(dose)

        if not channels:
            raise ValueError(
                f"No ct or dose data found in {hdf5_path}. "
                f"Available keys: {list(f.keys())}"
            )

        # Stack channels: [C, D, H, W] where C is number of modalities
        volume = np.stack(channels, axis=0)

        # Return as list with single volume (one patient per file)
        return [volume]


def create_synthetic_data(
    num_samples: int = 8,
    num_channels: int = 2,
    depth: int = 32,
    height: int = 64,
    width: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic volumetric data for testing.

    Args:
        num_samples: Number of samples to generate.
        num_channels: Number of input channels.
        depth: Volume depth.
        height: Volume height.
        width: Volume width.

    Returns:
        Tuple of (volumes, labels) tensors.
    """
    volumes = torch.randn(num_samples, num_channels, depth, height, width)
    labels = torch.randint(0, 2, (num_samples,))
    return volumes, labels


def train_epoch(
    model: nn.Module,
    volumes: torch.Tensor,
    labels: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    tabular: dict[str, torch.Tensor] | None = None,
    batch_size: int = 2,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        volumes: Volume tensor (N, C, D, H, W).
        labels: Label tensor (N,).
        optimizer: Optimizer.
        criterion: Loss function.
        tabular: Optional tabular data dict.
        batch_size: Batch size.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    num_samples = volumes.size(0)
    indices = torch.randperm(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_volumes = volumes[batch_indices]
        batch_labels = labels[batch_indices]

        # Prepare tabular batch if needed
        batch_tabular = None
        if tabular is not None:
            batch_tabular = {k: v[batch_indices] for k, v in tabular.items()}

        optimizer.zero_grad()
        outputs = model(batch_volumes, tabular=batch_tabular)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Demo training script for ResNet3D")
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to HDF5 file or directory containing HDF5 files from m3cv-prep",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of HDF5",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--base-filters",
        type=int,
        default=16,
        help="Base filter count (default: 16 for smaller model)",
    )
    parser.add_argument(
        "--with-fusion",
        action="store_true",
        help="Enable late tabular fusion",
    )
    parser.add_argument(
        "--tabular-dim",
        type=int,
        default=10,
        help="Tabular feature dimension for fusion (default: 10)",
    )
    args = parser.parse_args()

    # Validate args
    if not args.synthetic and not args.data:
        print("Error: Must specify --data or --synthetic")
        return 1

    # Load or create data
    print("Loading data...")
    if args.synthetic:
        volumes, labels = create_synthetic_data(
            num_samples=8,
            num_channels=2,
            depth=32,
            height=64,
            width=64,
        )
        print(f"Created synthetic data: {volumes.shape}")
    else:
        volumes, labels = load_hdf5_data(args.data, max_samples=None)
        print(f"Loaded {len(volumes)} patient(s): {volumes.shape}")

    num_channels = volumes.size(1)

    # Create fusion config if requested
    fusion_config = None
    tabular = None
    if args.with_fusion:
        fusion_config = FusionConfig(
            late=FusionPoint(tabular_dim=args.tabular_dim, mode="concat")
        )
        # Create synthetic tabular data
        tabular = {"late": torch.randn(volumes.size(0), args.tabular_dim)}
        print(f"Enabled late fusion with {args.tabular_dim}-dim tabular features")

    # Build model
    print(f"Building ResNet-18 with {args.base_filters} base filters...")
    model = ResNet3DBuilder.build_resnet_18(
        in_channels=num_channels,
        num_classes=2,
        base_filters=args.base_filters,
        fusion_config=fusion_config,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(
            model=model,
            volumes=volumes,
            labels=labels,
            optimizer=optimizer,
            criterion=criterion,
            tabular=tabular,
            batch_size=2,
        )
        print(f"  Epoch {epoch + 1}/{args.epochs}: loss = {loss:.4f}")

    print("\nTraining complete! Gradients flow correctly.")
    return 0


if __name__ == "__main__":
    exit(main())
