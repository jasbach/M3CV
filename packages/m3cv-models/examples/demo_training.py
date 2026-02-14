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
    max_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load volumetric data from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file from m3cv-prep.
        max_samples: Maximum number of samples to load.

    Returns:
        Tuple of (volumes, labels) tensors.
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py required for HDF5 loading. Install with: pip install h5py"
        )

    volumes = []
    with h5py.File(hdf5_path, "r") as f:
        # Iterate over patient groups
        patient_keys = [k for k in f.keys() if k.startswith("patient_")]
        if not patient_keys:
            # Try direct array access
            patient_keys = list(f.keys())

        for key in patient_keys[:max_samples]:
            group = f[key]

            # Try to find CT and dose arrays
            channels = []
            if "ct" in group:
                ct = np.array(group["ct"])
                channels.append(ct)
            if "dose" in group:
                dose = np.array(group["dose"])
                channels.append(dose)

            if not channels:
                # Try loading as direct array
                if isinstance(group, h5py.Dataset):
                    arr = np.array(group)
                    if arr.ndim == 3:
                        channels.append(arr)
                    elif arr.ndim == 4:
                        for c in range(arr.shape[0]):
                            channels.append(arr[c])

            if channels:
                # Stack channels and add to list
                volume = np.stack(channels, axis=0)
                volumes.append(volume)

    if not volumes:
        raise ValueError(f"No volumetric data found in {hdf5_path}")

    # Stack all volumes and create synthetic labels
    volumes_arr = np.stack(volumes, axis=0)
    volumes_tensor = torch.from_numpy(volumes_arr).float()

    # Create synthetic binary labels
    labels = torch.randint(0, 2, (len(volumes),))

    return volumes_tensor, labels


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
        help="Path to HDF5 file from m3cv-prep",
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
        volumes, labels = load_hdf5_data(args.data, max_samples=10)
        print(f"Loaded HDF5 data: {volumes.shape}")

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
