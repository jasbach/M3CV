"""Pytest fixtures for m3cv-data tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def sample_h5_path(tmp_path: Path) -> str:
    """Create a sample HDF5 file with patient data."""
    h5_path = tmp_path / "patient_001.h5"

    with h5py.File(h5_path, "w") as f:
        # Create CT array (Z, Y, X) = (10, 64, 64)
        ct = np.random.randn(10, 64, 64).astype(np.float32) * 1000 - 1000
        f.create_dataset("ct", data=ct)

        # Create dose array
        dose = np.random.rand(10, 64, 64).astype(np.float32) * 50
        f.create_dataset("dose", data=dose)

        # Create sparse structure masks
        f.create_group("structures")

        # GTV structure - small blob in center
        gtv_group = f.create_group("structures/GTV")
        rows = np.array([32, 32, 33, 33], dtype=np.int32)
        cols = np.array([32, 33, 32, 33], dtype=np.int32)
        slices = np.array([5, 5, 5, 5], dtype=np.int32)
        gtv_group.create_dataset("rows", data=rows)
        gtv_group.create_dataset("cols", data=cols)
        gtv_group.create_dataset("slices", data=slices)

        # PTV structure - larger blob
        ptv_group = f.create_group("structures/PTV")
        rows = np.arange(30, 35, dtype=np.int32)
        cols = np.arange(30, 35, dtype=np.int32)
        slices = np.full(5, 5, dtype=np.int32)
        ptv_group.create_dataset("rows", data=rows)
        ptv_group.create_dataset("cols", data=cols)
        ptv_group.create_dataset("slices", data=slices)

        # Add patient identifiers
        f.attrs["patient_id"] = "TEST001"
        f.attrs["study_uid"] = "1.2.3.4.5"
        f.attrs["frame_of_reference"] = "1.2.3.4.6"

        # Add spatial metadata
        spatial_group = f.create_group("spatial_metadata")
        spatial_group.attrs["position_x"] = -160.0
        spatial_group.attrs["position_y"] = -160.0
        spatial_group.attrs["position_z"] = 0.0
        spatial_group.attrs["pixel_size_row"] = 1.0
        spatial_group.attrs["pixel_size_col"] = 1.0
        spatial_group.attrs["slice_thickness"] = 3.0
        spatial_group.attrs["even_spacing"] = True
        spatial_group.create_dataset(
            "slice_ref", data=np.arange(0, 30, 3, dtype=np.float64)
        )

    return str(h5_path)


@pytest.fixture
def sample_h5_minimal(tmp_path: Path) -> str:
    """Create a minimal HDF5 file with only CT."""
    h5_path = tmp_path / "patient_minimal.h5"

    with h5py.File(h5_path, "w") as f:
        ct = np.random.randn(5, 32, 32).astype(np.float32)
        f.create_dataset("ct", data=ct)
        f.attrs["patient_id"] = "MINIMAL001"

    return str(h5_path)


@pytest.fixture
def sample_h5_directory(tmp_path: Path) -> str:
    """Create a directory with multiple HDF5 files."""
    for i in range(3):
        h5_path = tmp_path / f"patient_{i:03d}.h5"
        with h5py.File(h5_path, "w") as f:
            ct = np.random.randn(10, 64, 64).astype(np.float32)
            f.create_dataset("ct", data=ct)
            f.attrs["patient_id"] = f"PATIENT{i:03d}"

            if i > 0:  # Add dose to some files
                dose = np.random.rand(10, 64, 64).astype(np.float32)
                f.create_dataset("dose", data=dose)

            if i == 2:  # Add structures to one file
                f.create_group("structures")
                gtv_group = f.create_group("structures/GTV")
                gtv_group.create_dataset("rows", data=np.array([32], dtype=np.int32))
                gtv_group.create_dataset("cols", data=np.array([32], dtype=np.int32))
                gtv_group.create_dataset("slices", data=np.array([5], dtype=np.int32))

    return str(tmp_path)
