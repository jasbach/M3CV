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


@pytest.fixture
def sample_h5_with_bilateral_structures(tmp_path: Path) -> str:
    """Create HDF5 file with bilateral structures for testing anatomical cropping."""
    h5_path = tmp_path / "patient_bilateral.h5"

    with h5py.File(h5_path, "w") as f:
        # Create CT array (Z, Y, X) = (40, 80, 80)
        ct = np.random.randn(40, 80, 80).astype(np.float32) * 1000 - 1000
        f.create_dataset("ct", data=ct)

        # Create dose array
        dose = np.random.rand(40, 80, 80).astype(np.float32) * 50
        f.create_dataset("dose", data=dose)

        # Create sparse structure masks
        f.create_group("structures")

        # Parotid_L - left side (lower X values)
        parotid_l_group = f.create_group("structures/Parotid_L")
        # Create a small blob around (20, 40, 25)
        parotid_l_rows = []
        parotid_l_cols = []
        parotid_l_slices = []
        for z in range(18, 23):
            for y in range(38, 43):
                for x in range(23, 28):
                    parotid_l_slices.append(z)
                    parotid_l_rows.append(y)
                    parotid_l_cols.append(x)
        parotid_l_group.create_dataset(
            "rows", data=np.array(parotid_l_rows, dtype=np.int32)
        )
        parotid_l_group.create_dataset(
            "cols", data=np.array(parotid_l_cols, dtype=np.int32)
        )
        parotid_l_group.create_dataset(
            "slices", data=np.array(parotid_l_slices, dtype=np.int32)
        )

        # Parotid_R - right side (higher X values)
        parotid_r_group = f.create_group("structures/Parotid_R")
        # Create a small blob around (20, 40, 55)
        parotid_r_rows = []
        parotid_r_cols = []
        parotid_r_slices = []
        for z in range(18, 23):
            for y in range(38, 43):
                for x in range(53, 58):
                    parotid_r_slices.append(z)
                    parotid_r_rows.append(y)
                    parotid_r_cols.append(x)
        parotid_r_group.create_dataset(
            "rows", data=np.array(parotid_r_rows, dtype=np.int32)
        )
        parotid_r_group.create_dataset(
            "cols", data=np.array(parotid_r_cols, dtype=np.int32)
        )
        parotid_r_group.create_dataset(
            "slices", data=np.array(parotid_r_slices, dtype=np.int32)
        )

        # GTV - center structure
        gtv_group = f.create_group("structures/GTV")
        gtv_rows = []
        gtv_cols = []
        gtv_slices = []
        for z in range(18, 23):
            for y in range(38, 43):
                for x in range(38, 43):
                    gtv_slices.append(z)
                    gtv_rows.append(y)
                    gtv_cols.append(x)
        gtv_group.create_dataset("rows", data=np.array(gtv_rows, dtype=np.int32))
        gtv_group.create_dataset("cols", data=np.array(gtv_cols, dtype=np.int32))
        gtv_group.create_dataset("slices", data=np.array(gtv_slices, dtype=np.int32))

        # Add patient identifiers
        f.attrs["patient_id"] = "BILATERAL001"
        f.attrs["study_uid"] = "1.2.3.4.7"
        f.attrs["frame_of_reference"] = "1.2.3.4.8"

        # Add spatial metadata
        spatial_group = f.create_group("spatial_metadata")
        spatial_group.attrs["position_x"] = -200.0
        spatial_group.attrs["position_y"] = -200.0
        spatial_group.attrs["position_z"] = 0.0
        spatial_group.attrs["pixel_size_row"] = 1.0
        spatial_group.attrs["pixel_size_col"] = 1.0
        spatial_group.attrs["slice_thickness"] = 3.0
        spatial_group.attrs["even_spacing"] = True
        spatial_group.create_dataset(
            "slice_ref", data=np.arange(0, 120, 3, dtype=np.float64)
        )

    return str(h5_path)
