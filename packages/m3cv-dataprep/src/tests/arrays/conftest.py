"""Shared fixtures for arrays tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class MockDicomDataset:
    """Mock pydicom Dataset for testing."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_ct_files() -> list[MockDicomDataset]:
    """Create a list of mock CT DICOM files."""
    files = []
    for i in range(10):
        z_pos = float(i * 2.5)  # 2.5mm slice spacing
        files.append(
            MockDicomDataset(
                StudyInstanceUID="1.2.3.4.5",
                FrameOfReferenceUID="1.2.3.4.6",
                PatientID="TEST001",
                SliceThickness=2.5,
                Rows=512,
                Columns=512,
                PixelSpacing=[0.9765625, 0.9765625],
                ImagePositionPatient=[-250.0, -250.0, z_pos],
                RescaleSlope=1,
                RescaleIntercept=-1024,
                pixel_array=np.zeros((512, 512), dtype=np.int16),
            )
        )
    return files


@pytest.fixture
def mock_dose_plan_file() -> MockDicomDataset:
    """Create a mock PLAN dose DICOM file."""
    return MockDicomDataset(
        StudyInstanceUID="1.2.3.4.5",
        FrameOfReferenceUID="1.2.3.4.6",
        PatientID="TEST001",
        Modality="RTDOSE",
        DoseSummationType="PLAN",
        DoseUnits="GY",
        DoseGridScaling=0.001,
        PixelSpacing=[2.5, 2.5],
        ImagePositionPatient=[-125.0, -125.0, 0.0],
        GridFrameOffsetVector=[0.0, 2.5, 5.0, 7.5, 10.0],
        pixel_array=np.random.rand(5, 100, 100) * 1000,
    )


@pytest.fixture
def mock_dose_beam_files() -> list[MockDicomDataset]:
    """Create a list of mock BEAM dose DICOM files."""
    base_array = np.random.rand(5, 100, 100) * 500
    files = []
    for _i in range(3):
        files.append(
            MockDicomDataset(
                StudyInstanceUID="1.2.3.4.5",
                FrameOfReferenceUID="1.2.3.4.6",
                PatientID="TEST001",
                Modality="RTDOSE",
                DoseSummationType="BEAM",
                DoseUnits="GY",
                DoseGridScaling=0.001,
                PixelSpacing=[2.5, 2.5],
                ImagePositionPatient=[-125.0, -125.0, 0.0],
                ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
                GridFrameOffsetVector=[0.0, 2.5, 5.0, 7.5, 10.0],
                Rows=100,
                Columns=100,
                pixel_array=base_array.copy(),
            )
        )
    return files


@pytest.fixture
def mock_rtstruct_file() -> MockDicomDataset:
    """Create a mock RTSTRUCT DICOM file."""
    roi_sequence = [
        MockDicomDataset(ROIName="PTV70", ROINumber=1),
        MockDicomDataset(ROIName="Parotid_L", ROINumber=2),
        MockDicomDataset(ROIName="Parotid_R", ROINumber=3),
    ]

    # Create mock contour data for PTV70
    contour_data = []
    for z in [5.0, 7.5, 10.0]:
        plane = MockDicomDataset(
            ContourData=[
                -10.0,
                -10.0,
                z,
                10.0,
                -10.0,
                z,
                10.0,
                10.0,
                z,
                -10.0,
                10.0,
                z,
            ]
        )
        contour_data.append(plane)

    roi_contour_sequence = [
        MockDicomDataset(
            ReferencedROINumber=1,
            ContourSequence=contour_data,
        ),
        MockDicomDataset(
            ReferencedROINumber=2,
            ContourSequence=[],
        ),
        MockDicomDataset(
            ReferencedROINumber=3,
            ContourSequence=[],
        ),
    ]

    return MockDicomDataset(
        StudyInstanceUID="1.2.3.4.5",
        PatientID="TEST001",
        StructureSetROISequence=roi_sequence,
        ROIContourSequence=roi_contour_sequence,
    )


@pytest.fixture
def sample_3d_array() -> np.ndarray:
    """Create a sample 3D array for testing."""
    return np.random.rand(10, 64, 64).astype(np.float64) * 1000 - 500


@pytest.fixture
def sample_binary_mask() -> np.ndarray:
    """Create a sample binary mask array for testing."""
    mask = np.zeros((10, 64, 64), dtype=np.int16)
    # Add a cube in the center
    mask[3:7, 20:44, 20:44] = 1
    return mask
