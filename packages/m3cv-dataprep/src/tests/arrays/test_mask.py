"""Tests for PatientMask class."""

import warnings

import numpy as np
import pytest

from m3cv_prep.arrays import PatientCT, PatientMask
from m3cv_prep.arrays.exceptions import ROINotFoundError
from m3cv_prep.arrays.protocols import SpatialMetadata


class TestPatientMask:
    """Tests for PatientMask class."""

    def test_init(self, sample_binary_mask):
        """Test direct initialization."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask = PatientMask(
            array=sample_binary_mask,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="TestROI",
            proper_name="Test Region",
        )

        assert mask.voidval == 0.0
        assert mask.roi_name == "TestROI"
        assert mask.proper_name == "Test Region"
        # Check array is binary
        assert set(np.unique(mask.array)).issubset({0, 1})

    def test_array_setter_enforces_binary(self, sample_binary_mask):
        """Test that array setter enforces binary values."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask = PatientMask(
            array=sample_binary_mask,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="TestROI",
        )

        # Set non-binary values
        non_binary = np.random.rand(10, 64, 64) * 2  # Values 0-2
        mask.array = non_binary

        # Should be clipped and rounded to binary
        assert set(np.unique(mask.array)).issubset({0, 1})

    def test_find_contour_sequence_found(self, mock_rtstruct_file):
        """Test finding a contour sequence that exists."""
        seq = PatientMask._find_contour_sequence(mock_rtstruct_file, "PTV70")

        assert len(seq) == 3  # 3 planes of contours

    def test_find_contour_sequence_not_found(self, mock_rtstruct_file):
        """Test that missing ROI raises ROINotFoundError."""
        with pytest.raises(ROINotFoundError) as exc_info:
            PatientMask._find_contour_sequence(mock_rtstruct_file, "NonExistent")

        assert exc_info.value.roi_name == "NonExistent"
        assert "PTV70" in exc_info.value.available_rois

    def test_join(self, sample_binary_mask):
        """Test joining two masks."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask1 = PatientMask(
            array=sample_binary_mask.copy(),
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="ROI1",
        )

        # Create second mask with different region
        mask2_array = np.zeros_like(sample_binary_mask)
        mask2_array[3:7, 30:50, 30:50] = 1

        mask2 = PatientMask(
            array=mask2_array,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="ROI2",
        )

        original_sum = np.sum(mask1.array)
        mask1.join(mask2)

        # Joined mask should have more or equal voxels
        assert np.sum(mask1.array) >= original_sum
        # Should still be binary
        assert set(np.unique(mask1.array)).issubset({0, 1})

    def test_join_type_error(self, sample_binary_mask):
        """Test that joining with non-mask raises TypeError."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask = PatientMask(
            array=sample_binary_mask,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="ROI1",
        )

        with pytest.raises(TypeError):
            mask.join("not a mask")

    def test_join_shape_error(self, sample_binary_mask):
        """Test that joining masks with different shapes raises ValueError."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask1 = PatientMask(
            array=sample_binary_mask,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="ROI1",
        )

        different_shape = np.zeros((5, 32, 32), dtype=np.int16)
        metadata2 = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(5)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask2 = PatientMask(
            array=different_shape,
            spatial_metadata=metadata2,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="ROI2",
        )

        with pytest.raises(ValueError, match="different shapes"):
            mask1.join(mask2)

    def test_com(self, sample_binary_mask):
        """Test center of mass calculation."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        mask = PatientMask(
            array=sample_binary_mask,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            roi_name="TestROI",
        )

        com = mask.com

        # COM should be near center of the cube (3:7, 20:44, 20:44)
        assert com[0] == pytest.approx(5.0, abs=0.5)  # Z: (3+6)/2 = 4.5
        assert com[1] == pytest.approx(32.0, abs=0.5)  # Y: (20+43)/2 = 31.5
        assert com[2] == pytest.approx(32.0, abs=0.5)  # X: (20+43)/2 = 31.5

    def test_study_uid_mismatch_warning(self, mock_ct_files, mock_rtstruct_file):
        """Test that StudyUID mismatch generates warning."""
        ct = PatientCT.from_dicom_files(mock_ct_files)
        mock_rtstruct_file.StudyInstanceUID = "different_study_uid"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                # This may fail due to coordinate mapping, but should still warn
                PatientMask.from_rtstruct(ct, mock_rtstruct_file, "PTV70")
            except Exception:
                pass

            # Check if warning was raised
            uid_warnings = [
                warning for warning in w if "StudyUID mismatch" in str(warning.message)
            ]
            assert len(uid_warnings) > 0
