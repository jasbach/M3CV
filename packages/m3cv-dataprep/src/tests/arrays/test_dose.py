"""Tests for PatientDose class."""

from unittest.mock import patch

import numpy as np
import pytest

from m3cv_prep.arrays import PatientDose
from m3cv_prep.arrays.exceptions import DoseTypeError, MetadataMismatchError
from m3cv_prep.arrays.protocols import SpatialMetadata


class TestPatientDose:
    """Tests for PatientDose class."""

    def test_init(self, sample_3d_array):
        """Test direct initialization."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(2.5, 2.5),
            slice_ref=(0.0, 2.5, 5.0, 7.5, 10.0),
            slice_thickness=2.5,
            even_spacing=True,
        )

        dose = PatientDose(
            array=sample_3d_array[:5],
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
            dose_units="GY",
        )

        assert dose.voidval == 0.0
        assert dose.dose_units == "GY"
        assert dose.studyUID == "1.2.3"

    def test_from_plan_file(self, mock_dose_plan_file):
        """Test construction from PLAN dose file."""
        dose = PatientDose.from_plan_file(mock_dose_plan_file)

        assert dose.array.shape == (5, 100, 100)
        assert dose.dose_units == "GY"
        assert dose.studyUID == "1.2.3.4.5"
        assert dose.even_spacing is True

    def test_from_plan_file_wrong_type(self, mock_dose_plan_file):
        """Test that non-PLAN file raises DoseTypeError."""
        mock_dose_plan_file.DoseSummationType = "BEAM"

        with pytest.raises(DoseTypeError) as exc_info:
            PatientDose.from_plan_file(mock_dose_plan_file)

        assert exc_info.value.expected == "PLAN"
        assert exc_info.value.actual == "BEAM"

    def test_from_beam_files(self, mock_dose_beam_files):
        """Test construction from BEAM dose files."""
        # Need to mock merge_doses since our mock files aren't real FileDatasets
        with patch("m3cv_prep.arrays.dose.merge_doses") as mock_merge:
            mock_merge.return_value = np.random.rand(5, 100, 100) * 1.5

            dose = PatientDose.from_beam_files(mock_dose_beam_files)

            assert dose.array.shape == (5, 100, 100)
            assert dose.dose_units == "GY"
            mock_merge.assert_called_once()

    def test_from_beam_files_empty(self):
        """Test that empty dcms list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PatientDose.from_beam_files([])

    def test_from_beam_files_wrong_type(self, mock_dose_beam_files):
        """Test that non-BEAM files raise DoseTypeError."""
        mock_dose_beam_files[1].DoseSummationType = "PLAN"

        with pytest.raises(DoseTypeError) as exc_info:
            PatientDose.from_beam_files(mock_dose_beam_files)

        assert exc_info.value.expected == "BEAM"

    def test_from_beam_files_mismatched(self, mock_dose_beam_files):
        """Test that mismatched BEAM files raise MetadataMismatchError."""
        mock_dose_beam_files[1].PixelSpacing = [5.0, 5.0]  # Different

        with pytest.raises(MetadataMismatchError):
            PatientDose.from_beam_files(mock_dose_beam_files)

    def test_from_dicom_single_plan(self, mock_dose_plan_file):
        """Test from_dicom dispatcher with single PLAN file."""
        dose = PatientDose.from_dicom(mock_dose_plan_file)

        assert dose.dose_units == "GY"

    def test_from_dicom_single_plan_in_list(self, mock_dose_plan_file):
        """Test from_dicom dispatcher with single PLAN file in list."""
        dose = PatientDose.from_dicom([mock_dose_plan_file])

        assert dose.dose_units == "GY"

    def test_from_dicom_beam_list(self, mock_dose_beam_files):
        """Test from_dicom dispatcher with BEAM file list."""
        with patch("m3cv_prep.arrays.dose.merge_doses") as mock_merge:
            mock_merge.return_value = np.random.rand(5, 100, 100) * 1.5

            dose = PatientDose.from_dicom(mock_dose_beam_files)

            assert dose.dose_units == "GY"

    def test_from_dicom_unknown_type(self, mock_dose_plan_file):
        """Test from_dicom with unknown dose type."""
        mock_dose_plan_file.DoseSummationType = "UNKNOWN"

        with pytest.raises(DoseTypeError) as exc_info:
            PatientDose.from_dicom(mock_dose_plan_file)

        assert "PLAN or BEAM" in exc_info.value.expected

    def test_properties(self, mock_dose_plan_file):
        """Test computed properties."""
        dose = PatientDose.from_plan_file(mock_dose_plan_file)

        assert dose.rows == 100
        assert dose.columns == 100
        assert dose.position == (-125.0, -125.0, 0.0)
        assert dose.pixel_size == (2.5, 2.5)
