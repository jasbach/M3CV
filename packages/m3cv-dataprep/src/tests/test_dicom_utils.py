"""Tests for dicom_utils module."""

import numpy as np
import pytest

from m3cv_prep import dicom_utils


class MockDataset:
    """Mock pydicom Dataset for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestGetScaledImg:
    """Tests for getscaledimg function."""

    def test_basic_scaling(self):
        """Test basic CT pixel scaling."""
        mock_file = MockDataset(
            pixel_array=np.array([[100, 200], [300, 400]], dtype=np.int16),
            RescaleSlope=1,
            RescaleIntercept=-1024,
        )

        result = dicom_utils.getscaledimg(mock_file)

        expected = np.array([[-924, -824], [-724, -624]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_with_slope(self):
        """Test scaling with non-unity slope."""
        mock_file = MockDataset(
            pixel_array=np.array([[100, 200]], dtype=np.int16),
            RescaleSlope=2,
            RescaleIntercept=-1024,
        )

        result = dicom_utils.getscaledimg(mock_file)

        expected = np.array([[-824, -624]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)


class TestAttrShared:
    """Tests for attr_shared function."""

    def test_shared_attribute(self):
        """Test when all files share the attribute."""
        files = [
            MockDataset(PatientID="TEST001"),
            MockDataset(PatientID="TEST001"),
            MockDataset(PatientID="TEST001"),
        ]

        assert dicom_utils.attr_shared(files, "PatientID") is True

    def test_different_attribute(self):
        """Test when files have different attribute values."""
        files = [
            MockDataset(PatientID="TEST001"),
            MockDataset(PatientID="TEST002"),
            MockDataset(PatientID="TEST001"),
        ]

        assert dicom_utils.attr_shared(files, "PatientID") is False


class TestWindowLevel:
    """Tests for window_level function."""

    def test_basic_windowing(self):
        """Test basic CT windowing."""
        array = np.array([-500, 0, 500, 1000], dtype=np.float64)

        result = dicom_utils.window_level(array, window=400, level=40)

        # Window: -160 to 240
        assert result[0] == -160  # Clipped from -500
        assert result[1] == 0  # Within window
        assert result[2] == 240  # Clipped from 500
        assert result[3] == 240  # Clipped from 1000

    def test_normalized_windowing(self):
        """Test normalized window/level."""
        array = np.array([-160, 40, 240], dtype=np.float64)

        result = dicom_utils.window_level(array, window=400, level=40, normalize=True)

        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] == pytest.approx(0.5, abs=0.01)
        assert result[2] == pytest.approx(1.0, abs=0.01)


class TestValidatePatientid:
    """Tests for validate_patientid function."""

    def test_single_patient(self):
        """Test with files from single patient."""
        files = [
            MockDataset(PatientID="TEST001"),
            MockDataset(PatientID="TEST001"),
        ]

        # Should not raise
        dicom_utils.validate_patientid(files)

    def test_multiple_patients(self):
        """Test with files from multiple patients."""
        files = [
            MockDataset(PatientID="TEST001"),
            MockDataset(PatientID="TEST002"),
        ]

        with pytest.raises(ValueError, match="multiple patients"):
            dicom_utils.validate_patientid(files)

    def test_empty_list(self):
        """Test with empty list."""
        # Should not raise
        dicom_utils.validate_patientid([])


class TestGroupDcmsByModality:
    """Tests for group_dcms_by_modality function."""

    def test_grouping(self):
        """Test modality grouping."""
        files = [
            MockDataset(Modality="CT"),
            MockDataset(Modality="CT"),
            MockDataset(Modality="RTDOSE"),
            MockDataset(Modality="RTSTRUCT"),
        ]

        result = dicom_utils.group_dcms_by_modality(files)

        assert len(result["CT"]) == 2
        assert len(result["RTDOSE"]) == 1
        assert len(result["RTSTRUCT"]) == 1
