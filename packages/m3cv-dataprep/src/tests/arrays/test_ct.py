"""Tests for PatientCT class."""

import pytest

from m3cv_prep.arrays import PatientCT
from m3cv_prep.arrays.exceptions import MetadataMismatchError
from m3cv_prep.arrays.protocols import SpatialMetadata


class TestPatientCT:
    """Tests for PatientCT class."""

    def test_init(self, sample_3d_array):
        """Test direct initialization."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i) for i in range(10)),
            slice_thickness=1.0,
            even_spacing=True,
        )

        ct = PatientCT(
            array=sample_3d_array,
            spatial_metadata=metadata,
            study_uid="1.2.3",
            frame_of_reference="1.2.4",
            patient_id="TEST001",
        )

        assert ct.voidval == -1000.0
        assert ct.array.shape == sample_3d_array.shape
        assert ct.studyUID == "1.2.3"
        assert ct.patient_id == "TEST001"

    def test_from_dicom_files(self, mock_ct_files):
        """Test construction from DICOM files."""
        ct = PatientCT.from_dicom_files(mock_ct_files)

        assert ct.array.shape == (10, 512, 512)
        assert ct.studyUID == "1.2.3.4.5"
        assert ct.patient_id == "TEST001"
        assert ct.even_spacing is True
        assert ct.slice_thickness == 2.5

    def test_from_dicom_files_empty(self):
        """Test that empty filelist raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PatientCT.from_dicom_files([])

    def test_from_dicom_files_mismatched_metadata(self, mock_ct_files):
        """Test that mismatched metadata raises MetadataMismatchError."""
        mock_ct_files[5].StudyInstanceUID = "different_uid"

        with pytest.raises(MetadataMismatchError) as exc_info:
            PatientCT.from_dicom_files(mock_ct_files)

        assert "StudyInstanceUID" in str(exc_info.value)

    def test_window_level(self, mock_ct_files):
        """Test window/level transformation."""
        ct = PatientCT.from_dicom_files(mock_ct_files)
        # Set some values to test windowing
        ct._array[0, 0, :5] = [-500, 0, 500, 1000, 1500]

        ct.window_level(window=400, level=40, normalize=False)

        # Values should be clipped to window
        assert ct.array[0, 0, 0] == -160  # level - window/2
        assert ct.array[0, 0, 4] == 240  # level + window/2

    def test_window_level_normalized(self, mock_ct_files):
        """Test normalized window/level transformation."""
        ct = PatientCT.from_dicom_files(mock_ct_files)
        # Window 400, level 40 means range is [-160, 240]
        ct._array[0, 0, :3] = [-160, 40, 240]

        ct.window_level(window=400, level=40, normalize=True)

        # Values should be normalized to [0, 1]
        assert ct.array[0, 0, 0] == pytest.approx(0.0, abs=0.01)  # -160 -> 0
        assert ct.array[0, 0, 1] == pytest.approx(0.5, abs=0.01)  # 40 -> 0.5
        assert ct.array[0, 0, 2] == pytest.approx(1.0, abs=0.01)  # 240 -> 1.0

    def test_properties(self, mock_ct_files):
        """Test computed properties."""
        ct = PatientCT.from_dicom_files(mock_ct_files)

        assert ct.rows == 512
        assert ct.columns == 512
        assert ct.height == pytest.approx(512 * 0.9765625)
        assert ct.width == pytest.approx(512 * 0.9765625)
        assert ct.position == (-250.0, -250.0, 0.0)

    def test_locate(self, mock_ct_files):
        """Test coordinate location."""
        ct = PatientCT.from_dicom_files(mock_ct_files)

        # Test a point in the center
        loc = ct.locate((-250.0 + 256, -250.0 + 256, 12.5))
        assert loc is not None
        assert loc[0] == 5  # z index
        assert loc[1] == pytest.approx(262, abs=1)  # y index
        assert loc[2] == pytest.approx(262, abs=1)  # x index

    def test_locate_outside(self, mock_ct_files):
        """Test location outside bounds returns None."""
        ct = PatientCT.from_dicom_files(mock_ct_files)

        # Outside z range
        assert ct.locate((0.0, 0.0, 100.0)) is None
        # Outside x range
        assert ct.locate((500.0, 0.0, 5.0)) is None
