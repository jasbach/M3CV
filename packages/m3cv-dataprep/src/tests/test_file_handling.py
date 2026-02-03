"""Tests for file_handling module."""

import os
import tempfile

import numpy as np
import pytest

from m3cv_prep.arrays import PatientCT, SpatialMetadata
from m3cv_prep.file_handling import load_array_from_h5, save_array_to_h5


class TestSaveLoadMetadata:
    """Test that spatial metadata and patient identifiers are preserved."""

    @pytest.fixture
    def sample_ct(self):
        """Create a sample PatientCT for testing."""
        array = np.random.randn(10, 64, 64).astype(np.float32)
        spatial = SpatialMetadata(
            position=(-150.0, -200.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=tuple(float(i * 2.5) for i in range(10)),
            slice_thickness=2.5,
            even_spacing=True,
        )
        return PatientCT(
            array=array,
            spatial_metadata=spatial,
            study_uid="1.2.3.4.5",
            frame_of_reference="1.2.3.4.6",
            patient_id="TEST_PATIENT_001",
        )

    def test_save_load_preserves_patient_id(self, sample_ct):
        """Test that patient_id is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            save_array_to_h5(path, sample_ct)
            data = load_array_from_h5(path)

            assert "patient_id" in data
            assert data["patient_id"] == "TEST_PATIENT_001"

    def test_save_load_preserves_study_uid(self, sample_ct):
        """Test that study_uid is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            save_array_to_h5(path, sample_ct)
            data = load_array_from_h5(path)

            assert "study_uid" in data
            assert data["study_uid"] == "1.2.3.4.5"

    def test_save_load_preserves_frame_of_reference(self, sample_ct):
        """Test that frame_of_reference is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            save_array_to_h5(path, sample_ct)
            data = load_array_from_h5(path)

            assert "frame_of_reference" in data
            assert data["frame_of_reference"] == "1.2.3.4.6"

    def test_save_load_preserves_spatial_metadata(self, sample_ct):
        """Test that SpatialMetadata is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            save_array_to_h5(path, sample_ct)
            data = load_array_from_h5(path)

            assert "spatial_metadata" in data
            loaded_spatial = data["spatial_metadata"]

            assert loaded_spatial.position == sample_ct.spatial_metadata.position
            assert loaded_spatial.pixel_size == sample_ct.spatial_metadata.pixel_size
            assert loaded_spatial.slice_ref == sample_ct.spatial_metadata.slice_ref
            assert (
                loaded_spatial.slice_thickness
                == sample_ct.spatial_metadata.slice_thickness
            )
            assert (
                loaded_spatial.even_spacing == sample_ct.spatial_metadata.even_spacing
            )

    def test_save_load_preserves_ct_array(self, sample_ct):
        """Test that CT array data is preserved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            save_array_to_h5(path, sample_ct)
            data = load_array_from_h5(path)

            assert "ct" in data
            np.testing.assert_array_almost_equal(data["ct"], sample_ct.array)

    def test_raw_array_without_metadata(self):
        """Test that raw numpy arrays still work (no metadata saved)."""
        array = np.random.randn(10, 64, 64).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            save_array_to_h5(path, array)
            data = load_array_from_h5(path)

            assert "ct" in data
            assert "patient_id" not in data
            assert "spatial_metadata" not in data
            np.testing.assert_array_almost_equal(data["ct"], array)
