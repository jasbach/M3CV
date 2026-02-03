"""Tests for protocols module."""

import numpy as np
import pytest

from m3cv_prep.arrays.protocols import Alignable, SpatialMetadata


class TestSpatialMetadata:
    """Tests for SpatialMetadata dataclass."""

    def test_creation(self):
        """Test basic SpatialMetadata creation."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=(0.0, 1.0, 2.0, 3.0),
            slice_thickness=1.0,
            even_spacing=True,
        )
        assert metadata.position == (0.0, 0.0, 0.0)
        assert metadata.pixel_size == (1.0, 1.0)
        assert metadata.slice_ref == (0.0, 1.0, 2.0, 3.0)
        assert metadata.slice_thickness == 1.0
        assert metadata.even_spacing is True

    def test_frozen(self):
        """Test that SpatialMetadata is immutable."""
        metadata = SpatialMetadata(
            position=(0.0, 0.0, 0.0),
            pixel_size=(1.0, 1.0),
            slice_ref=(0.0, 1.0, 2.0),
            slice_thickness=1.0,
            even_spacing=True,
        )
        with pytest.raises(AttributeError):
            metadata.position = (1.0, 1.0, 1.0)

    def test_from_dicom_even_spacing(self, mock_ct_files):
        """Test from_dicom with evenly spaced slices."""
        ref_file = mock_ct_files[0]
        z_positions = [0.0, 2.5, 5.0, 7.5, 10.0]

        metadata = SpatialMetadata.from_dicom(ref_file, z_positions)

        assert metadata.position == (-250.0, -250.0, 0.0)
        assert metadata.pixel_size == (0.9765625, 0.9765625)
        assert metadata.slice_ref == tuple(z_positions)
        assert metadata.slice_thickness == 2.5
        assert metadata.even_spacing is True

    def test_from_dicom_uneven_spacing(self, mock_ct_files):
        """Test from_dicom with unevenly spaced slices."""
        ref_file = mock_ct_files[0]
        z_positions = [0.0, 2.5, 6.0, 7.5, 10.0]  # Uneven spacing

        metadata = SpatialMetadata.from_dicom(ref_file, z_positions)

        assert metadata.even_spacing is False

    def test_from_dose(self, mock_dose_plan_file):
        """Test from_dose factory method."""
        metadata = SpatialMetadata.from_dose(mock_dose_plan_file)

        assert metadata.position == (-125.0, -125.0, 0.0)
        assert metadata.pixel_size == (2.5, 2.5)
        assert len(metadata.slice_ref) == 5
        assert metadata.even_spacing is True


class TestAlignable:
    """Tests for Alignable protocol."""

    def test_protocol_check(self):
        """Test that Alignable is a runtime checkable protocol."""
        from m3cv_prep.arrays import SpatialMetadata

        # Create a minimal PatientArray-like object
        class MockAlignable:
            @property
            def spatial_metadata(self):
                return SpatialMetadata(
                    position=(0.0, 0.0, 0.0),
                    pixel_size=(1.0, 1.0),
                    slice_ref=(0.0,),
                    slice_thickness=1.0,
                    even_spacing=True,
                )

            @property
            def array(self):
                return np.zeros((1, 10, 10))

            @property
            def voidval(self):
                return 0.0

            def align_with(self, other):
                pass

        mock = MockAlignable()
        assert isinstance(mock, Alignable)

    def test_non_alignable(self):
        """Test that non-Alignable objects don't pass check."""

        class NotAlignable:
            pass

        obj = NotAlignable()
        assert not isinstance(obj, Alignable)
