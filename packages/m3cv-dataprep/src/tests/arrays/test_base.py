"""Tests for base module alignment and slice compatibility."""

import numpy as np
import pytest

from m3cv_prep.arrays import (
    PatientArray,
    SliceCompatibilityError,
    SpatialMetadata,
    check_slice_compatibility,
)


class TestCheckSliceCompatibility:
    """Tests for the check_slice_compatibility function."""

    def test_identical_slices(self):
        """Test when source and target have identical slice positions."""
        source = (0.0, 3.0, 6.0, 9.0, 12.0)
        target = (0.0, 3.0, 6.0, 9.0, 12.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []
        assert missing == []

    def test_source_subset_of_target(self):
        """Test when source is a contiguous subset of target."""
        source = (6.0, 9.0, 12.0)
        target = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []
        assert missing == []

    def test_source_subset_at_start(self):
        """Test when source covers the start of target."""
        source = (0.0, 3.0, 6.0)
        target = (0.0, 3.0, 6.0, 9.0, 12.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []
        assert missing == []

    def test_misaligned_slices(self):
        """Test when source slices don't align with any target slice."""
        source = (1.0, 4.0, 7.0)  # Offset by 1mm from target
        target = (0.0, 3.0, 6.0, 9.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        # All source slices are misaligned (1.0mm > 0.3mm tolerance)
        assert len(misaligned) == 3
        assert 1.0 in misaligned
        assert 4.0 in misaligned
        assert 7.0 in misaligned

    def test_interleaved_slices_detected(self):
        """Test that interleaved slices (non-contiguous) are detected."""
        # Source at every other target slice - would leave gaps
        source = (0.0, 6.0, 12.0)
        target = (0.0, 3.0, 6.0, 9.0, 12.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []  # All source slices align
        # But target slices at 3.0 and 9.0 are missing (gaps)
        assert 3.0 in missing
        assert 9.0 in missing

    def test_coarser_source_with_gaps(self):
        """Test source with coarser spacing that creates gaps."""
        source = (0.0, 5.0, 10.0)  # 5mm spacing
        target = (0.0, 2.5, 5.0, 7.5, 10.0)  # 2.5mm spacing

        misaligned, missing = check_slice_compatibility(source, target, 5.0, 2.5)

        assert misaligned == []
        # Target slices at 2.5 and 7.5 are within source extent but not covered
        assert 2.5 in missing
        assert 7.5 in missing

    def test_tolerance_allows_close_slices(self):
        """Test that slices within tolerance are considered aligned."""
        # Source is very slightly offset (< 10% of slice thickness)
        source = (0.02, 3.02, 6.02)
        target = (0.0, 3.0, 6.0, 9.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []
        assert missing == []

    def test_tolerance_rejects_far_slices(self):
        """Test that slices beyond tolerance are flagged as misaligned."""
        # Source offset by 0.5mm (> 10% of 3mm thickness = 0.3mm tolerance)
        source = (0.5, 3.5, 6.5)
        target = (0.0, 3.0, 6.0, 9.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert len(misaligned) == 3

    def test_single_source_slice(self):
        """Test with a single source slice."""
        source = (6.0,)
        target = (0.0, 3.0, 6.0, 9.0, 12.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []
        assert missing == []

    def test_source_extends_beyond_target(self):
        """Test that source slices beyond target extent are ignored (will be trimmed)."""
        # Source extends before and after target
        source = (-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0)
        target = (0.0, 3.0, 6.0, 9.0, 12.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        # No errors - slices outside target will be trimmed
        assert misaligned == []
        assert missing == []

    def test_source_partial_coverage(self):
        """Test that source not covering full target is OK (will be padded)."""
        # Source only covers middle of target
        source = (3.0, 6.0, 9.0)
        target = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        # No errors - target slices at 0.0, 12.0, 15.0 will be padded with voidval
        assert misaligned == []
        assert missing == []

    def test_source_extends_and_has_gaps(self):
        """Test source that extends beyond target BUT has gaps within target extent."""
        # Source extends beyond target, but within target extent has gaps
        source = (-3.0, 0.0, 6.0, 12.0, 15.0)  # Missing 3.0 and 9.0 within target
        target = (0.0, 3.0, 6.0, 9.0, 12.0)

        misaligned, missing = check_slice_compatibility(source, target, 3.0, 3.0)

        assert misaligned == []
        # Gaps within coverage should still be detected
        assert 3.0 in missing
        assert 9.0 in missing


class TestPatientArrayAlignWith:
    """Tests for PatientArray.align_with with slice compatibility checks."""

    @pytest.fixture
    def create_patient_array(self, sample_3d_array):
        """Factory fixture to create PatientArray instances."""

        def _create(slice_ref, position=(0.0, 0.0, 0.0), array=None):
            if array is None:
                array = np.zeros((len(slice_ref), 64, 64))
            metadata = SpatialMetadata(
                position=position,
                pixel_size=(1.0, 1.0),
                slice_ref=tuple(slice_ref),
                slice_thickness=abs(slice_ref[1] - slice_ref[0])
                if len(slice_ref) > 1
                else 1.0,
                even_spacing=True,
            )

            class ConcretePatientArray(PatientArray):
                voidval = 0.0

            return ConcretePatientArray(
                array=array,
                spatial_metadata=metadata,
                study_uid="1.2.3",
                frame_of_reference="1.2.3.4",
                patient_id="TEST001",
            )

        return _create

    def test_align_with_compatible_slices(self, create_patient_array):
        """Test alignment succeeds with compatible slice grids."""
        source = create_patient_array([3.0, 6.0, 9.0])
        target = create_patient_array([0.0, 3.0, 6.0, 9.0, 12.0])

        # Should not raise
        source.align_with(target)

        # Source should now have same shape as target
        assert source.array.shape == target.array.shape

    def test_align_with_misaligned_slices_strict(self, create_patient_array):
        """Test alignment fails with misaligned slices in strict mode."""
        source = create_patient_array([1.0, 4.0, 7.0])  # Offset from target
        target = create_patient_array([0.0, 3.0, 6.0, 9.0])

        with pytest.raises(SliceCompatibilityError) as exc_info:
            source.align_with(target, strict_slice_alignment=True)

        assert exc_info.value.misaligned_slices is not None
        assert 1.0 in exc_info.value.misaligned_slices

    def test_align_with_interleaved_slices_strict(self, create_patient_array):
        """Test alignment fails with interleaved (non-contiguous) slices."""
        source = create_patient_array([0.0, 6.0, 12.0])  # Gaps at 3.0, 9.0
        target = create_patient_array([0.0, 3.0, 6.0, 9.0, 12.0])

        with pytest.raises(SliceCompatibilityError) as exc_info:
            source.align_with(target, strict_slice_alignment=True)

        assert exc_info.value.missing_slices is not None
        assert 3.0 in exc_info.value.missing_slices
        assert 9.0 in exc_info.value.missing_slices

    def test_align_with_bypass_strict_mode(self, create_patient_array):
        """Test that strict mode can be bypassed."""
        source = create_patient_array([0.0, 6.0, 12.0])  # Interleaved
        target = create_patient_array([0.0, 3.0, 6.0, 9.0, 12.0])

        # Should not raise when strict mode is disabled
        source.align_with(target, strict_slice_alignment=False)
