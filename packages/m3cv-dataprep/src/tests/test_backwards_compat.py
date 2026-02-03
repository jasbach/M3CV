"""Tests for backwards compatibility of deprecated imports."""

import warnings


class TestDeprecatedImports:
    """Tests for deprecated import paths."""

    def test_arrayclasses_import_warning(self):
        """Test that importing from arrayclasses shows deprecation warning."""
        import sys

        # Remove from cache if present so we get a fresh import
        if "m3cv_prep.arrayclasses" in sys.modules:
            del sys.modules["m3cv_prep.arrayclasses"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import the deprecated module - this should trigger the warning
            import m3cv_prep.arrayclasses  # noqa: F401

            # Check for deprecation warning
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert "m3cv_prep.arrays" in str(deprecation_warnings[0].message)

    def test_arrayclasses_exports(self):
        """Test that deprecated module still exports the classes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from m3cv_prep.arrayclasses import (
                PatientArray,
                PatientCT,
                PatientDose,
                PatientMask,
            )

            # Verify they are the same classes
            from m3cv_prep.arrays import (
                PatientArray as NewPatientArray,
            )
            from m3cv_prep.arrays import (
                PatientCT as NewPatientCT,
            )
            from m3cv_prep.arrays import (
                PatientDose as NewPatientDose,
            )
            from m3cv_prep.arrays import (
                PatientMask as NewPatientMask,
            )

            assert PatientArray is NewPatientArray
            assert PatientCT is NewPatientCT
            assert PatientDose is NewPatientDose
            assert PatientMask is NewPatientMask

    def test_preprocess_util_deprecation_warnings(self):
        """Test that deprecated functions in _preprocess_util show warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from m3cv_prep._preprocess_util import getscaledimg

            # Create a mock file to call the function
            class MockFile:
                pixel_array = __import__("numpy").zeros((10, 10), dtype="int16")
                RescaleSlope = 1
                RescaleIntercept = 0

            getscaledimg(MockFile())

            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert "dicom_utils" in str(deprecation_warnings[0].message)


class TestNewImports:
    """Tests for new import paths."""

    def test_arrays_module_imports(self):
        """Test that new arrays module imports work correctly."""
        from m3cv_prep.arrays import (
            PatientArray,
            PatientCT,
            PatientDose,
            PatientMask,
            SpatialMetadata,
        )

        # Basic sanity checks
        assert PatientArray is not None
        assert PatientCT is not None
        assert PatientDose is not None
        assert PatientMask is not None
        assert SpatialMetadata is not None

    def test_dicom_utils_imports(self):
        """Test that dicom_utils exports new functions."""
        from m3cv_prep.dicom_utils import (
            attr_shared,
            getscaledimg,
            merge_doses,
            window_level,
        )

        assert callable(getscaledimg)
        assert callable(attr_shared)
        assert callable(merge_doses)
        assert callable(window_level)
