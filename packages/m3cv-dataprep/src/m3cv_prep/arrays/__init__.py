"""Arrays subpackage for m3cv_prep.

This module provides classes for working with patient arrays from DICOM data:
- PatientArray: Base class with common operations
- PatientCT: CT image arrays
- PatientDose: Dose distribution arrays
- PatientMask: Structure mask arrays

Also provides protocols and exceptions for type-safe operations.
"""

from m3cv_prep.arrays.base import PatientArray, check_slice_compatibility
from m3cv_prep.arrays.ct import PatientCT
from m3cv_prep.arrays.dose import PatientDose
from m3cv_prep.arrays.exceptions import (
    AlignmentError,
    ArrayError,
    DoseTypeError,
    MetadataMismatchError,
    ROINotFoundError,
    SliceCompatibilityError,
    UnevenSpacingError,
)
from m3cv_prep.arrays.mask import PatientMask
from m3cv_prep.arrays.protocols import Alignable, SpatialMetadata

__all__ = [
    # Base class
    "PatientArray",
    # Modality classes
    "PatientCT",
    "PatientDose",
    "PatientMask",
    # Protocols
    "Alignable",
    "SpatialMetadata",
    # Utility functions
    "check_slice_compatibility",
    # Exceptions
    "ArrayError",
    "MetadataMismatchError",
    "AlignmentError",
    "SliceCompatibilityError",
    "UnevenSpacingError",
    "ROINotFoundError",
    "DoseTypeError",
]
