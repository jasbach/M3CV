"""Transforms for Patient data preprocessing.

This module provides transforms that operate on Patient objects before channel
stacking, enabling anatomical reference-based cropping and other preprocessing.

Example:
    >>> from m3cv_data.transforms import AnatomicalCrop, BilateralStructureMidpoint
    >>> strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")
    >>> crop = AnatomicalCrop(
    ...     crop_shape=(90, 128, 128),
    ...     reference_strategy=strategy,
    ...     allow_fallback=False,
    ... )
    >>> cropped_patient = crop(patient)
"""

from m3cv_data.transforms.cropping import AnatomicalCrop
from m3cv_data.transforms.exceptions import ReferenceNotFoundError, TransformError
from m3cv_data.transforms.reference_strategies import (
    BilateralStructureMidpoint,
    CombinedStructuresCOM,
    FallbackStrategy,
    ReferenceStrategy,
    SingleStructureCOM,
    VolumeCenterStrategy,
)

__all__ = [
    "AnatomicalCrop",
    "BilateralStructureMidpoint",
    "CombinedStructuresCOM",
    "FallbackStrategy",
    "ReferenceNotFoundError",
    "ReferenceStrategy",
    "SingleStructureCOM",
    "TransformError",
    "VolumeCenterStrategy",
]
