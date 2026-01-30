"""Preprocessing utility functions.

This module previously contained domain-specific functions for working with
DICOM data. Most functions have been moved to more appropriate locations:

- Contour sorting and rasterization: m3cv_prep.arrays.mask
- Sparse mask packing/unpacking: m3cv_prep.array_tools
- DICOM utilities: m3cv_prep.dicom_utils

Some functions are re-exported here with deprecation warnings for backwards
compatibility.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


# Deprecated functions - re-exported from dicom_utils for backwards compatibility


def getscaledimg(file: Dataset) -> np.ndarray:
    """Extract and scale CT pixel array.

    .. deprecated::
        Use m3cv_prep.dicom_utils.getscaledimg instead.
    """
    warnings.warn(
        "getscaledimg is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.getscaledimg instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import getscaledimg as _getscaledimg

    return _getscaledimg(file)


def attr_shared(dcms: list[Dataset], attr: str) -> bool:
    """Check if all DICOM files share the same value for an attribute.

    .. deprecated::
        Use m3cv_prep.dicom_utils.attr_shared instead.
    """
    warnings.warn(
        "attr_shared is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.attr_shared instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import attr_shared as _attr_shared

    return _attr_shared(dcms, attr)


def merge_doses(*args: Dataset) -> np.ndarray:
    """Merge multiple BEAM dose files.

    .. deprecated::
        Use m3cv_prep.dicom_utils.merge_doses instead.
    """
    warnings.warn(
        "merge_doses is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.merge_doses instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import merge_doses as _merge_doses

    return _merge_doses(*args)


def window_level(
    array: np.ndarray,
    window: float,
    level: float,
    normalize: bool = False,
) -> np.ndarray:
    """Apply window/level transformation to CT array.

    .. deprecated::
        Use m3cv_prep.dicom_utils.window_level instead.
    """
    warnings.warn(
        "window_level is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.window_level instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import window_level as _window_level

    return _window_level(array, window, level, normalize)
