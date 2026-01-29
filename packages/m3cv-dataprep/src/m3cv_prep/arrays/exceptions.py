"""Custom exceptions for the arrays module."""

from __future__ import annotations


class ArrayError(Exception):
    """Base exception for all array-related errors."""

    pass


class MetadataMismatchError(ArrayError):
    """Raised when DICOM metadata is incompatible across files."""

    def __init__(self, attribute: str, message: str | None = None):
        self.attribute = attribute
        if message is None:
            message = f"Incompatible metadata for attribute: {attribute}"
        super().__init__(message)


class AlignmentError(ArrayError):
    """Raised when array alignment fails."""

    pass


class UnevenSpacingError(AlignmentError):
    """Raised when alignment fails due to uneven slice spacing."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Cannot align arrays with uneven spacing or missing slices"
        super().__init__(message)


class ROINotFoundError(ArrayError):
    """Raised when a requested ROI is not found in the structure set."""

    def __init__(self, roi_name: str, available_rois: list[str] | None = None):
        self.roi_name = roi_name
        self.available_rois = available_rois
        message = f"ROI '{roi_name}' not found in structure set"
        if available_rois:
            message += f". Available ROIs: {available_rois}"
        super().__init__(message)


class DoseTypeError(ArrayError):
    """Raised when an unexpected dose file type is encountered."""

    def __init__(self, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        message = f"Expected dose type '{expected}', got '{actual}'"
        super().__init__(message)
