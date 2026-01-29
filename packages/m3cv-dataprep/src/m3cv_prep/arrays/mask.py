"""PatientMask class for structure mask arrays."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from m3cv_prep.arrays.base import PatientArray
from m3cv_prep.arrays.exceptions import ROINotFoundError
from m3cv_prep.arrays.protocols import Alignable, SpatialMetadata

if TYPE_CHECKING:
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence


def _sort_coords(coords: NDArray[np.number]) -> NDArray[np.number]:
    """Sort contour coordinates clockwise around their centroid.

    Args:
        coords: Array of (x, y) coordinates.

    Returns:
        Sorted array of coordinates.
    """
    import math

    origin = coords.mean(axis=0)
    refvec = [0, 1]

    def clockwiseangle_and_dist(point: NDArray[np.number]) -> tuple[float, float]:
        vector = [point[0] - origin[0], point[1] - origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0.0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        return angle, lenvector

    sorted_coords = sorted(coords, key=clockwiseangle_and_dist)
    return np.array(sorted_coords)


class PatientMask(PatientArray):
    """Patient structure mask array.

    Array is stored as (Z, Y, X) with binary values [0, 1].
    Created by rasterizing contours from an RTSTRUCT file onto a reference array.

    Attributes:
        voidval: Void value is 0 (outside structure).
        roi_name: Name of the ROI this mask represents.
        proper_name: Optional standardized name for the ROI.
    """

    voidval: float = 0.0
    roi_name: str
    proper_name: str | None

    def __init__(
        self,
        array: NDArray[np.int16],
        spatial_metadata: SpatialMetadata,
        study_uid: str,
        frame_of_reference: str,
        patient_id: str,
        roi_name: str,
        proper_name: str | None = None,
    ) -> None:
        """Initialize PatientMask with pre-built array.

        For constructing from DICOM, use from_rtstruct() or the legacy constructor.

        Args:
            array: 3D binary numpy array with shape (Z, Y, X).
            spatial_metadata: Immutable spatial metadata.
            study_uid: DICOM Study Instance UID.
            frame_of_reference: DICOM Frame of Reference UID.
            patient_id: Patient ID.
            roi_name: Name of the ROI.
            proper_name: Optional standardized name.
        """
        super().__init__(
            array=array.astype(np.float64),
            spatial_metadata=spatial_metadata,
            study_uid=study_uid,
            frame_of_reference=frame_of_reference,
            patient_id=patient_id,
        )
        self.roi_name = roi_name
        self.proper_name = proper_name
        self._enforce_binary()

    @property
    def array(self) -> NDArray[np.int16]:
        """Return the underlying binary array."""
        return self._array.astype(np.int16)

    @array.setter
    def array(self, value: NDArray[np.floating]) -> None:
        """Set array and enforce binary [0, 1] values."""
        self._array = value.astype(np.float64)
        self._enforce_binary()

    def _enforce_binary(self) -> None:
        """Enforce that array contains only [0, 1] values."""
        self._array = np.clip(self._array, 0, 1)
        self._array = np.abs(np.round(self._array))

    @classmethod
    def _find_contour_sequence(
        cls,
        ssfile: Dataset,
        roi_name: str,
    ) -> Sequence:
        """Find the ContourSequence for a named ROI.

        Args:
            ssfile: RTSTRUCT pydicom Dataset.
            roi_name: Name of the ROI to find.

        Returns:
            ContourSequence for the ROI.

        Raises:
            ROINotFoundError: If ROI is not found in structure set.
        """
        available_rois = [
            roi_info.ROIName for roi_info in ssfile.StructureSetROISequence
        ]

        ref_num = None
        for roi_info in ssfile.StructureSetROISequence:
            if roi_info.ROIName == roi_name:
                ref_num = roi_info.ROINumber
                break

        if ref_num is None:
            raise ROINotFoundError(roi_name, available_rois)

        for data in ssfile.ROIContourSequence:
            if data.ReferencedROINumber == ref_num:
                if hasattr(data, "ContourSequence"):
                    return data.ContourSequence
                raise ROINotFoundError(
                    roi_name,
                    message=f"ROI '{roi_name}' has no ContourSequence",
                )

        raise ROINotFoundError(roi_name, available_rois)

    @classmethod
    def _rasterize_contours(
        cls,
        contourseq: Sequence,
        reference: Alignable,
    ) -> NDArray[np.int16]:
        """Convert contour data to a rasterized mask array.

        Args:
            contourseq: DICOM ContourSequence.
            reference: Reference array for coordinate mapping.

        Returns:
            Rasterized binary mask array.
        """
        array = np.zeros(reference.array.shape, dtype=np.int16)

        for plane in contourseq:
            coords = np.reshape(
                plane.ContourData,
                (int(len(plane.ContourData) / 3), 3),
            )
            for point in coords:
                loc = reference.locate(tuple(point))
                if loc is not None:
                    array[loc] = 1

        for i in range(array.shape[0]):
            mask_slice = array[i]
            if np.sum(mask_slice) == 0:
                continue
            points = np.array(np.where(mask_slice))
            points = np.array([points[1, :], points[0, :]]).T
            array[i] = cv2.fillPoly(
                mask_slice,
                pts=[_sort_coords(points)],
                color=1,
            )

        return array

    @classmethod
    def from_rtstruct(
        cls,
        reference: Alignable,
        ssfile: Dataset,
        roi_name: str,
        proper_name: str | None = None,
    ) -> PatientMask:
        """Create PatientMask from RTSTRUCT and reference array.

        Args:
            reference: Reference PatientArray for coordinate mapping.
            ssfile: RTSTRUCT pydicom Dataset.
            roi_name: Name of the ROI to rasterize.
            proper_name: Optional standardized name for the ROI.

        Returns:
            New PatientMask instance.

        Raises:
            ROINotFoundError: If ROI is not found in structure set.
        """
        if ssfile.StudyInstanceUID != reference.studyUID:
            warnings.warn(
                "Reference file and structure set file StudyUID mismatch",
                UserWarning,
                stacklevel=2,
            )

        contourseq = cls._find_contour_sequence(ssfile, roi_name)
        array = cls._rasterize_contours(contourseq, reference)

        return cls(
            array=array,
            spatial_metadata=reference.spatial_metadata,
            study_uid=str(ssfile.StudyInstanceUID),
            frame_of_reference=reference.FoR,
            patient_id=str(ssfile.PatientID),
            roi_name=roi_name,
            proper_name=proper_name,
        )

    # Legacy constructor for backwards compatibility
    @classmethod
    def _legacy_init(
        cls,
        reference: Alignable,
        ssfile: Dataset,
        roi: str,
        proper_name: str | None = None,
    ) -> PatientMask:
        """Legacy constructor matching original __init__ signature."""
        return cls.from_rtstruct(reference, ssfile, roi, proper_name)

    def __new__(
        cls,
        *args,
        **kwargs,
    ):
        """Support both legacy and new constructor signatures."""
        return object.__new__(cls)

    def join(self, other: PatientMask) -> None:
        """Join another mask into this one (logical OR).

        Args:
            other: Another PatientMask to join.

        Raises:
            TypeError: If other is not a PatientMask.
            ValueError: If shapes don't match.
        """
        if not isinstance(other, PatientMask):
            raise TypeError("Can only join with another PatientMask")
        if self._array.shape != other._array.shape:
            raise ValueError("Cannot join masks with different shapes")

        self._array = self._array + other._array
        self._array[self._array > 0] = 1

    @property
    def com(self) -> NDArray[np.floating]:
        """Calculate center of mass of the mask.

        Returns:
            (Z, Y, X) coordinates of the center of mass.
        """
        livecoords = np.argwhere(self._array)
        return np.sum(livecoords, axis=0) / len(livecoords)


# Backwards-compatible constructor wrapper
_original_init = PatientMask.__init__


def _compat_init(self, *args, **kwargs):
    """Backwards-compatible __init__ supporting both old and new signatures.

    New signature (keyword args): array, spatial_metadata, study_uid,
        frame_of_reference, patient_id, roi_name, proper_name

    Legacy signature (positional): reference, ssfile, roi, proper_name
    """
    # Check for new signature by looking for 'array' keyword
    if "array" in kwargs:
        _original_init(self, **kwargs)
    # Check if first positional arg is an array (new signature via positional)
    elif args and isinstance(args[0], np.ndarray):
        _original_init(self, *args, **kwargs)
    # Legacy signature: reference, ssfile, roi, proper_name
    else:
        # Extract legacy arguments
        if args:
            reference = args[0]
            ssfile = args[1] if len(args) > 1 else kwargs.get("ssfile")
            roi = args[2] if len(args) > 2 else kwargs.get("roi")
            proper_name = args[3] if len(args) > 3 else kwargs.get("proper_name")
        else:
            reference = kwargs.get("reference")
            ssfile = kwargs.get("ssfile")
            roi = kwargs.get("roi")
            proper_name = kwargs.get("proper_name")

        mask = PatientMask.from_rtstruct(
            reference=reference,
            ssfile=ssfile,
            roi_name=roi,
            proper_name=proper_name,
        )
        self._array = mask._array
        self._spatial_metadata = mask._spatial_metadata
        self.studyUID = mask.studyUID
        self.FoR = mask.FoR
        self.patient_id = mask.patient_id
        self.roi_name = mask.roi_name
        self.proper_name = mask.proper_name


PatientMask.__init__ = _compat_init  # type: ignore[method-assign]
