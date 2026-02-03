"""Protocols and data structures for the arrays module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


@dataclass(frozen=True)
class SpatialMetadata:
    """Immutable container for spatial metadata of a patient array.

    Attributes:
        position: (X, Y, Z) corner position in patient coordinates.
        pixel_size: (row_spacing, col_spacing) in mm.
        slice_ref: Z positions of each slice.
        slice_thickness: Slice thickness in mm.
        even_spacing: Whether slices are evenly spaced.
    """

    position: tuple[float, float, float]
    pixel_size: tuple[float, float]
    slice_ref: tuple[float, ...]
    slice_thickness: float
    even_spacing: bool

    @classmethod
    def from_dicom(
        cls,
        ref_file: Dataset,
        z_positions: list[float] | tuple[float, ...],
    ) -> SpatialMetadata:
        """Create SpatialMetadata from a reference DICOM file.

        Args:
            ref_file: A pydicom Dataset with ImagePositionPatient and PixelSpacing.
            z_positions: Sorted list of Z positions for all slices.

        Returns:
            A new SpatialMetadata instance.
        """
        position = tuple(float(p) for p in ref_file.ImagePositionPatient)
        pixel_size = tuple(float(p) for p in ref_file.PixelSpacing)

        z_tuple = tuple(z_positions)
        diffs = np.diff(z_tuple)
        unique_diffs = np.unique(diffs)
        even_spacing = len(unique_diffs) == 1

        if even_spacing:
            slice_thickness = float(unique_diffs[0])
        elif hasattr(ref_file, "SliceThickness"):
            slice_thickness = float(ref_file.SliceThickness)
        else:
            slice_thickness = float(np.median(diffs))

        return cls(
            position=position,  # type: ignore[arg-type]
            pixel_size=pixel_size,  # type: ignore[arg-type]
            slice_ref=z_tuple,
            slice_thickness=slice_thickness,
            even_spacing=even_spacing,
        )

    @classmethod
    def from_dose(cls, dcm: Dataset) -> SpatialMetadata:
        """Create SpatialMetadata from a RTDOSE DICOM file.

        Args:
            dcm: A pydicom Dataset with dose grid information.

        Returns:
            A new SpatialMetadata instance.
        """
        position = tuple(float(p) for p in dcm.ImagePositionPatient)
        pixel_size = tuple(float(p) for p in dcm.PixelSpacing)

        offset = np.array(dcm.GridFrameOffsetVector)
        z_positions = tuple(float(z) for z in (offset + position[2]))

        diffs = np.diff(z_positions)
        unique_diffs = np.unique(diffs)
        even_spacing = len(unique_diffs) == 1

        if even_spacing:
            slice_thickness = float(unique_diffs[0])
        else:
            slice_thickness = float(np.median(diffs))

        return cls(
            position=position,  # type: ignore[arg-type]
            pixel_size=pixel_size,  # type: ignore[arg-type]
            slice_ref=z_positions,
            slice_thickness=slice_thickness,
            even_spacing=even_spacing,
        )


@runtime_checkable
class Alignable(Protocol):
    """Protocol for objects that can be spatially aligned."""

    @property
    def spatial_metadata(self) -> SpatialMetadata:
        """Return the spatial metadata for this object."""
        ...

    @property
    def array(self) -> NDArray[np.floating]:
        """Return the underlying array."""
        ...

    @property
    def voidval(self) -> float:
        """Return the void/padding value for this array type."""
        ...

    def align_with(self, other: Alignable) -> None:
        """Align this array with another Alignable object."""
        ...
