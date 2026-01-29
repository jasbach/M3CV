"""Base class for patient arrays."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy.ndimage.interpolation as scipy_mods
from numpy.typing import NDArray

from m3cv_prep.arrays.exceptions import UnevenSpacingError
from m3cv_prep.arrays.protocols import Alignable, SpatialMetadata

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


class PatientArray:
    """Base class to support array operations for CT, mask, dose patient arrays.

    Provides common operations for all modality-specific array types.
    Subclasses should set voidval and implement appropriate __init__ methods.

    Attributes:
        array: The 3D numpy array storing voxel data (Z, Y, X ordering).
        voidval: The value used for void/padding regions.
        studyUID: DICOM Study Instance UID.
        FoR: Frame of Reference UID.
        patient_id: Patient ID from DICOM.
    """

    voidval: float = 0.0
    _spatial_metadata: SpatialMetadata
    _array: NDArray[np.floating]

    def __init__(
        self,
        array: NDArray[np.floating],
        spatial_metadata: SpatialMetadata,
        study_uid: str,
        frame_of_reference: str,
        patient_id: str,
    ) -> None:
        """Initialize PatientArray with required metadata.

        Args:
            array: 3D numpy array with shape (Z, Y, X).
            spatial_metadata: Immutable spatial metadata for the array.
            study_uid: DICOM Study Instance UID.
            frame_of_reference: DICOM Frame of Reference UID.
            patient_id: Patient ID.
        """
        self._array = array
        self._spatial_metadata = spatial_metadata
        self.studyUID = study_uid
        self.FoR = frame_of_reference
        self.patient_id = patient_id

    @classmethod
    def _init_from_ref_file(cls, ref_file: Dataset) -> dict[str, str]:
        """Extract common metadata from a reference DICOM file.

        Args:
            ref_file: A pydicom Dataset.

        Returns:
            Dictionary with study_uid, frame_of_reference, patient_id.
        """
        return {
            "study_uid": str(ref_file.StudyInstanceUID),
            "frame_of_reference": str(ref_file.FrameOfReferenceUID),
            "patient_id": str(ref_file.PatientID),
        }

    @property
    def spatial_metadata(self) -> SpatialMetadata:
        """Return the immutable spatial metadata."""
        return self._spatial_metadata

    @property
    def array(self) -> NDArray[np.floating]:
        """Return the underlying array."""
        return self._array

    @array.setter
    def array(self, value: NDArray[np.floating]) -> None:
        """Set the underlying array."""
        self._array = value

    @property
    def position(self) -> tuple[float, float, float]:
        """Return (X, Y, Z) corner position."""
        return self._spatial_metadata.position

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Return (row_spacing, col_spacing) in mm."""
        return self._spatial_metadata.pixel_size

    @property
    def slice_ref(self) -> tuple[float, ...]:
        """Return Z positions of each slice."""
        return self._spatial_metadata.slice_ref

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness in mm."""
        return self._spatial_metadata.slice_thickness

    @property
    def even_spacing(self) -> bool:
        """Return whether slices are evenly spaced."""
        return self._spatial_metadata.even_spacing

    @property
    def rows(self) -> int:
        """Return number of rows (Y dimension)."""
        return self._array.shape[1]

    @property
    def columns(self) -> int:
        """Return number of columns (X dimension)."""
        return self._array.shape[2]

    @property
    def height(self) -> float:
        """Return physical height in mm."""
        return self.rows * self.pixel_size[0]

    @property
    def width(self) -> float:
        """Return physical width in mm."""
        return self.columns * self.pixel_size[1]

    def _update_spatial_metadata(
        self,
        position: tuple[float, float, float] | None = None,
        pixel_size: tuple[float, float] | None = None,
        slice_ref: tuple[float, ...] | None = None,
        slice_thickness: float | None = None,
        even_spacing: bool | None = None,
    ) -> None:
        """Update spatial metadata by creating a new immutable instance.

        Args:
            position: New position, or None to keep current.
            pixel_size: New pixel size, or None to keep current.
            slice_ref: New slice reference, or None to keep current.
            slice_thickness: New slice thickness, or None to keep current.
            even_spacing: New even spacing flag, or None to keep current.
        """
        self._spatial_metadata = SpatialMetadata(
            position=position
            if position is not None
            else self._spatial_metadata.position,
            pixel_size=pixel_size
            if pixel_size is not None
            else self._spatial_metadata.pixel_size,
            slice_ref=slice_ref
            if slice_ref is not None
            else self._spatial_metadata.slice_ref,
            slice_thickness=slice_thickness
            if slice_thickness is not None
            else self._spatial_metadata.slice_thickness,
            even_spacing=even_spacing
            if even_spacing is not None
            else self._spatial_metadata.even_spacing,
        )

    def rescale(self, new_pix_size: float | tuple[float, float]) -> None:
        """Rescale the array to a new pixel size.

        Args:
            new_pix_size: New pixel size as scalar (isotropic) or (row, col) tuple.
        """
        if isinstance(new_pix_size, int | float):
            new_pix_size = (float(new_pix_size), float(new_pix_size))
        else:
            new_pix_size = (float(new_pix_size[0]), float(new_pix_size[1]))

        row_scaling = self.pixel_size[0] / new_pix_size[0]
        col_scaling = self.pixel_size[1] / new_pix_size[1]
        new_cols = round(self.columns * col_scaling)
        new_rows = round(self.rows * row_scaling)
        new_array = np.zeros((self._array.shape[0], new_rows, new_cols))

        for i, img in enumerate(self._array):
            rescaled = cv2.resize(
                img,
                (new_cols, new_rows),
                interpolation=cv2.INTER_AREA,
            )
            new_array[i, :, :] = rescaled

        self._array = new_array
        self._update_spatial_metadata(pixel_size=new_pix_size)

    def locate(self, coord: tuple[float, float, float]) -> tuple[int, int, int] | None:
        """Find voxel indices for a real-space coordinate.

        Args:
            coord: (X, Y, Z) coordinate in patient space.

        Returns:
            (z_idx, y_idx, x_idx) indices, or None if outside array bounds.
        """
        x, y, z = coord
        slice_ref_array = np.array(self.slice_ref)

        if any(
            (
                z < np.amin(slice_ref_array),
                z > np.amax(slice_ref_array),
                x < self.position[0],
                x > (self.position[0] + self.columns * self.pixel_size[1]),
                y < self.position[1],
                y > (self.position[1] + self.rows * self.pixel_size[0]),
            )
        ):
            return None

        z_idx = int(round(np.argmin(np.abs(slice_ref_array - z))))
        y_idx = int(round((y - self.position[1]) / self.pixel_size[0]))
        x_idx = int(round((x - self.position[0]) / self.pixel_size[1]))
        return (z_idx, y_idx, x_idx)

    def align_with(self, other: Alignable) -> None:
        """Align this array to match another array's spatial extent.

        Args:
            other: Another Alignable object to align with.

        Raises:
            UnevenSpacingError: If either array has uneven slice spacing.
        """
        if not (self.even_spacing and other.spatial_metadata.even_spacing):
            raise UnevenSpacingError()

        if other.spatial_metadata.pixel_size != self.pixel_size:
            self.rescale(other.spatial_metadata.pixel_size)

        voxel_size = [
            float(self.pixel_size[1]),  # X
            float(self.pixel_size[0]),  # Y
            float(self.slice_thickness),  # Z
        ]

        front_pad = []
        other_pos = other.spatial_metadata.position
        for p_s, p_o, size in zip(self.position, other_pos, voxel_size, strict=False):
            voxel_steps = round((p_s - p_o) / size)
            front_pad.append(voxel_steps)

        back_pad = [
            other.array.shape[2] - (self.columns + front_pad[0]),  # X
            other.array.shape[1] - (self.rows + front_pad[1]),  # Y
            len(other.array) - (len(self._array) + front_pad[2]),  # Z
        ]

        pad_value = self.voidval
        new_position = list(self.position)
        new_slice_ref = list(self.slice_ref)

        # Trim front if needed
        if front_pad[0] < 0:
            self._array = self._array[:, :, -front_pad[0] :]
            new_position[0] -= front_pad[0] * self.pixel_size[1]
            front_pad[0] = 0
        if front_pad[1] < 0:
            self._array = self._array[:, -front_pad[1] :, :]
            new_position[1] -= front_pad[1] * self.pixel_size[0]
            front_pad[1] = 0
        if front_pad[2] < 0:
            self._array = self._array[-front_pad[2] :, :, :]
            new_slice_ref = new_slice_ref[-front_pad[2] :]
            new_position[2] -= front_pad[2] * self.slice_thickness
            front_pad[2] = 0

        # Trim back if needed
        if back_pad[0] < 0:
            self._array = self._array[:, :, : back_pad[0]]
            back_pad[0] = 0
        if back_pad[1] < 0:
            self._array = self._array[:, : back_pad[1], :]
            back_pad[1] = 0
        if back_pad[2] < 0:
            self._array = self._array[: back_pad[2], :, :]
            new_slice_ref = new_slice_ref[: back_pad[2]]
            back_pad[2] = 0

        # Pad to match
        padarg = [
            (front, back) for front, back in zip(front_pad, back_pad, strict=False)
        ]
        padarg.reverse()  # Shape is (Z, Y, X)

        self._array = np.pad(
            self._array,
            pad_width=padarg,
            constant_values=pad_value,
        )

        position_adjust = np.array(front_pad) * np.array(voxel_size)
        final_position = tuple(np.array(new_position) - position_adjust)

        self._update_spatial_metadata(
            position=final_position,  # type: ignore[arg-type]
            slice_ref=other.spatial_metadata.slice_ref,
        )

    def bounding_box(
        self,
        shape: tuple[int, ...],
        center: tuple[float, float, float] | None = None,
    ) -> NDArray[np.floating]:
        """Extract a bounding box from the array.

        Args:
            shape: Desired shape as (Z, Y, X) or (Y, X) for 2D.
            center: Center indices (Z, Y, X), or None for array center.

        Returns:
            Cropped array of the specified shape.
        """
        if len(shape) == 2:
            shape = (self._array.shape[0], shape[0], shape[1])

        if center is None:
            center_idx = [
                self._array.shape[0] // 2,
                self._array.shape[1] // 2,
                self._array.shape[2] // 2,
            ]
        else:
            center_idx = [round(pos) for pos in center]

        start = [
            center_idx[0] - (shape[0] // 2),
            center_idx[1] - (shape[1] // 2),
            center_idx[2] - (shape[2] // 2),
        ]

        return self._array[
            start[0] : start[0] + shape[0],
            start[1] : start[1] + shape[1],
            start[2] : start[2] + shape[2],
        ]

    def rotate(
        self,
        degree_range: float = 15.0,
        rng: np.random.Generator | None = None,
        degrees: float | None = None,
    ) -> None:
        """Rotate array about the Z axis.

        Args:
            degree_range: Maximum degrees +/- of rotation. Ignored if degrees specified.
            rng: NumPy random Generator for reproducibility. If None, creates new one.
            degrees: Explicit rotation angle, overrides random generation.
        """
        if not hasattr(self, "original"):
            self.original = copy.deepcopy(self._array)

        if rng is None:
            rng = np.random.default_rng()

        if degrees is None:
            intensity = rng.random()
            degrees = intensity * degree_range * 2 - degree_range

        self._array = scipy_mods.rotate(
            self._array,
            angle=degrees,
            axes=(1, 2),
            reshape=False,
            mode="constant",
            cval=self.voidval,
        )

    def shift(
        self,
        max_shift: float = 0.2,
        rng: np.random.Generator | None = None,
        pixelshift: tuple[int, int] | None = None,
    ) -> None:
        """Shift array along Y and X dimensions.

        Args:
            max_shift: Maximum shift as fraction of array size (0.0 to 1.0).
            rng: NumPy random Generator for reproducibility.
            pixelshift: Explicit (Y, X) pixel shift, overrides random generation.
        """
        if not hasattr(self, "original"):
            self.original = copy.deepcopy(self._array)

        max_y_pix = max_shift * self.rows
        max_x_pix = max_shift * self.columns

        if rng is None:
            rng = np.random.default_rng()

        if pixelshift is None:
            y_intensity = rng.random()
            x_intensity = rng.random()
            yshift = round(y_intensity * max_y_pix * 2 - max_y_pix)
            xshift = round(x_intensity * max_x_pix * 2 - max_x_pix)
            shiftspec = (0, yshift, xshift)
        else:
            shiftspec = (0, pixelshift[0], pixelshift[1])

        self._array = scipy_mods.shift(
            self._array,
            shift=shiftspec,
            mode="constant",
            cval=self.voidval,
        )

    def zoom(
        self,
        max_zoom_factor: float = 0.2,
        rng: np.random.Generator | None = None,
        zoom_factor: float | None = None,
    ) -> None:
        """Zoom array uniformly in all dimensions.

        Args:
            max_zoom_factor: Maximum zoom deviation from 1.0 (e.g., 0.2 means 0.8-1.2x).
            rng: NumPy random Generator for reproducibility.
            zoom_factor: Explicit zoom factor, overrides random generation.
        """
        if not hasattr(self, "original"):
            self.original = copy.deepcopy(self._array)

        if rng is None:
            rng = np.random.default_rng()

        if zoom_factor is None:
            intensity = rng.random()
            zoom_factor = 1 + intensity * max_zoom_factor * 2 - max_zoom_factor

        original_shape = self._array.shape

        self._array = scipy_mods.zoom(
            self._array,
            zoom=[zoom_factor, zoom_factor, zoom_factor],
            mode="constant",
            cval=self.voidval,
        )

        if zoom_factor > 1.0:
            self._array = self.bounding_box(original_shape)
        elif zoom_factor < 1.0:
            diffs = np.array(original_shape) - np.array(self._array.shape)
            diffs = np.round(diffs / 2).astype(int)
            pad_spec = [
                (diffs[0], diffs[0]),
                (diffs[1], diffs[1]),
                (diffs[2], diffs[2]),
            ]
            self._array = np.pad(
                self._array,
                pad_width=pad_spec,
                mode="constant",
                constant_values=self.voidval,
            )

    def reset_augments(self) -> None:
        """Reset array to pre-augmentation state."""
        if hasattr(self, "original"):
            self._array = copy.deepcopy(self.original)
            del self.original
