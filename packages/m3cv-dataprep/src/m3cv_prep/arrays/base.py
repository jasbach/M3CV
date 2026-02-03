"""Base class for patient arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from m3cv_prep.arrays.exceptions import SliceCompatibilityError, UnevenSpacingError
from m3cv_prep.arrays.protocols import Alignable, SpatialMetadata

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


def check_slice_compatibility(
    source_slice_ref: tuple[float, ...],
    target_slice_ref: tuple[float, ...],
    source_slice_thickness: float,
    target_slice_thickness: float,
) -> tuple[list[float], list[float]]:
    """Check that source slices align with and are contiguous within target slices.

    Validates two conditions:
    1. Each source slice WITHIN the target extent aligns with a target slice
    2. The aligned target slices form a contiguous block (no gaps)

    Source slices outside the target extent are ignored - they will be trimmed
    during alignment, which is expected behavior. Similarly, target slices outside
    the source extent will be padded with voidval, which is also expected.

    The contiguity check prevents scenarios where source slices are interleaved
    through the target volume (e.g., source at Z=0,6,12 with target at Z=0,3,6,9,12),
    which would leave intermediate target slices with incorrect voidval data.

    Args:
        source_slice_ref: Z positions of source slices (e.g., dose).
        target_slice_ref: Z positions of target slices (e.g., CT).
        source_slice_thickness: Slice thickness of source array.
        target_slice_thickness: Slice thickness of target array.

    Returns:
        Tuple of (misaligned_slices, missing_slices):
        - misaligned_slices: Source Z positions within target extent that don't
          align with any target slice
        - missing_slices: Target Z positions that fall within source's coverage
          but have no corresponding source slice (indicating gaps/interleaving)
    """
    tolerance = min(source_slice_thickness, target_slice_thickness) * 0.1
    target_z = np.array(target_slice_ref)
    target_z_min = float(np.min(target_z))
    target_z_max = float(np.max(target_z))

    # Check 1: Each source slice WITHIN target extent must align with a target slice
    # Source slices outside target extent will be trimmed, which is fine
    misaligned = []
    aligned_target_indices = []
    for z in source_slice_ref:
        # Skip source slices outside target extent - they'll be trimmed
        if z < target_z_min - tolerance or z > target_z_max + tolerance:
            continue

        distances = np.abs(target_z - z)
        min_idx = np.argmin(distances)
        if distances[min_idx] > tolerance:
            misaligned.append(z)
        else:
            aligned_target_indices.append(min_idx)

    # Check 2: Aligned target indices must be contiguous (no gaps)
    missing = []
    if aligned_target_indices and not misaligned:
        aligned_target_indices.sort()
        expected_range = range(
            aligned_target_indices[0], aligned_target_indices[-1] + 1
        )
        for idx in expected_range:
            if idx not in aligned_target_indices:
                missing.append(float(target_z[idx]))

    return misaligned, missing


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

    def align_with(self, other: Alignable, strict_slice_alignment: bool = True) -> None:
        """Align this array to match another array's spatial extent.

        Args:
            other: Another Alignable object to align with.
            strict_slice_alignment: If True (default), verify that this array's
                slices align with the target's slices and form a contiguous block.
                This prevents silent data misrepresentation when slice grids differ.

        Raises:
            UnevenSpacingError: If either array has uneven slice spacing.
            SliceCompatibilityError: If strict_slice_alignment is True and slices
                don't align or aren't contiguous within the target volume.
        """
        if not (self.even_spacing and other.spatial_metadata.even_spacing):
            raise UnevenSpacingError()

        if strict_slice_alignment:
            misaligned, missing = check_slice_compatibility(
                self.slice_ref,
                other.spatial_metadata.slice_ref,
                self.slice_thickness,
                other.spatial_metadata.slice_thickness,
            )
            if misaligned:
                raise SliceCompatibilityError(
                    f"Source slices at Z={misaligned} don't align with any target slice. "
                    "Set strict_slice_alignment=False to bypass this check.",
                    misaligned_slices=misaligned,
                )
            if missing:
                raise SliceCompatibilityError(
                    f"Target slices at Z={missing} have no corresponding source slice. "
                    "This indicates non-contiguous coverage that would leave gaps. "
                    "Set strict_slice_alignment=False to bypass this check.",
                    missing_slices=missing,
                )

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
