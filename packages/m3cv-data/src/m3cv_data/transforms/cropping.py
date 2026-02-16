"""Anatomical cropping transforms for Patient objects."""

import warnings
from typing import TYPE_CHECKING

import numpy as np

from m3cv_data.transforms.exceptions import ReferenceNotFoundError
from m3cv_data.transforms.reference_strategies import (
    FallbackStrategy,
    ReferenceStrategy,
    VolumeCenterStrategy,
)

if TYPE_CHECKING:
    from m3cv_data.structures import Patient


class AnatomicalCrop:
    """Crop Patient volumes around anatomical reference point.

    Applies 3D cropping to all Patient arrays (CT, dose, structures) centered
    on a reference point calculated by a ReferenceStrategy. If cropping extends
    beyond volume boundaries, arrays are padded with appropriate void values.

    This transform operates on Patient objects before channel stacking, allowing
    access to individual structure masks for anatomical reference calculation.
    """

    def __init__(
        self,
        crop_shape: tuple[int, int, int],
        reference_strategy: ReferenceStrategy,
        allow_fallback: bool = False,
        warn_on_fallback: bool = True,
    ) -> None:
        """Initialize anatomical crop transform.

        Args:
            crop_shape: Target crop dimensions (Z, Y, X) in voxels.
            reference_strategy: Strategy for calculating reference point.
            allow_fallback: If True, fall back to volume center when reference
                structures are missing. If False, raise ReferenceNotFoundError.
            warn_on_fallback: If True, emit warning when fallback is used.
        """
        self._crop_shape = crop_shape
        self._reference_strategy = reference_strategy
        self._allow_fallback = allow_fallback
        self._warn_on_fallback = warn_on_fallback

        # Wrap strategy with fallback if enabled
        if allow_fallback:
            self._strategy = FallbackStrategy(
                [reference_strategy, VolumeCenterStrategy()]
            )
        else:
            self._strategy = reference_strategy

    def __call__(self, patient: "Patient") -> "Patient":
        """Apply anatomical cropping to patient data.

        Args:
            patient: Patient object to crop.

        Returns:
            New Patient object with cropped arrays.

        Raises:
            ReferenceNotFoundError: If reference structures are missing and
                allow_fallback is False.
        """
        # Calculate reference point
        reference = self._strategy.calculate(patient)

        if reference is None:
            # All strategies failed
            required = self._reference_strategy.get_required_structures()
            available = list(patient.structures.keys())
            raise ReferenceNotFoundError(
                patient_id=patient.patient_id,
                missing_structures=required,
                available_structures=available,
                allow_fallback=self._allow_fallback,
            )
        elif self._allow_fallback:
            # Check if primary strategy failed (indicating fallback was used)
            primary_result = self._reference_strategy.calculate(patient)
            if primary_result is None:
                if self._warn_on_fallback:
                    required = self._reference_strategy.get_required_structures()
                    warnings.warn(
                        f"Patient '{patient.patient_id}': Reference structures "
                        f"{required} not found. Using volume center as fallback.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Crop arrays around reference point
        # Standard void values: CT=-1000, dose=0, structures=0
        cropped_ct = None
        if patient.ct is not None:
            cropped_ct = self._crop_array(patient.ct, reference, voidval=-1000.0)

        cropped_dose = None
        if patient.dose is not None:
            cropped_dose = self._crop_array(patient.dose, reference, voidval=0.0)

        cropped_structures = {}
        for name, mask in patient.structures.items():
            cropped_structures[name] = self._crop_array(mask, reference, voidval=0.0)

        # Import Patient here to avoid circular imports
        from m3cv_data.patient import Patient

        # Create new Patient with cropped data
        return Patient(
            patient_id=patient.patient_id,
            ct=cropped_ct,
            dose=cropped_dose,
            structures=cropped_structures,
            source_path=patient.source_path,
            study_uid=patient.study_uid,
            frame_of_reference=patient.frame_of_reference,
            spatial_metadata=patient.spatial_metadata,
        )

    def _crop_array(
        self,
        array: np.ndarray,
        center: tuple[int, int, int],
        voidval: float,
    ) -> np.ndarray:
        """Crop a 3D array around a center point with padding if needed.

        Args:
            array: Input array with shape (Z, Y, X).
            center: Center voxel indices (z, y, x).
            voidval: Value to use for padding if crop extends beyond bounds.

        Returns:
            Cropped array with shape matching crop_shape. Padded with voidval
            if crop extends beyond original array boundaries.
        """
        center_z, center_y, center_x = center
        crop_z, crop_y, crop_x = self._crop_shape

        # Calculate start and end indices for crop
        start_z = center_z - crop_z // 2
        start_y = center_y - crop_y // 2
        start_x = center_x - crop_x // 2

        end_z = start_z + crop_z
        end_y = start_y + crop_y
        end_x = start_x + crop_x

        # Determine if padding is needed
        pad_before = [0, 0, 0]
        pad_after = [0, 0, 0]
        slice_start = [start_z, start_y, start_x]
        slice_end = [end_z, end_y, end_x]

        for dim in range(3):
            if slice_start[dim] < 0:
                pad_before[dim] = -slice_start[dim]
                slice_start[dim] = 0
            if slice_end[dim] > array.shape[dim]:
                pad_after[dim] = slice_end[dim] - array.shape[dim]
                slice_end[dim] = array.shape[dim]

        # Extract crop from array
        cropped = array[
            slice_start[0] : slice_end[0],
            slice_start[1] : slice_end[1],
            slice_start[2] : slice_end[2],
        ]

        # Apply padding if needed
        if any(pad_before) or any(pad_after):
            pad_width = list(zip(pad_before, pad_after, strict=False))
            cropped = np.pad(
                cropped,
                pad_width=pad_width,
                mode="constant",
                constant_values=voidval,
            )

        return cropped
