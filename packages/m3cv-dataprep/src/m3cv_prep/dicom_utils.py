"""DICOM utility functions for m3cv_prep."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


def validate_patientid(dcms: list[Dataset]) -> None:
    """Validate that all DICOM files belong to the same patient.

    Args:
        dcms: List of pydicom Dataset objects.

    Raises:
        ValueError: If the DICOM files belong to multiple patients.
    """
    if not dcms:
        return
    patient_ids = {dcm.PatientID for dcm in dcms}
    if len(patient_ids) > 1:
        raise ValueError("DICOM files belong to multiple patients.")


def validate_fields(dcms: list[Dataset], fields: list[str]) -> None:
    """Ensure all DICOM files share the same value for specified attributes.

    Args:
        dcms: List of pydicom Dataset objects.
        fields: List of attribute names to validate.

    Raises:
        ValueError: If files don't share the same value for any attribute.
    """
    core_fields = ["PatientID", "StudyInstanceUID"]
    fields = list(set(fields) | set(core_fields))
    errors = []
    for field in fields:
        first_value = getattr(dcms[0], field, None)
        for dcm in dcms[1:]:
            if getattr(dcm, field, None) != first_value:
                errors.append(
                    f"DICOM files do not share the same value for attribute '{field}'."
                )
                break
    if errors:
        raise ValueError(" \n".join(errors))


def group_dcms_by_modality(dcms: list[Dataset]) -> dict[str, list[Dataset]]:
    """Sort DICOM files by their Modality attribute.

    Args:
        dcms: List of pydicom Dataset objects.

    Returns:
        A dictionary mapping modality strings to lists of DICOM Dataset objects.
    """
    modality_dict: dict[str, list[Dataset]] = {}
    for dcm in dcms:
        modality = dcm.Modality
        if modality not in modality_dict:
            modality_dict[modality] = []
        modality_dict[modality].append(dcm)
    return modality_dict


def getscaledimg(file: Dataset) -> NDArray[np.int16]:
    """Extract and scale CT pixel array using RescaleSlope and RescaleIntercept.

    Args:
        file: A pydicom Dataset containing CT image data.

    Returns:
        Scaled pixel array as int16.
    """
    image = file.pixel_array.astype(np.int16)
    slope = file.RescaleSlope
    intercept = file.RescaleIntercept

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return image


def attr_shared(dcms: list[Dataset], attr: str) -> bool:
    """Check if all DICOM files share the same value for an attribute.

    Args:
        dcms: List of pydicom Dataset objects.
        attr: Attribute name to check.

    Returns:
        True if all files have the same value, False otherwise.
    """
    result = True
    for i in range(1, len(dcms)):
        result = getattr(dcms[0], attr) == getattr(dcms[i], attr)
        if result is False:
            break
    return result


def merge_doses(*args: Dataset) -> NDArray[np.floating]:
    """Merge multiple BEAM dose files into a single array.

    Args:
        *args: pydicom Dataset objects representing BEAM dose files.

    Returns:
        Combined dose array.

    Raises:
        TypeError: If any argument is not a pydicom FileDataset.
        ValueError: If files are not RTDOSE, not BEAM type, or have mismatched arrays.
    """
    from pydicom.dataset import FileDataset

    shape = None
    ipp = None
    iop = None
    mergedarray: NDArray[np.floating] | None = None

    for dose in args:
        if not isinstance(dose, FileDataset):
            raise TypeError(
                "Merge doses function can only operate on pydicom FileDataset objects"
            )
        if dose.Modality != "RTDOSE":
            raise ValueError(
                "Merge doses function can only operate on dose file objects."
            )
        if dose.DoseSummationType != "BEAM":
            raise ValueError(
                f"Merge doses only intended to be applied to beam dose files, "
                f"file is {dose.DoseSummationType}"
            )
        if shape is None:
            shape = dose.pixel_array.shape
            ipp = dose.ImagePositionPatient
            iop = dose.ImageOrientationPatient
        else:
            if not all(
                (
                    dose.pixel_array.shape == shape,
                    dose.ImagePositionPatient == ipp,
                    dose.ImageOrientationPatient == iop,
                )
            ):
                raise ValueError("Mismatched arrays - cannot merge dose files")

        if mergedarray is None:
            mergedarray = dose.pixel_array * dose.DoseGridScaling
        else:
            mergedarray += dose.pixel_array * dose.DoseGridScaling

    if mergedarray is None:
        raise ValueError("No dose files provided")
    return mergedarray


def window_level(
    array: NDArray[np.number],
    window: float,
    level: float,
    normalize: bool = False,
) -> NDArray[np.floating]:
    """Apply window/level transformation to CT array.

    Args:
        array: Input array (typically CT values in HU).
        window: Window width.
        level: Window center.
        normalize: If True, normalize output to [0.0, 1.0].

    Returns:
        Windowed array.
    """
    array = array.astype(np.float64)
    upper = level + round(window / 2)
    lower = level - round(window / 2)

    array[array > upper] = upper
    array[array < lower] = lower

    if normalize:
        array -= lower
        array = array / window
    return array
