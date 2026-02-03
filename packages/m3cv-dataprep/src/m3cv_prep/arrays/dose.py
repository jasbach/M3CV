"""PatientDose class for dose arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from m3cv_prep.arrays.base import PatientArray
from m3cv_prep.arrays.exceptions import DoseTypeError, MetadataMismatchError
from m3cv_prep.arrays.protocols import SpatialMetadata
from m3cv_prep.dicom_utils import attr_shared, merge_doses

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


class PatientDose(PatientArray):
    """Patient dose array from RTDOSE files.

    Array is stored as (Z, Y, X) with dose values.
    Pixel spacing is (row_spacing, col_spacing) while position is (X, Y, Z).

    Attributes:
        voidval: Void value is 0.0 (no dose).
        dose_units: Units of the dose values (e.g., 'GY', 'CGY').
    """

    voidval: float = 0.0
    dose_units: str

    def __init__(
        self,
        array: NDArray[np.floating],
        spatial_metadata: SpatialMetadata,
        study_uid: str,
        frame_of_reference: str,
        patient_id: str,
        dose_units: str,
    ) -> None:
        """Initialize PatientDose with pre-built array.

        For constructing from DICOM files, use factory methods instead.

        Args:
            array: 3D numpy array with shape (Z, Y, X).
            spatial_metadata: Immutable spatial metadata.
            study_uid: DICOM Study Instance UID.
            frame_of_reference: DICOM Frame of Reference UID.
            patient_id: Patient ID.
            dose_units: Units of dose values.
        """
        super().__init__(
            array=array,
            spatial_metadata=spatial_metadata,
            study_uid=study_uid,
            frame_of_reference=frame_of_reference,
            patient_id=patient_id,
        )
        self.dose_units = dose_units

    @classmethod
    def from_plan_file(cls, dcm: Dataset) -> PatientDose:
        """Create PatientDose from a single PLAN dose file.

        Args:
            dcm: pydicom Dataset with DoseSummationType == 'PLAN'.

        Returns:
            New PatientDose instance.

        Raises:
            DoseTypeError: If the file is not a PLAN dose file.
        """
        if dcm.DoseSummationType != "PLAN":
            raise DoseTypeError(expected="PLAN", actual=dcm.DoseSummationType)

        array = dcm.pixel_array * dcm.DoseGridScaling
        spatial_metadata = SpatialMetadata.from_dose(dcm)
        metadata = cls._init_from_ref_file(dcm)

        return cls(
            array=array,
            spatial_metadata=spatial_metadata,
            dose_units=str(dcm.DoseUnits),
            **metadata,
        )

    @classmethod
    def from_beam_files(cls, dcms: list[Dataset]) -> PatientDose:
        """Create PatientDose from multiple BEAM dose files.

        Args:
            dcms: List of pydicom Dataset objects with DoseSummationType == 'BEAM'.

        Returns:
            New PatientDose instance.

        Raises:
            DoseTypeError: If any file is not a BEAM dose file.
            MetadataMismatchError: If files have incompatible metadata.
            ValueError: If dcms is empty.
        """
        if not dcms:
            raise ValueError("dcms cannot be empty")

        for dcm in dcms:
            if dcm.DoseSummationType != "BEAM":
                raise DoseTypeError(expected="BEAM", actual=dcm.DoseSummationType)

        mismatches = []
        for attr in [
            "StudyInstanceUID",
            "FrameOfReferenceUID",
            "PixelSpacing",
            "GridFrameOffsetVector",
            "Rows",
            "Columns",
            "DoseUnits",
            "DoseGridScaling",
        ]:
            if not attr_shared(dcms, attr):
                mismatches.append(attr)

        if mismatches:
            raise MetadataMismatchError(
                attribute=", ".join(mismatches),
                message=f"Mismatched shape attributes in dose files: {mismatches}",
            )

        array = merge_doses(*dcms)
        ref_file = dcms[0]
        spatial_metadata = SpatialMetadata.from_dose(ref_file)
        metadata = cls._init_from_ref_file(ref_file)

        return cls(
            array=array,
            spatial_metadata=spatial_metadata,
            dose_units=str(ref_file.DoseUnits),
            **metadata,
        )

    @classmethod
    def from_dicom(cls, dcm: Dataset | list[Dataset]) -> PatientDose:
        """Create PatientDose from DICOM, dispatching to appropriate factory.

        This method provides backwards compatibility by automatically detecting
        whether the input is a PLAN file or BEAM files.

        Args:
            dcm: Single pydicom Dataset (PLAN) or list of Datasets (BEAM files).

        Returns:
            New PatientDose instance.

        Raises:
            DoseTypeError: If dose type is unexpected.
            MetadataMismatchError: If BEAM files have incompatible metadata.
        """
        if isinstance(dcm, list):
            if len(dcm) == 1:
                dcm = dcm[0]
            else:
                return cls.from_beam_files(dcm)

        if dcm.DoseSummationType == "PLAN":
            return cls.from_plan_file(dcm)
        elif dcm.DoseSummationType == "BEAM":
            return cls.from_beam_files([dcm])
        else:
            raise DoseTypeError(
                expected="PLAN or BEAM",
                actual=dcm.DoseSummationType,
            )
