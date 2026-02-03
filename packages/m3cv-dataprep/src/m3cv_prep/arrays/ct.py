"""PatientCT class for CT image arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from m3cv_prep.arrays.base import PatientArray
from m3cv_prep.arrays.exceptions import MetadataMismatchError
from m3cv_prep.arrays.protocols import SpatialMetadata
from m3cv_prep.dicom_utils import getscaledimg
from m3cv_prep.dicom_utils import window_level as wl_func

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


class PatientCT(PatientArray):
    """Patient CT image array.

    Array is stored as (Z, Y, X) with values in Hounsfield Units.
    Pixel spacing is (row_spacing, col_spacing) while position is (X, Y, Z).

    Attributes:
        voidval: Void value is -1000 HU (air).
    """

    voidval: float = -1000.0

    def __init__(
        self,
        array: NDArray[np.floating],
        spatial_metadata: SpatialMetadata,
        study_uid: str,
        frame_of_reference: str,
        patient_id: str,
        slice_thickness: float | None = None,
    ) -> None:
        """Initialize PatientCT with pre-built array.

        For constructing from DICOM files, use from_dicom_files() instead.

        Args:
            array: 3D numpy array with shape (Z, Y, X) in HU.
            spatial_metadata: Immutable spatial metadata.
            study_uid: DICOM Study Instance UID.
            frame_of_reference: DICOM Frame of Reference UID.
            patient_id: Patient ID.
            slice_thickness: Optional override for slice thickness from metadata.
        """
        super().__init__(
            array=array,
            spatial_metadata=spatial_metadata,
            study_uid=study_uid,
            frame_of_reference=frame_of_reference,
            patient_id=patient_id,
        )
        if slice_thickness is not None:
            self._update_spatial_metadata(slice_thickness=slice_thickness)

    @classmethod
    def from_dicom_files(cls, filelist: list[Dataset]) -> PatientCT:
        """Create PatientCT from a list of CT DICOM files.

        Args:
            filelist: List of pydicom Dataset objects for CT slices.

        Returns:
            New PatientCT instance.

        Raises:
            MetadataMismatchError: If CT files have incompatible metadata.
            ValueError: If filelist is empty.
        """
        if not filelist:
            raise ValueError("filelist cannot be empty")

        ref_file = filelist[0]
        refrows = ref_file.Rows
        refcols = ref_file.Columns

        enforcement = ["StudyInstanceUID", "FrameOfReferenceUID", "SliceThickness"]
        for dicom_attr in enforcement:
            unique = {getattr(file, dicom_attr) for file in filelist}
            if len(unique) != 1:
                raise MetadataMismatchError(
                    attribute=dicom_attr,
                    message=f"Incompatible metadata in CT files for {dicom_attr}",
                )

        zlist: list[tuple[float, Dataset]] = []
        for file in filelist:
            zlist.append((float(file.ImagePositionPatient[-1]), file))

        sortedlist = sorted(zlist, key=lambda x: x[0])
        z_positions = [tup[0] for tup in sortedlist]

        array = np.zeros((len(filelist), refcols, refrows), dtype=np.float64)
        for i, (_z, file) in enumerate(sortedlist):
            array[i, :, :] = getscaledimg(file)

        first_file = sortedlist[0][1]
        spatial_metadata = SpatialMetadata.from_dicom(first_file, z_positions)
        metadata = cls._init_from_ref_file(first_file)

        return cls(
            array=array,
            spatial_metadata=spatial_metadata,
            slice_thickness=float(ref_file.SliceThickness),
            **metadata,
        )

    def window_level(
        self,
        window: float,
        level: float,
        normalize: bool = False,
    ) -> None:
        """Apply window/level transformation to the CT array.

        Args:
            window: Window width in HU.
            level: Window center in HU.
            normalize: If True, normalize output to [0.0, 1.0].
        """
        self._array = wl_func(self._array, window, level, normalize)
