"""Array tools for packing/unpacking and constructing patient arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from numpy.typing import NDArray

from m3cv_prep.arrays import PatientCT, PatientDose, PatientMask

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


def pack_array_sparsely(array: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Convert a 3D mask array into sparse row, col, slice arrays for storage.

    Args:
        array: 3D numpy array representing the mask.

    Returns:
        Tuple of (row, col, slices) 1D numpy arrays representing the
        indices of the non-zero elements.
    """
    row = np.array([], dtype=int)
    col = np.array([], dtype=int)
    slices = np.array([], dtype=int)
    for i in range(array.shape[0]):
        sp = scipy.sparse.coo_matrix(array[i, ...])
        sl = np.full(sp.data.shape, fill_value=i, dtype=np.int32)
        slices = np.concatenate([slices, sl])
        col = np.concatenate([col, sp.col])
        row = np.concatenate([row, sp.row])
    return row, col, slices


def unpack_sparse_array(
    rows: NDArray,
    cols: NDArray,
    slices: NDArray,
    refshape: tuple[int, int, int],
) -> NDArray:
    """Convert sparse row, col, slice arrays back into a 3D mask array.

    Args:
        rows: 1D array of row indices.
        cols: 1D array of column indices.
        slices: 1D array of slice indices.
        refshape: The shape of the original 3D array.

    Returns:
        The reconstructed 3D array.
    """
    dense = np.zeros(refshape, dtype=float)
    dense[rows, cols, slices] = 1
    return dense


def construct_arrays(
    grouped_dcms: dict[str, list[Dataset]],
    structure_names: list[str] | None = None,
) -> tuple[PatientCT, PatientDose | None, dict[str, PatientMask] | None]:
    """Construct patient arrays from grouped DICOM files.

    Args:
        grouped_dcms: Dictionary mapping modality to list of DICOM Datasets.
        structure_names: List of ROI names to create masks for.

    Returns:
        Tuple of (ct_array, dose_array, structure_masks).
        dose_array is None if no RTDOSE files provided.
        structure_masks is None if no RTSTRUCT or structure_names provided.

    Raises:
        ValueError: If multiple RTSTRUCT files found.
    """
    ct_array = PatientCT.from_dicom_files(grouped_dcms["CT"])
    dose_array = None
    structure_masks = None

    if "RTDOSE" in grouped_dcms:
        dose_array = PatientDose.from_dicom(grouped_dcms["RTDOSE"])
        dose_array.align_with(ct_array)

    if "RTSTRUCT" in grouped_dcms and structure_names is not None:
        if len(grouped_dcms["RTSTRUCT"]) > 1:
            raise ValueError("Multiple RTSTRUCT files found; please provide only one.")
        structure_masks = {}
        for name in structure_names:
            structure_masks[name] = PatientMask.from_rtstruct(
                reference=ct_array,
                ssfile=grouped_dcms["RTSTRUCT"][0],
                roi_name=name,
            )

    return ct_array, dose_array, structure_masks
