import numpy as np
from numpy import ndarray
import scipy.sparse
from pydicom.dataset import Dataset

from m3cv_prep.dicom_utils import validate_fields, check_ROI_exists
from m3cv_prep.arrayclasses import PatientCT, PatientDose, PatientMask


def pack_array_sparsely(array: ndarray):
    """Convert a 3D mask array into sparse row, col, slice arrays for storage.
    
    Args:
        mask (ndarray): 3D numpy array representing the mask.
    Returns:
        tuple: Three 1D numpy arrays representing the row indices, column indices,
               and slice indices of the non-zero elements in the mask.
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
        rows: ndarray,
        cols: ndarray,
        slices: ndarray,
        refshape: tuple[int, int, int]
        ) -> ndarray:
    """Convert sparse row, col, slice arrays back into a 3D mask array.

    Args:
        rows (ndarray): 1D array of row indices.
        cols (ndarray): 1D array of column indices.
        slices (ndarray): 1D array of slice indices.
        refshape (tuple[int, int, int]): The shape of the original 3D array.

    Returns:
        ndarray: The reconstructed 3D array.
    """
    dense = np.zeros(refshape, dtype=float)
    dense[slices, rows, cols] = 1
    return dense

def construct_arrays(grouped_dcms, structure_dict: dict[str, list] | None = None):
    ct_array = PatientCT(grouped_dcms['CT'])
    dose_array = None
    structure_masks = None
    if 'RTDOSE' in grouped_dcms:
        dose_array = PatientDose(grouped_dcms['RTDOSE'])
        dose_array.align_with(ct_array)
    if 'RTSTRUCT' in grouped_dcms and structure_dict is not None:
        if len(grouped_dcms['RTSTRUCT']) > 1:
            raise ValueError("Multiple RTSTRUCT files found; please provide only one.")
        structure_masks = {}
        for key, value in structure_dict.items():
            for v in value:
                # check if this ROI name is in the RTSTRUCT
                if check_ROI_exists(grouped_dcms['RTSTRUCT'][0], v):
                    structure_masks[key] = PatientMask(
                        reference=ct_array,
                        ssfile=grouped_dcms['RTSTRUCT'][0],
                        roi=v
                    )
                    break
            if key not in structure_masks:
                print(f"[yellow]Warning: None of the ROI names {value} were found in the RTSTRUCT.[/yellow]")
    return ct_array, dose_array, structure_masks