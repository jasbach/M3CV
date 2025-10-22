import numpy as np
from numpy import ndarray
import scipy.sparse
from pydicom.dataset import Dataset

from m3cv_prep.dicom_utils import validate_fields


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
    dense[rows, cols, slices] = 1
    return dense

# WIP
"""
def build_ct_array(ct_dcms: list[Dataset], void_val: int = -1000):
    shared_fields = [
        "PatientID",
        "StudyInstanceUID",
        "FrameOfReferenceUID",
        "PixelSpacing",
        "SliceThickness",
        "ImageOrientationPatient",
        "Rows",
        "Columns",
        "Modality",
    ]
    validate_fields(ct_dcms, shared_fields)
    slice_positions = []
    for dcm in ct_dcms:
        position = dcm.ImagePositionPatient
        slice_positions.append((position[2], dcm))
    slice_positions.sort(key=lambda x: x[0])
"""