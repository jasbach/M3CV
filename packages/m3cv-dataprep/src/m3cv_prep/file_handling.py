import os
import pydicom
import h5py
import scipy
import numpy as np

from m3cv_prep.logging import get_logger
from m3cv_prep.array_tools import pack_array_sparsely, unpack_sparse_array
from m3cv_prep.dicom_utils import validate_patientid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydicom.dataset import Dataset
    from numpy import ndarray

logger = get_logger("file_handling")

def load_dicom_files_from_directory(directory, validate: bool = True):
    logger.debug(f"Loading DICOM files from directory: {directory}")
    dcms = []
    for file in os.listdir(directory):
        logger.debug(f"Attempting to load file: {file}")
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path):
            try:
                dcm = pydicom.dcmread(full_path)
                dcms.append(dcm)
                logger.debug("Successfully loaded DICOM file.")
            except pydicom.errors.InvalidDicomError:
                logger.debug("File is not a valid DICOM. Skipping.")
                continue
    if validate:
        validate_patientid(dcms)
    return dcms



def sort_dcms_by_modality(dcms: list['Dataset']) -> dict[str, list['Dataset']]:
    modality_dict = {}
    for dcm in dcms:
        modality = dcm.Modality
        if modality not in modality_dict:
            modality_dict[modality] = []
        modality_dict[modality].append(dcm)
    return modality_dict

def save_array_to_h5(
        out_path: str,
        ct_array: ndarray,
        dose_array: ndarray | None = None,
        structure_masks: dict[str, ndarray] | None = None,
        metadata: dict | None = None,
        ):
    # verify that all shapes match
    shapes = [ct_array.shape]
    if dose_array is not None:
        shapes.append(dose_array.shape)
    if structure_masks is not None:
        shapes.extend(mask.shape for mask in structure_masks.values())
    if len(set(shapes)) > 1:
        raise ValueError("All arrays must have the same shape to be saved together.")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('ct', data=ct_array)
        if dose_array is not None:
            f.create_dataset('dose', data=dose_array)
        if structure_masks is not None:
            f.create_group('structures')
            for roi_name, mask in structure_masks.items():
                # create group in which to write sparse array mask data
                group = f.create_group(f'structures/{roi_name}')
                r, c, sl = pack_array_sparsely(mask)
                group.create_dataset('rows', data=r)
                group.create_dataset('cols', data=c)
                group.create_dataset('slices', data=sl)
        if metadata is not None:
            for key, value in metadata.items():
                f.attrs[key] = value

def load_array_from_h5(
        in_path: str,
        ) -> dict[str, ndarray]:
    data = {}
    with h5py.File(in_path, 'r') as f:
        data['ct'] = f['ct'][...]
        if 'dose' in f:
            data['dose'] = f['dose'][...]
        if 'structures' in f:
            data['structures'] = {}
            struct_group = f['structures']
            for roi_name in struct_group:
                group = struct_group[roi_name]
                r = group['rows'][...]
                c = group['cols'][...]
                sl = group['slices'][...]
                dense_mask = unpack_sparse_array(
                    r, c, sl, refshape=data['ct'].shape
                )
                data['structures'][roi_name] = dense_mask
    return data


def save_tabular_data_to_file(
        h5_file: h5py.File,
        table_name: str,
        data: dict,
        ):
    """
    Add tabular data as attributes to an HDF5 file.

    Args:
        h5_file (h5py.File): Open HDF5 file object.
        table_name (str): Name of the table/group to create.
        data (dict): Dictionary containing tabular data.
    """
    if table_name in h5_file:
        group = h5_file[table_name]
    else:
        group = h5_file.create_group(table_name)
    
    for key, value in data.items():
        group.attrs[key] = value

def load_tabular_data_from_file(
        h5_file: h5py.File,
        table_name: str,
        ) -> dict:
    """
    Load tabular data stored as attributes in an HDF5 file.

    Args:
        h5_file (h5py.File): Open HDF5 file object.
        table_name (str): Name of the table/group to read.

    Returns:
        dict: Dictionary containing the tabular data.
    """
    if table_name not in h5_file:
        raise KeyError(f"Table '{table_name}' not found in HDF5 file.")
    
    group = h5_file[table_name]
    data = {key: group.attrs[key] for key in group.attrs}
    return data