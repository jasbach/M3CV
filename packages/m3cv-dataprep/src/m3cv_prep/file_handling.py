from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pydicom
from numpy import ndarray
from pydicom.dataset import Dataset

from m3cv_prep.array_tools import pack_array_sparsely, unpack_sparse_array
from m3cv_prep.dicom_utils import validate_patientid
from m3cv_prep.log_util import get_logger

if TYPE_CHECKING:
    from m3cv_prep.arrays import PatientCT, PatientDose, PatientMask, SpatialMetadata

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


def sort_dcms_by_modality(dcms: list[Dataset]) -> dict[str, list[Dataset]]:
    modality_dict = {}
    for dcm in dcms:
        modality = dcm.Modality
        if modality not in modality_dict:
            modality_dict[modality] = []
        modality_dict[modality].append(dcm)
    return modality_dict


def _save_spatial_metadata(group: h5py.Group, spatial: SpatialMetadata) -> None:
    """Save SpatialMetadata to an HDF5 group as attributes and datasets.

    Args:
        group: HDF5 group to save metadata into.
        spatial: SpatialMetadata instance to save.
    """
    group.attrs["position_x"] = spatial.position[0]
    group.attrs["position_y"] = spatial.position[1]
    group.attrs["position_z"] = spatial.position[2]
    group.attrs["pixel_size_row"] = spatial.pixel_size[0]
    group.attrs["pixel_size_col"] = spatial.pixel_size[1]
    group.attrs["slice_thickness"] = spatial.slice_thickness
    group.attrs["even_spacing"] = spatial.even_spacing
    # slice_ref can be large, store as dataset
    group.create_dataset("slice_ref", data=np.array(spatial.slice_ref))


def _load_spatial_metadata(group: h5py.Group) -> SpatialMetadata:
    """Load SpatialMetadata from an HDF5 group.

    Args:
        group: HDF5 group containing spatial metadata.

    Returns:
        Reconstructed SpatialMetadata instance.
    """
    from m3cv_prep.arrays import SpatialMetadata

    return SpatialMetadata(
        position=(
            float(group.attrs["position_x"]),
            float(group.attrs["position_y"]),
            float(group.attrs["position_z"]),
        ),
        pixel_size=(
            float(group.attrs["pixel_size_row"]),
            float(group.attrs["pixel_size_col"]),
        ),
        slice_ref=tuple(float(z) for z in group["slice_ref"][...]),
        slice_thickness=float(group.attrs["slice_thickness"]),
        even_spacing=bool(group.attrs["even_spacing"]),
    )


def save_array_to_h5(
    out_path: str,
    ct_array: PatientCT | ndarray,
    dose_array: PatientDose | ndarray | None = None,
    structure_masks: dict[str, PatientMask | ndarray] | None = None,
    metadata: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Save CT array, optional dose array, and optional structure masks to an HDF5 file.

    If PatientCT/PatientDose/PatientMask objects are provided, their spatial metadata
    and patient identifiers are also saved. This enables future incremental updates
    by preserving the coordinate system reference.

    Args:
        out_path: Path to the output HDF5 file.
        ct_array: PatientCT object or 3D numpy array representing the CT scan.
        dose_array: PatientDose object or 3D numpy array for dose distribution.
        structure_masks: Dictionary mapping ROI names to PatientMask objects or arrays.
        metadata: Additional metadata to store as attributes in the HDF5 file.
        overwrite: Whether to overwrite existing file at out_path.

    Raises:
        ValueError: If the shapes of the provided arrays do not match.
        FileExistsError: If file exists and overwrite is False.
    """
    # Extract patient metadata from CT if available
    patient_id = None
    study_uid = None
    frame_of_reference = None
    spatial_metadata = None

    if hasattr(ct_array, "patient_id"):
        patient_id = ct_array.patient_id
        study_uid = ct_array.studyUID
        frame_of_reference = ct_array.FoR
        spatial_metadata = ct_array.spatial_metadata

    # Extract raw arrays
    ct_data = ct_array.array if hasattr(ct_array, "array") else ct_array
    dose_data = None
    if dose_array is not None:
        dose_data = dose_array.array if hasattr(dose_array, "array") else dose_array

    # Extract mask arrays and handle PatientMask objects
    mask_data = None
    if structure_masks is not None:
        mask_data = {}
        for roi_name, mask in structure_masks.items():
            mask_data[roi_name] = mask.array if hasattr(mask, "array") else mask

    # Verify that all shapes match
    shapes = [ct_data.shape]
    if dose_data is not None:
        shapes.append(dose_data.shape)
    if mask_data is not None:
        shapes.extend(m.shape for m in mask_data.values())
    if len(set(shapes)) > 1:
        raise ValueError("All arrays must have the same shape to be saved together.")

    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"File {out_path} already exists - set overwrite to True to force write."
        )

    with h5py.File(out_path, "w") as f:
        # Save patient identifiers as root attributes
        if patient_id is not None:
            f.attrs["patient_id"] = patient_id
        if study_uid is not None:
            f.attrs["study_uid"] = study_uid
        if frame_of_reference is not None:
            f.attrs["frame_of_reference"] = frame_of_reference

        # Save spatial metadata in dedicated group
        if spatial_metadata is not None:
            spatial_group = f.create_group("spatial_metadata")
            _save_spatial_metadata(spatial_group, spatial_metadata)

        # Save array data
        f.create_dataset("ct", data=ct_data)
        if dose_data is not None:
            f.create_dataset("dose", data=dose_data)
        if mask_data is not None:
            f.create_group("structures")
            for roi_name, mask in mask_data.items():
                group = f.create_group(f"structures/{roi_name}")
                r, c, sl = pack_array_sparsely(mask)
                group.create_dataset("rows", data=r)
                group.create_dataset("cols", data=c)
                group.create_dataset("slices", data=sl)

        # Save any additional user metadata
        if metadata is not None:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_array_from_h5(
    in_path: str,
) -> dict:
    """Load arrays and metadata from an HDF5 file.

    Args:
        in_path: Path to the HDF5 file.

    Returns:
        Dictionary containing:
        - "ct": CT array (ndarray)
        - "dose": Dose array (ndarray), if present
        - "structures": Dict mapping ROI names to mask arrays, if present
        - "patient_id": Patient ID string, if present
        - "study_uid": Study Instance UID, if present
        - "frame_of_reference": Frame of Reference UID, if present
        - "spatial_metadata": SpatialMetadata instance, if present
    """
    data = {}
    with h5py.File(in_path, "r") as f:
        # Load patient identifiers
        if "patient_id" in f.attrs:
            data["patient_id"] = str(f.attrs["patient_id"])
        if "study_uid" in f.attrs:
            data["study_uid"] = str(f.attrs["study_uid"])
        if "frame_of_reference" in f.attrs:
            data["frame_of_reference"] = str(f.attrs["frame_of_reference"])

        # Load spatial metadata
        if "spatial_metadata" in f:
            data["spatial_metadata"] = _load_spatial_metadata(f["spatial_metadata"])

        # Load array data
        data["ct"] = f["ct"][...]
        if "dose" in f:
            data["dose"] = f["dose"][...]
        if "structures" in f:
            data["structures"] = {}
            struct_group = f["structures"]
            for roi_name in struct_group:
                group = struct_group[roi_name]
                r = group["rows"][...]
                c = group["cols"][...]
                sl = group["slices"][...]
                dense_mask = unpack_sparse_array(r, c, sl, refshape=data["ct"].shape)
                data["structures"][roi_name] = dense_mask
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


def update_label_in_h5(h5_file: h5py.File, label: int) -> None:
    """
    Saves the label integer into the HDF5 file attributes.
    """
    h5_file.attrs["label"] = label


def read_label_from_h5(h5_file: h5py.File) -> int | None:
    """
    Reads the label integer from the HDF5 file attributes.
    """
    if "label" not in h5_file.attrs:
        return None
    return int(h5_file.attrs["label"])
