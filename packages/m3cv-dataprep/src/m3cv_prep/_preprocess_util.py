"""Preprocessing utility functions.

This module contains domain-specific functions for finding ROIs and
working with contour data. Some functions have been moved to dicom_utils.py
and are re-exported here for backwards compatibility.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy
import scipy.sparse

if TYPE_CHECKING:
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence


class ShapeError(Exception):
    """Raised when array shapes are incompatible."""

    pass


# Domain functions (kept here)


def sort_coords(coords: np.ndarray) -> np.ndarray:
    """Sort contour coordinates clockwise around their centroid.

    Args:
        coords: Array of (x, y) coordinates.

    Returns:
        Sorted array of coordinates.
    """
    origin = coords.mean(axis=0)
    refvec = [0, 1]

    def clockwiseangle_and_dist(point):
        nonlocal origin
        nonlocal refvec
        vector = [point[0] - origin[0], point[1] - origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        return angle, lenvector

    sorted_coords = sorted(coords, key=clockwiseangle_and_dist)
    return np.array(sorted_coords)


def get_contour(ss: Dataset, ROI: str | int) -> Sequence | None:
    """Retrieve ContourSequence for the requested ROI.

    Args:
        ss: RTSTRUCT pydicom Dataset.
        ROI: ROI name (str) or number (int).

    Returns:
        ContourSequence if found, None otherwise.
    """
    try:
        ROI_num = int(ROI)
    except ValueError:
        ROI_num = None
        for info in ss.StructureSetROISequence:
            if info.ROIName == ROI:
                ROI_num = info.ROINumber
                break

    if ROI_num is None:
        return None

    for contourseq in ss.ROIContourSequence:
        if contourseq.ReferencedROINumber == ROI_num:
            if hasattr(contourseq, "ContourSequence"):
                return contourseq.ContourSequence
            else:
                return None
    return None


def find_parotid_info(ss: Dataset, side: str) -> tuple[str | None, int | None]:
    """Find parotid gland ROI info for the specified side.

    Args:
        ss: RTSTRUCT pydicom Dataset.
        side: 'l' for left or 'r' for right.

    Returns:
        Tuple of (ROI name, ROI number), or (None, None) if not found.
    """
    for roi in ss.StructureSetROISequence:
        name = roi.ROIName.lower()
        if "parotid" in name:
            strippedname = name.split("parotid")
            if any("stem" in elem for elem in strippedname):
                continue
            if any(side in elem for elem in strippedname):
                return roi.ROIName, roi.ROINumber
    return None, None


def find_PTV_info(ss: Dataset) -> tuple[str | None, int | None]:
    """Find PTV ROI info, prioritizing PTV70.

    Args:
        ss: RTSTRUCT pydicom Dataset.

    Returns:
        Tuple of (ROI name, ROI number), or (None, None) if not found.
    """
    store: tuple[str, int | None] = ("", None)
    all_roi: list[tuple[str, int]] = []
    for roi in ss.StructureSetROISequence:
        name = roi.ROIName.lower()
        all_roi.append((roi.ROIName, roi.ROINumber))
        if "ptv" in name:
            if "70" in name:
                return roi.ROIName, roi.ROINumber
            else:
                store = (roi.ROIName, roi.ROINumber)
    if "56" in store[0]:
        return store
    if "66" in store[0]:
        return store
    print(all_roi)
    choice = input("Enter the number corresponding to desired ROI from list:\n")
    for roi in all_roi:
        if choice == str(roi[1]):
            return roi
    return (None, None)


def same_shape(dicom_list: list[Dataset]) -> bool:
    """Check if all DICOM files have the same pixel array shape.

    Args:
        dicom_list: List of pydicom Dataset objects.

    Returns:
        True if all shapes match, False otherwise.
    """
    shapes = []
    for file in dicom_list:
        shapes.append(file.pixel_array.shape)
    return len(set(shapes)) <= 1


def backfill_labels(ds, patientID, labelsfolder, condition_descriptor):
    """Bespoke function to backfill labels for multiple label sets.

    Takes h5py File object as dataset and assumes that each
    label file follows the naming convention: {timing}_xero_label.csv
    """
    groupname = "labels"
    i = 1
    while groupname in ds.keys():
        i += 1
        groupname = f"labels_{i}"
    lblgrp = ds.create_group(groupname)
    lblgrp.attrs["desc"] = condition_descriptor
    for file in os.listdir(labelsfolder):
        if file.endswith("xero_label.csv"):
            timing = file.split("_")[0]
            lbl_df = pd.read_csv(os.path.join(labelsfolder, file), index_col=0)
            lbl_df.index = lbl_df.index.astype(str)
            if str(patientID) in lbl_df.index:
                labelvalue = lbl_df.loc[str(patientID), "label"]
            else:
                labelvalue = 99
            lblgrp.attrs[timing] = labelvalue


def unpack_mask(f, key):
    """Unpack a sparse mask from an HDF5 file."""
    dense = np.zeros_like(f["ct"][...])
    slices = f[key]["slices"][...]
    slice_nums = np.unique(slices).astype(int)
    for sl in slice_nums:
        rows = f[key]["rows"][np.where(slices == sl)]
        cols = f[key]["cols"][np.where(slices == sl)]
        sparse = scipy.sparse.coo_matrix(
            (np.ones_like(cols), (rows, cols)),
            shape=f["ct"][...].shape[1:],
            dtype=int,
        )
        dense[sl, ...] = sparse.todense()
    return dense


def pack_mask(densemask):
    """Pack a dense mask into sparse format."""
    row = np.array([])
    col = np.array([])
    slic = np.array([])
    for i in range(densemask.shape[0]):
        sp = scipy.sparse.coo_matrix(densemask[i, ...])
        sl = np.full(sp.data.shape, fill_value=i, dtype=np.int32)
        slic = np.concatenate([slic, sl])
        col = np.concatenate([col, sp.col])
        row = np.concatenate([row, sp.row])
    return slic, row, col


# Deprecated functions - re-exported from dicom_utils for backwards compatibility


def getscaledimg(file: Dataset) -> np.ndarray:
    """Extract and scale CT pixel array.

    .. deprecated::
        Use m3cv_prep.dicom_utils.getscaledimg instead.
    """
    warnings.warn(
        "getscaledimg is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.getscaledimg instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import getscaledimg as _getscaledimg

    return _getscaledimg(file)


def attr_shared(dcms: list[Dataset], attr: str) -> bool:
    """Check if all DICOM files share the same value for an attribute.

    .. deprecated::
        Use m3cv_prep.dicom_utils.attr_shared instead.
    """
    warnings.warn(
        "attr_shared is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.attr_shared instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import attr_shared as _attr_shared

    return _attr_shared(dcms, attr)


def merge_doses(*args: Dataset) -> np.ndarray:
    """Merge multiple BEAM dose files.

    .. deprecated::
        Use m3cv_prep.dicom_utils.merge_doses instead.
    """
    warnings.warn(
        "merge_doses is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.merge_doses instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import merge_doses as _merge_doses

    return _merge_doses(*args)


def window_level(
    array: np.ndarray,
    window: float,
    level: float,
    normalize: bool = False,
) -> np.ndarray:
    """Apply window/level transformation to CT array.

    .. deprecated::
        Use m3cv_prep.dicom_utils.window_level instead.
    """
    warnings.warn(
        "window_level is deprecated from _preprocess_util. "
        "Use m3cv_prep.dicom_utils.window_level instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from m3cv_prep.dicom_utils import window_level as _window_level

    return _window_level(array, window, level, normalize)
