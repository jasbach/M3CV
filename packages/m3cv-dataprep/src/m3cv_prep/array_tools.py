"""Array tools for packing/unpacking and constructing patient arrays."""

from __future__ import annotations

import warnings
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


def _build_masks_by_name(
    ssfile: Dataset,
    ct_array: PatientCT,
    roi_for_map: dict[str, str],
    structure_names: list[str],
) -> dict[str, PatientMask]:
    """Build structure masks by exact ROI name match.

    Args:
        ssfile: RTSTRUCT DICOM dataset.
        ct_array: CT array to use as reference geometry.
        roi_for_map: Map of ROI name -> FrameOfReferenceUID from the structure set.
        structure_names: ROI names to extract.

    Returns:
        Dict of ROI name -> PatientMask for each successfully extracted structure.
    """
    from m3cv_prep.arrays.exceptions import ROINotFoundError

    masks: dict[str, PatientMask] = {}
    ct_for = ct_array.FoR
    for name in structure_names:
        if roi_for_map.get(name) != ct_for:
            warnings.warn(
                f"Skipping structure '{name}': FrameOfReferenceUID does not match CT",
                UserWarning,
                stacklevel=3,
            )
            continue
        try:
            masks[name] = PatientMask.from_rtstruct(
                reference=ct_array,
                ssfile=ssfile,
                roi_name=name,
            )
        except ROINotFoundError as e:
            warnings.warn(
                f"Skipping structure '{name}': {e}", UserWarning, stacklevel=3
            )
    return masks


def _build_masks_by_aliases(
    ssfile: Dataset,
    ct_array: PatientCT,
    roi_for_map: dict[str, str],
    structure_aliases: dict[str, list[str]],
) -> dict[str, PatientMask]:
    """Build structure masks by trying alias ROI names for each canonical name.

    Args:
        ssfile: RTSTRUCT DICOM dataset.
        ct_array: CT array to use as reference geometry.
        roi_for_map: Map of ROI name -> FrameOfReferenceUID from the structure set.
        structure_aliases: Map of canonical name -> list of alias ROI names to try.
            First matching alias wins; result is stored under the canonical name.

    Returns:
        Dict of canonical name -> PatientMask for each successfully extracted structure.
    """
    from m3cv_prep.arrays.exceptions import ROINotFoundError

    masks: dict[str, PatientMask] = {}
    ct_for = ct_array.FoR
    for canonical_name, aliases in structure_aliases.items():
        mask = None
        for alias in aliases:
            if roi_for_map.get(alias) != ct_for:
                continue
            try:
                mask = PatientMask.from_rtstruct(
                    reference=ct_array,
                    ssfile=ssfile,
                    roi_name=alias,
                    proper_name=canonical_name,
                )
                break
            except ROINotFoundError:
                continue
        if mask is None:
            warnings.warn(
                f"Skipping structure '{canonical_name}': no matching alias found in structure set",
                UserWarning,
                stacklevel=3,
            )
        else:
            masks[canonical_name] = mask
    return masks


def construct_arrays(
    grouped_dcms: dict[str, list[Dataset]],
    structure_names: list[str] | None = None,
    structure_aliases: dict[str, list[str]] | None = None,
) -> tuple[PatientCT, PatientDose | None, dict[str, PatientMask] | None]:
    """Construct patient arrays from grouped DICOM files.

    Args:
        grouped_dcms: Dictionary mapping modality to list of DICOM Datasets.
        structure_names: List of ROI names to create masks for (exact-match).
        structure_aliases: Map of canonical name -> list of alias ROI names to try.
            First matching alias wins; result is stored under the canonical name.
            Mutually exclusive with structure_names (caller's responsibility).

    Returns:
        Tuple of (ct_array, dose_array, structure_masks).
        dose_array is None if no RTDOSE files provided.
        structure_masks is None if no RTSTRUCT or structure_names/structure_aliases
        provided.

    Raises:
        ValueError: If multiple RTSTRUCT files found, or if both structure_names
            and structure_aliases are provided.
    """
    if structure_names is not None and structure_aliases is not None:
        raise ValueError(
            "structure_names and structure_aliases are mutually exclusive; provide one or the other."
        )

    ct_array = PatientCT.from_dicom_files(grouped_dcms["CT"])
    dose_array = None
    structure_masks = None

    if "RTDOSE" in grouped_dcms:
        dose_array = PatientDose.from_dicom(grouped_dcms["RTDOSE"])
        dose_array.align_with(ct_array)

    if "RTSTRUCT" in grouped_dcms and (
        structure_names is not None or structure_aliases is not None
    ):
        if len(grouped_dcms["RTSTRUCT"]) > 1:
            raise ValueError("Multiple RTSTRUCT files found; please provide only one.")

        ssfile = grouped_dcms["RTSTRUCT"][0]
        roi_for_map = {
            roi_info.ROIName: roi_info.ReferencedFrameOfReferenceUID
            for roi_info in ssfile.StructureSetROISequence
        }

        if structure_names is not None:
            structure_masks = _build_masks_by_name(
                ssfile, ct_array, roi_for_map, structure_names
            )
        else:
            structure_masks = _build_masks_by_aliases(
                ssfile, ct_array, roi_for_map, structure_aliases
            )

    return ct_array, dose_array, structure_masks
