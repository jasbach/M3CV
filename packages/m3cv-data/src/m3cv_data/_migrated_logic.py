"""Migrated logic from m3cv-dataprep.

WARNING: This module contains code migrated from legacy modules for
preservation purposes. It is NOT intended for use in its current form.
This code will be refactored into proper modules when m3cv-data is built out.

Migrated from:
- packages/m3cv-dataprep/src/m3cv_prep/handler.py (2025-01-29)
- packages/m3cv-dataprep/src/m3cv_prep/pending_refactor/data_augmentation.py
"""

import copy

import numpy as np
import scipy.ndimage.interpolation as scipy_mods

# =============================================================================
# CORE AUGMENTATION TRANSFORMS
# =============================================================================
# These are standalone functions that perform spatial transformations on numpy
# arrays. They form the foundation for data augmentation.
#
# Note: These functions work on raw numpy arrays, not PatientArray objects.
# The original implementation used a channel-stacked approach (Z, Y, X, C).
# When refactoring, consider whether to keep this or use separate arrays.
#
# TODO: Refactor into m3cv_data.transforms module with:
#   - Modern np.random.Generator API instead of np.random.seed()
#   - Support for both stacked and separate array approaches
#   - Configurable interpolation orders for different modalities
# =============================================================================


def rotate_array(array, voidval, degree_range=15, seed=None, degrees=None):
    """Rotate a 3D array about the Z axis.

    Args:
        array: 3D numpy array with shape (Z, Y, X).
        voidval: Value to use for void/background regions.
        degree_range: Max degrees +/- of rotation. Ignored if degrees specified.
        seed: Random seed for reproducibility. If None, uses random seed.
        degrees: Explicit rotation angle (overrides random generation).

    Returns:
        Rotated array with same shape as input.
    """
    if seed is None:
        seed = np.random.randint(1000)

    result = copy.deepcopy(array)
    np.random.seed(seed)
    intensity = np.random.random()

    if degrees is None:
        degrees = intensity * degree_range * 2 - degree_range

    result = scipy_mods.rotate(
        result,
        angle=degrees,
        axes=(1, 2),
        reshape=False,
        mode="constant",
        cval=voidval,
    )
    return result


def shift_array(array, voidval, max_shift=0.2, seed=None, pixelshift=None):
    """Shift a 3D array along Y and X dimensions.

    Args:
        array: 3D numpy array with shape (Z, Y, X).
        voidval: Value to use for void/background regions.
        max_shift: Max shift as fraction of array size (0.0-1.0). Default 0.2.
        seed: Random seed for reproducibility. If None, uses random seed.
        pixelshift: Explicit (Y, X) pixel shift tuple (overrides random).

    Returns:
        Shifted array with same shape as input.
    """
    max_y_pix = max_shift * array.shape[1]
    max_x_pix = max_shift * array.shape[2]

    if seed is None:
        seed = np.random.randint(1000)

    result = copy.deepcopy(array)
    np.random.seed(seed)
    y_intensity = np.random.random()
    x_intensity = np.random.random()

    if pixelshift is None:
        yshift = round(y_intensity * max_y_pix * 2 - max_y_pix)
        xshift = round(x_intensity * max_x_pix * 2 - max_x_pix)
        shiftspec = (0, yshift, xshift)
    else:
        shiftspec = (0, pixelshift[0], pixelshift[1])

    result = scipy_mods.shift(
        result,
        shift=shiftspec,
        mode="constant",
        cval=voidval,
    )
    return result


def zoom_array(array, voidval, max_zoom_factor=0.2, seed=None, zoom_factor=None):
    """Zoom a 3D array uniformly in all dimensions.

    Args:
        array: 3D numpy array with shape (Z, Y, X).
        voidval: Value to use for void/background regions.
        max_zoom_factor: Max zoom deviation from 1.0 (e.g., 0.2 = 0.8-1.2x).
        seed: Random seed for reproducibility. If None, uses random seed.
        zoom_factor: Explicit zoom factor (overrides random generation).

    Returns:
        Zoomed array with same shape as input (cropped or padded as needed).
    """
    if seed is None:
        seed = np.random.randint(1000)

    result = copy.deepcopy(array)
    np.random.seed(seed)
    intensity = np.random.random()

    if zoom_factor is None:
        zoom_factor = 1 + intensity * max_zoom_factor * 2 - max_zoom_factor

    original_shape = result.shape

    result = scipy_mods.zoom(
        result,
        zoom=[zoom_factor, zoom_factor, zoom_factor],
        mode="constant",
        cval=voidval,
    )

    if zoom_factor > 1.0:
        # Crop to original shape
        result = bounding_box(result, original_shape)
    elif zoom_factor < 1.0:
        # Pad to original shape
        diffs = np.array(original_shape) - np.array(result.shape)
        odd_val_offset = diffs % 2
        diffs = diffs // 2
        pad_spec = [
            (diffs[0], diffs[0] + odd_val_offset[0]),
            (diffs[1], diffs[1] + odd_val_offset[1]),
            (diffs[2], diffs[2] + odd_val_offset[2]),
        ]
        result = np.pad(
            result,
            pad_width=pad_spec,
            mode="constant",
            constant_values=voidval,
        )

    return result


def bounding_box(array, shape, center=None):
    """Extract a bounding box from a 3D array.

    Args:
        array: 3D numpy array with shape (Z, Y, X).
        shape: Desired output shape as (Z, Y, X) or (Y, X) for 2D.
        center: Center indices (Z, Y, X), or None for array center.

    Returns:
        Cropped array of the specified shape.
    """
    if len(shape) == 2:
        shape = (array.shape[0], shape[0], shape[1])

    if center is None:
        center = [array.shape[0] // 2, array.shape[1] // 2, array.shape[2] // 2]
    else:
        center = [round(pos) for pos in center]

    start = [
        center[0] - (shape[0] // 2),
        center[1] - (shape[1] // 2),
        center[2] - (shape[2] // 2),
    ]

    return array[
        start[0] : start[0] + shape[0],
        start[1] : start[1] + shape[1],
        start[2] : start[2] + shape[2],
    ]


# =============================================================================
# AUGMENTATION ORCHESTRATION
# =============================================================================
# These functions coordinate augmentation across multiple modalities (CT, dose,
# mask) using a shared seed to ensure spatial consistency.
#
# NOTE: These currently reference methods on PatientArray objects that have
# been removed from m3cv-dataprep. When refactoring, update these to use the
# standalone functions above (rotate_array, shift_array, zoom_array).
#
# TODO: Refactor into m3cv_data.transforms module with:
#   - Modern np.random.Generator API
#   - Transform parameter objects (generate once, apply to many)
#   - Composable transform pipelines
# =============================================================================


def coordinated_zoom(ct, dose, masks, maximum=0.2, exact=None):
    """Apply zoom transform to all modalities with shared seed.

    Args:
        ct: PatientCT array
        dose: PatientDose array (or None)
        masks: List of PatientMask arrays
        maximum: Max zoom factor deviation from 1.0 (default 0.2 = 0.8-1.2x)
        exact: Explicit zoom factor (overrides random)
    """
    seed = np.random.randint(0, 10000)
    if ct is not None:
        ct.zoom(max_zoom_factor=maximum, seed=seed, zoom_factor=exact)
    if dose is not None:
        dose.zoom(max_zoom_factor=maximum, seed=seed, zoom_factor=exact)
    for mask in masks:
        mask.zoom(max_zoom_factor=maximum, seed=seed, zoom_factor=exact)


def coordinated_shift(ct, dose, masks, maximum=0.2, exact=None):
    """Apply shift transform to all modalities with shared seed.

    Args:
        ct: PatientCT array
        dose: PatientDose array (or None)
        masks: List of PatientMask arrays
        maximum: Max shift as fraction of array size (default 0.2)
        exact: Explicit (Y, X) pixel shift tuple (overrides random)
    """
    seed = np.random.randint(0, 10000)
    if ct is not None:
        ct.shift(max_shift=maximum, seed=seed, pixelshift=exact)
    if dose is not None:
        dose.shift(max_shift=maximum, seed=seed, pixelshift=exact)
    for mask in masks:
        mask.shift(max_shift=maximum, seed=seed, pixelshift=exact)


def coordinated_rotate(ct, dose, masks, maximum=15, exact=None):
    """Apply rotation transform to all modalities with shared seed.

    Args:
        ct: PatientCT array
        dose: PatientDose array (or None)
        masks: List of PatientMask arrays
        maximum: Max rotation in degrees (default 15)
        exact: Explicit rotation angle in degrees (overrides random)
    """
    seed = np.random.randint(0, 10000)
    if ct is not None:
        ct.rotate(degree_range=maximum, seed=seed, degrees=exact)
    if dose is not None:
        dose.rotate(degree_range=maximum, seed=seed, degrees=exact)
    for mask in masks:
        mask.rotate(degree_range=maximum, seed=seed, degrees=exact)


def random_augment(ct, dose, masks, num_augs=2, replace=False):
    """Apply random augmentations to all modalities.

    Args:
        ct: PatientCT array
        dose: PatientDose array (or None)
        masks: List of PatientMask arrays
        num_augs: Number of augmentations to apply (default 2)
        replace: Whether to allow same augmentation multiple times (default False)
    """
    augs = [coordinated_zoom, coordinated_rotate, coordinated_shift]
    aug_idx = list(range(len(augs)))
    performed = 0
    while performed < num_augs:
        select = np.random.choice(aug_idx)
        op = augs[select]
        if replace is False:
            aug_idx.remove(select)
        op(ct, dose, masks)
        performed += 1


def reset_augments(ct, dose, masks):
    """Reset all arrays to pre-augmentation state.

    Args:
        ct: PatientCT array
        dose: PatientDose array (or None)
        masks: List of PatientMask arrays
    """
    if ct is not None:
        ct.reset_augments()
    if dose is not None:
        dose.reset_augments()
    for mask in masks:
        mask.reset_augments()


# =============================================================================
# SUPPLEMENTAL DATA LOADING
# =============================================================================
# These functions load non-volumetric data (labels, surveys, clinical variables)
# from DataFrames by patient ID lookup.
#
# TODO: Refactor into m3cv_data.datasets or m3cv_data.utils module with:
#   - Better error handling
#   - Type hints
#   - Integration with Dataset classes
# =============================================================================


def get_label(patient_id, labeldf):
    """Look up label for a patient from a DataFrame.

    Args:
        patient_id: Patient identifier
        labeldf: DataFrame with patient IDs as index and 'label' column

    Returns:
        Label value, or 99 if patient not found
    """
    if patient_id is None:
        raise ValueError("Cannot fetch label without patient ID")
    labeldf.index = labeldf.index.astype(str)
    if str(patient_id) in labeldf.index:
        return labeldf.loc[str(patient_id), "label"]
    return 99  # Default/missing label value


def populate_surveys(patient_id, binned_survey_df):
    """Extract survey data for a patient from a DataFrame.

    Args:
        patient_id: Patient identifier
        binned_survey_df: DataFrame with 'mrn' column and survey responses

    Returns:
        Tuple of (survey_data_array, field_names) or (None, None) if not found
    """
    surveys = binned_survey_df.copy()
    surveys["mrn"] = surveys["mrn"].fillna(0)
    surveys["mrn"] = surveys["mrn"].astype(int).astype(str)

    # Drop date/timestamp columns (PII concern)
    for col in surveys.columns:
        if any(("date" in col.lower(), "timestamp" in col.lower())):
            surveys = surveys.drop(columns=[col])

    subset = surveys[surveys["mrn"] == str(patient_id)]
    if len(subset) == 0:
        return None, None

    subset = subset.drop(columns=["mrn"])
    return subset.to_numpy(), list(subset.columns)


def get_pt_chars(patient_id, pc_file):
    """Extract patient characteristics for a patient from a DataFrame.

    Args:
        patient_id: Patient identifier
        pc_file: DataFrame with patient IDs as index and clinical variables

    Returns:
        Tuple of (pt_chars_array, field_names) or (None, None) if not found
    """
    if patient_id is None:
        raise ValueError("Cannot fetch pt_chars without patient ID")

    pc_file = pc_file.copy()
    pc_file.index = pc_file.index.astype(str)

    # Drop date columns (PII concern)
    if "Date of Diagnosis" in pc_file.columns:
        pc_file = pc_file.drop(columns=["Date of Diagnosis"])

    if str(patient_id) in pc_file.index:
        return pc_file.loc[str(patient_id)].to_numpy(), list(pc_file.columns)
    return None, None


# =============================================================================
# BOUNDING BOX / CROPPING LOGIC
# =============================================================================
# This logic handles extracting cropped regions centered on structures.
# Used for training crops focused on anatomy of interest.
#
# TODO: Refactor into m3cv_data.transforms.spatial or similar with:
#   - Cleaner center calculation
#   - Support for different centering strategies
# =============================================================================


def calculate_crop_center(masks, strategy="combined", ct_shape=None):
    """Calculate center point for cropping based on mask locations.

    Args:
        masks: List of PatientMask arrays
        strategy: Centering strategy:
            - "combined": Center of mass of all masks combined
            - "parotid": Z from parotid masks, X/Y from array center
            - None: Use array center
        ct_shape: Shape of CT array (required for "parotid" strategy)

    Returns:
        Tuple of (Z, Y, X) center coordinates, or None for array center
    """
    if strategy is None:
        return None

    if strategy == "combined":
        if len(masks) == 1:
            return tuple(masks[0].com)
        elif len(masks) > 1:
            # Combine all masks and find center of mass
            combined = masks[0].array.copy()
            for i in range(1, len(masks)):
                combined = combined + masks[i].array
            combined[combined > 1] = 1
            livecoords = np.argwhere(combined)
            center = np.sum(livecoords, axis=0) / len(livecoords)
            return tuple(center)
        return None

    if strategy == "parotid":
        # Find parotid masks and center Z on them, but keep X/Y centered
        parotid_masks = [m for m in masks if "parotid" in (m.proper_name or "").lower()]
        if not parotid_masks:
            return None

        merged = sum(m.array for m in parotid_masks)
        livecoords = np.argwhere(merged)
        center = np.sum(livecoords, axis=0) / len(livecoords)

        if ct_shape is not None:
            return (
                center[0],
                round(ct_shape[1] / 2),
                round(ct_shape[2] / 2),
            )
        return tuple(center)

    return None


def extract_cropped_arrays(ct, dose, masks, boxshape, center=None):
    """Extract bounding box regions from all arrays.

    Args:
        ct: PatientCT array
        dose: PatientDose array (or None)
        masks: List of PatientMask arrays
        boxshape: Desired output shape (Z, Y, X)
        center: Center point for crop, or None for array center

    Returns:
        Tuple of (ct_cropped, dose_cropped, masks_cropped)
    """
    ct_cropped = ct.bounding_box(shape=boxshape, center=center)
    dose_cropped = dose.bounding_box(shape=boxshape, center=center) if dose else None
    masks_cropped = [m.bounding_box(shape=boxshape, center=center) for m in masks]
    return ct_cropped, dose_cropped, masks_cropped
