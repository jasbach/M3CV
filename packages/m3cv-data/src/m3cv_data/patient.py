"""Patient dataclass and loading functions for HDF5 medical imaging data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from numpy.typing import NDArray

from m3cv_data._compat import HAS_DATAPREP

if TYPE_CHECKING:
    from m3cv_prep.arrays import SpatialMetadata


@dataclass
class Patient:
    """Container for patient medical imaging data loaded from HDF5.

    Attributes:
        ct: CT array in Hounsfield Units, shape (Z, Y, X).
        patient_id: Patient identifier string.
        source_path: Path to the HDF5 file this patient was loaded from.
        dose: Optional radiation dose array, shape (Z, Y, X).
        structures: Dictionary mapping ROI names to binary mask arrays.
        study_uid: DICOM Study Instance UID.
        frame_of_reference: DICOM Frame of Reference UID.
        spatial_metadata: SpatialMetadata instance if m3cv-dataprep available.
    """

    ct: NDArray[np.floating]
    patient_id: str
    source_path: str
    dose: NDArray[np.floating] | None = None
    structures: dict[str, NDArray] = field(default_factory=dict)
    study_uid: str | None = None
    frame_of_reference: str | None = None
    spatial_metadata: Any | None = None  # SpatialMetadata if available

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the CT array (Z, Y, X)."""
        return self.ct.shape  # type: ignore[return-value]

    @property
    def available_structures(self) -> list[str]:
        """Return list of available structure/ROI names."""
        return list(self.structures.keys())

    def stack_channels(
        self,
        channels: list[str] | None = None,
        include_ct: bool = True,
        include_dose: bool = False,
        merges: list[tuple[str, ...]] | None = None,
    ) -> NDArray[np.floating]:
        """Stack CT, dose, and structure masks into a multi-channel array.

        Args:
            channels: List of structure names to include. If None, no structures
                are included.
            include_ct: Whether to include CT as the first channel.
            include_dose: Whether to include dose as a channel.
            merges: Optional list of tuples specifying structures to merge into
                single channels. Each tuple contains structure names that will be
                combined using logical OR. All structures in merges must also
                appear in channels. The merged channel appears at the position
                of the first structure in the tuple.

        Returns:
            Stacked array with shape (C, Z, Y, X) where C is the number of
            channels. Channel order: CT (if included), dose (if included),
            then structures in the order specified (with merged structures
            combined into single channels).

        Raises:
            ValueError: If include_dose is True but no dose array is available.
            ValueError: If a structure in merges is not in channels.
            KeyError: If a requested structure name is not available.

        Example:
            >>> # Merge left and right parotids into one channel
            >>> volume = patient.stack_channels(
            ...     channels=["Parotid_L", "Parotid_R", "GTV"],
            ...     merges=[("Parotid_L", "Parotid_R")],
            ... )
            >>> volume.shape[0]  # 2 channels: merged parotids + GTV
            2
        """
        arrays_to_stack = []

        if include_ct:
            arrays_to_stack.append(self.ct)

        if include_dose:
            if self.dose is None:
                raise ValueError("Dose array not available for this patient")
            arrays_to_stack.append(self.dose)

        if channels:
            # Build lookup: structure name -> merge group it belongs to
            merge_groups: dict[str, tuple[str, ...]] = {}
            if merges:
                for group in merges:
                    for name in group:
                        if name not in channels:
                            raise ValueError(
                                f"Structure '{name}' in merges must also be in channels"
                            )
                        merge_groups[name] = group

            # Track which structures we've already processed via merge
            processed: set[str] = set()

            for name in channels:
                if name in processed:
                    continue

                if name not in self.structures:
                    raise KeyError(
                        f"Structure '{name}' not found. "
                        f"Available: {self.available_structures}"
                    )

                if name in merge_groups:
                    # Merge all structures in this group
                    group = merge_groups[name]
                    merged = np.zeros_like(self.structures[name], dtype=np.int8)
                    for member in group:
                        if member not in self.structures:
                            raise KeyError(
                                f"Structure '{member}' not found. "
                                f"Available: {self.available_structures}"
                            )
                        merged = np.maximum(merged, self.structures[member])
                        processed.add(member)
                    arrays_to_stack.append(merged)
                else:
                    arrays_to_stack.append(self.structures[name])

        if not arrays_to_stack:
            raise ValueError(
                "At least one channel must be included "
                "(include_ct, include_dose, or channels)"
            )

        return np.stack(arrays_to_stack, axis=0)


def _unpack_sparse_array(
    rows: NDArray,
    cols: NDArray,
    slices: NDArray,
    refshape: tuple[int, int, int],
) -> NDArray:
    """Convert sparse row, col, slice arrays back into a 3D mask array.

    The m3cv-dataprep pack_array_sparsely stores:
    - rows: Y indices (from coo_matrix.row of each 2D slice)
    - cols: X indices (from coo_matrix.col of each 2D slice)
    - slices: Z indices (the slice index being iterated)

    For an array with shape (Z, Y, X), we reconstruct using:
    dense[slices, rows, cols] = dense[Z, Y, X]
    """
    dense = np.zeros(refshape, dtype=np.int8)
    dense[slices, rows, cols] = 1
    return dense


def _load_spatial_metadata(group: h5py.Group) -> SpatialMetadata | None:
    """Load SpatialMetadata from an HDF5 group if m3cv-dataprep is available."""
    if not HAS_DATAPREP:
        return None

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


def load_patient(path: str) -> Patient:
    """Load patient data from an HDF5 file.

    Args:
        path: Path to the HDF5 file created by m3cv-dataprep.

    Returns:
        Patient object containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required 'ct' dataset is missing.
    """
    with h5py.File(path, "r") as f:
        # CT is required
        if "ct" not in f:
            raise KeyError(f"Required 'ct' dataset not found in {path}")
        ct = f["ct"][...].astype(np.float32)

        # Patient identifiers
        patient_id = str(f.attrs.get("patient_id", "unknown"))
        study_uid = str(f.attrs["study_uid"]) if "study_uid" in f.attrs else None
        frame_of_reference = (
            str(f.attrs["frame_of_reference"])
            if "frame_of_reference" in f.attrs
            else None
        )

        # Optional dose
        dose = None
        if "dose" in f:
            dose = f["dose"][...].astype(np.float32)

        # Optional structures (stored as sparse arrays)
        structures: dict[str, NDArray] = {}
        if "structures" in f:
            struct_group = f["structures"]
            for roi_name in struct_group:
                group = struct_group[roi_name]
                rows = group["rows"][...]
                cols = group["cols"][...]
                slices = group["slices"][...]
                structures[roi_name] = _unpack_sparse_array(
                    rows, cols, slices, ct.shape
                )

        # Optional spatial metadata
        spatial_metadata = None
        if "spatial_metadata" in f:
            spatial_metadata = _load_spatial_metadata(f["spatial_metadata"])

    return Patient(
        ct=ct,
        patient_id=patient_id,
        source_path=path,
        dose=dose,
        structures=structures,
        study_uid=study_uid,
        frame_of_reference=frame_of_reference,
        spatial_metadata=spatial_metadata,
    )


def load_patients(
    paths: list[str],
    show_progress: bool = True,
) -> list[Patient]:
    """Load multiple patients from HDF5 files.

    Args:
        paths: List of paths to HDF5 files.
        show_progress: Whether to show a progress indicator. Requires tqdm
            if True; falls back to no progress if tqdm unavailable.

    Returns:
        List of Patient objects.
    """
    patients = []

    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(paths, desc="Loading patients")
        except ImportError:
            iterator = paths
    else:
        iterator = paths

    for path in iterator:
        patients.append(load_patient(path))

    return patients
