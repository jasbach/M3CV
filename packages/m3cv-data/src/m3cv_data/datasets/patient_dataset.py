"""PyTorch Dataset implementation for patient medical imaging data."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from glob import glob
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from m3cv_data.patient import Patient, load_patient


class PatientDataset(Dataset):
    """PyTorch Dataset for loading patient medical imaging data from HDF5 files.

    Supports both lazy loading (load from disk on each __getitem__) and
    preloading (load all data into memory at initialization).

    Args:
        paths: Either a directory path containing HDF5 files, or a sequence of
            file paths. If a directory, searches for *.h5 files.
        channels: List of structure names to include as channels. If None,
            only CT (and optionally dose) are included.
        include_ct: Whether to include CT as the first channel.
        include_dose: Whether to include dose as a channel.
        preload: If True, load all patient data into memory at initialization.
            Useful for small datasets that fit in memory.
        transform: Optional callable that takes a tensor and returns a
            transformed tensor. Applied to the volume before returning.
        label_fn: Optional callable that takes a Patient and returns an integer
            label. If None, returns -1 for all samples.

    Example:
        >>> dataset = PatientDataset(
        ...     paths="data/processed/",
        ...     channels=["GTV", "Parotid_L"],
        ...     include_ct=True,
        ...     include_dose=True,
        ... )
        >>> volume, label = dataset[0]
        >>> volume.shape  # (C, Z, Y, X)
    """

    def __init__(
        self,
        paths: str | Sequence[str],
        channels: list[str] | None = None,
        include_ct: bool = True,
        include_dose: bool = False,
        preload: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        label_fn: Callable[[Patient], int] | None = None,
    ) -> None:
        self._file_paths = self._resolve_paths(paths)
        self._channels = channels
        self._include_ct = include_ct
        self._include_dose = include_dose
        self._preload = preload
        self._transform = transform
        self._label_fn = label_fn

        # Preload data if requested
        self._patients: list[Patient] | None = None
        if preload:
            self._patients = [load_patient(p) for p in self._file_paths]

    def _resolve_paths(self, paths: str | Sequence[str]) -> list[str]:
        """Resolve paths to a list of HDF5 file paths."""
        if isinstance(paths, str):
            if os.path.isdir(paths):
                # Directory: find all .h5 files
                pattern = os.path.join(paths, "*.h5")
                file_paths = sorted(glob(pattern))
                if not file_paths:
                    raise ValueError(f"No .h5 files found in directory: {paths}")
                return file_paths
            elif os.path.isfile(paths):
                # Single file
                return [paths]
            else:
                raise FileNotFoundError(f"Path does not exist: {paths}")
        else:
            # Sequence of paths
            file_paths = list(paths)
            for p in file_paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"File not found: {p}")
            return file_paths

    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (volume, label) where volume is a tensor of shape
            (C, Z, Y, X) and label is an integer.
        """
        # Load or retrieve patient
        if self._patients is not None:
            patient = self._patients[idx]
        else:
            patient = load_patient(self._file_paths[idx])

        # Stack channels
        volume = patient.stack_channels(
            channels=self._channels,
            include_ct=self._include_ct,
            include_dose=self._include_dose,
        )

        # Convert to tensor
        tensor = torch.from_numpy(volume.astype(np.float32))

        # Apply transform if provided
        if self._transform is not None:
            tensor = self._transform(tensor)

        # Get label
        if self._label_fn is not None:
            label = self._label_fn(patient)
        else:
            label = -1

        return tensor, label

    def get_patient(self, idx: int) -> Patient:
        """Get the full Patient object for debugging and inspection.

        Args:
            idx: Index of the patient.

        Returns:
            Patient object with all loaded data.
        """
        if self._patients is not None:
            return self._patients[idx]
        return load_patient(self._file_paths[idx])

    @property
    def file_paths(self) -> list[str]:
        """Return the list of file paths in this dataset."""
        return self._file_paths.copy()

    def random_split(
        self,
        fractions: dict[str, float],
        seed: int | None = None,
        stratify_by: Callable[[Patient], Any] | None = None,
    ) -> dict[str, PatientDataset]:
        """Split the dataset into multiple subsets.

        Args:
            fractions: Dictionary mapping split names to fractions. Fractions
                must sum to 1.0. Example: {"train": 0.8, "val": 0.1, "test": 0.1}
            seed: Random seed for reproducibility.
            stratify_by: Optional callable that takes a Patient and returns a
                stratification key. If provided, splits will be stratified to
                maintain the same distribution of keys in each split.

        Returns:
            Dictionary mapping split names to PatientDataset instances.

        Raises:
            ValueError: If fractions don't sum to 1.0.
        """
        total = sum(fractions.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Fractions must sum to 1.0, got {total}")

        rng = np.random.default_rng(seed)
        n = len(self)

        if stratify_by is not None:
            # Stratified split
            return self._stratified_split(fractions, rng, stratify_by)

        # Random split
        indices = rng.permutation(n)
        split_indices: dict[str, list[int]] = {}
        start = 0

        for name, frac in fractions.items():
            end = start + int(frac * n)
            # Handle rounding for last split
            if name == list(fractions.keys())[-1]:
                end = n
            split_indices[name] = list(indices[start:end])
            start = end

        # Create new datasets for each split
        result = {}
        for name, idxs in split_indices.items():
            split_paths = [self._file_paths[i] for i in idxs]
            result[name] = PatientDataset(
                paths=split_paths,
                channels=self._channels,
                include_ct=self._include_ct,
                include_dose=self._include_dose,
                preload=self._preload,
                transform=self._transform,
                label_fn=self._label_fn,
            )

        return result

    def _stratified_split(
        self,
        fractions: dict[str, float],
        rng: np.random.Generator,
        stratify_by: Callable[[Patient], Any],
    ) -> dict[str, PatientDataset]:
        """Perform stratified split based on a stratification key."""
        # Group indices by stratification key
        groups: dict[Any, list[int]] = {}
        for i in range(len(self)):
            patient = self.get_patient(i)
            key = stratify_by(patient)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # Split each group
        split_indices: dict[str, list[int]] = {name: [] for name in fractions}

        for group_indices in groups.values():
            shuffled = rng.permutation(group_indices)
            start = 0
            n_group = len(shuffled)

            for name, frac in fractions.items():
                end = start + int(frac * n_group)
                if name == list(fractions.keys())[-1]:
                    end = n_group
                split_indices[name].extend(shuffled[start:end])
                start = end

        # Create datasets
        result = {}
        for name, idxs in split_indices.items():
            split_paths = [self._file_paths[i] for i in idxs]
            result[name] = PatientDataset(
                paths=split_paths,
                channels=self._channels,
                include_ct=self._include_ct,
                include_dose=self._include_dose,
                preload=self._preload,
                transform=self._transform,
                label_fn=self._label_fn,
            )

        return result
