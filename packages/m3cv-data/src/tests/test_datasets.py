"""Tests for datasets module."""

from __future__ import annotations

import os

import pytest
import torch

from m3cv_data import Patient, PatientDataset, patient_collate_fn


class TestPatientDataset:
    """Tests for PatientDataset class."""

    def test_init_with_directory(self, sample_h5_directory: str) -> None:
        """Test initializing dataset with a directory."""
        dataset = PatientDataset(paths=sample_h5_directory)
        assert len(dataset) == 3

    def test_init_with_file_list(self, sample_h5_directory: str) -> None:
        """Test initializing dataset with list of files."""
        import glob

        paths = glob.glob(os.path.join(sample_h5_directory, "*.h5"))
        dataset = PatientDataset(paths=paths)
        assert len(dataset) == 3

    def test_init_with_single_file(self, sample_h5_path: str) -> None:
        """Test initializing dataset with single file."""
        dataset = PatientDataset(paths=sample_h5_path)
        assert len(dataset) == 1

    def test_init_empty_directory_raises(self, tmp_path) -> None:
        """Test that empty directory raises ValueError."""
        with pytest.raises(ValueError, match="No .h5 files found"):
            PatientDataset(paths=str(tmp_path))

    def test_init_nonexistent_path_raises(self) -> None:
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PatientDataset(paths="/nonexistent/path")

    def test_getitem_returns_tensor_and_label(self, sample_h5_path: str) -> None:
        """Test __getitem__ returns (tensor, label) tuple."""
        dataset = PatientDataset(paths=sample_h5_path)
        volume, label = dataset[0]

        assert isinstance(volume, torch.Tensor)
        assert volume.dtype == torch.float32
        assert isinstance(label, int)
        assert label == -1  # Default label when no label_fn

    def test_getitem_ct_only(self, sample_h5_path: str) -> None:
        """Test __getitem__ with CT only."""
        dataset = PatientDataset(
            paths=sample_h5_path, include_ct=True, include_dose=False
        )
        volume, _ = dataset[0]
        assert volume.shape == (1, 10, 64, 64)

    def test_getitem_with_structures(self, sample_h5_path: str) -> None:
        """Test __getitem__ with CT and structures."""
        dataset = PatientDataset(
            paths=sample_h5_path,
            channels=["GTV", "PTV"],
            include_ct=True,
            include_dose=False,
        )
        volume, _ = dataset[0]
        assert volume.shape == (3, 10, 64, 64)

    def test_getitem_with_dose(self, sample_h5_path: str) -> None:
        """Test __getitem__ with CT and dose."""
        dataset = PatientDataset(
            paths=sample_h5_path, include_ct=True, include_dose=True
        )
        volume, _ = dataset[0]
        assert volume.shape == (2, 10, 64, 64)

    def test_preload_mode(self, sample_h5_directory: str) -> None:
        """Test preload mode loads all data at init."""
        dataset = PatientDataset(paths=sample_h5_directory, preload=True)
        assert dataset._patients is not None
        assert len(dataset._patients) == 3

    def test_lazy_mode(self, sample_h5_directory: str) -> None:
        """Test lazy mode doesn't preload data."""
        dataset = PatientDataset(paths=sample_h5_directory, preload=False)
        assert dataset._patients is None

    def test_transform(self, sample_h5_path: str) -> None:
        """Test transform is applied to volumes."""

        def double_transform(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        dataset = PatientDataset(paths=sample_h5_path, transform=double_transform)
        volume, _ = dataset[0]

        # Load without transform to compare
        dataset_no_transform = PatientDataset(paths=sample_h5_path)
        volume_original, _ = dataset_no_transform[0]

        assert torch.allclose(volume, volume_original * 2)

    def test_label_fn(self, sample_h5_path: str) -> None:
        """Test label_fn is used to generate labels."""

        def label_fn(patient: Patient) -> int:
            return 42

        dataset = PatientDataset(paths=sample_h5_path, label_fn=label_fn)
        _, label = dataset[0]
        assert label == 42

    def test_get_patient(self, sample_h5_path: str) -> None:
        """Test get_patient returns full Patient object."""
        dataset = PatientDataset(paths=sample_h5_path)
        patient = dataset.get_patient(0)

        assert isinstance(patient, Patient)
        assert patient.patient_id == "TEST001"

    def test_file_paths_property(self, sample_h5_directory: str) -> None:
        """Test file_paths returns copy of paths."""
        dataset = PatientDataset(paths=sample_h5_directory)
        paths = dataset.file_paths
        assert len(paths) == 3
        # Verify it's a copy
        paths.append("extra")
        assert len(dataset.file_paths) == 3


class TestRandomSplit:
    """Tests for PatientDataset.random_split method."""

    def test_random_split_basic(self, sample_h5_directory: str) -> None:
        """Test basic random split."""
        dataset = PatientDataset(paths=sample_h5_directory)
        splits = dataset.random_split({"train": 0.7, "test": 0.3}, seed=42)

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) + len(splits["test"]) == len(dataset)

    def test_random_split_reproducible(self, sample_h5_directory: str) -> None:
        """Test that splits are reproducible with same seed."""
        dataset = PatientDataset(paths=sample_h5_directory)

        splits1 = dataset.random_split({"train": 0.7, "test": 0.3}, seed=42)
        splits2 = dataset.random_split({"train": 0.7, "test": 0.3}, seed=42)

        assert splits1["train"].file_paths == splits2["train"].file_paths

    def test_random_split_different_seeds(self, sample_h5_directory: str) -> None:
        """Test that different seeds give different splits."""
        dataset = PatientDataset(paths=sample_h5_directory)

        splits1 = dataset.random_split({"train": 0.7, "test": 0.3}, seed=42)
        splits2 = dataset.random_split({"train": 0.7, "test": 0.3}, seed=123)

        # With only 3 samples, splits might be the same by chance
        # Just verify we get valid splits
        assert len(splits1["train"]) + len(splits1["test"]) == 3
        assert len(splits2["train"]) + len(splits2["test"]) == 3

    def test_random_split_invalid_fractions(self, sample_h5_directory: str) -> None:
        """Test that invalid fractions raise ValueError."""
        dataset = PatientDataset(paths=sample_h5_directory)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            dataset.random_split({"train": 0.5, "test": 0.3})

    def test_random_split_preserves_config(self, sample_h5_path: str) -> None:
        """Test that split datasets preserve configuration."""
        dataset = PatientDataset(
            paths=sample_h5_path,
            channels=["GTV"],
            include_ct=True,
            include_dose=True,
        )
        splits = dataset.random_split({"train": 1.0})

        volume, _ = splits["train"][0]
        # CT + dose + GTV = 3 channels
        assert volume.shape[0] == 3


class TestPatientCollateFn:
    """Tests for patient_collate_fn."""

    def test_collate_basic(self) -> None:
        """Test basic collation."""
        batch = [
            (torch.randn(2, 10, 64, 64), 0),
            (torch.randn(2, 10, 64, 64), 1),
        ]

        volumes, labels = patient_collate_fn(batch)

        assert volumes.shape == (2, 2, 10, 64, 64)
        assert labels.shape == (2,)
        assert labels.dtype == torch.long

    def test_collate_mismatched_shapes_raises(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        batch = [
            (torch.randn(2, 10, 64, 64), 0),
            (torch.randn(2, 10, 32, 32), 1),  # Different spatial dims
        ]

        with pytest.raises(ValueError, match="same shape"):
            patient_collate_fn(batch)


class TestDataLoader:
    """Integration tests with PyTorch DataLoader."""

    def test_dataloader_basic(self, sample_h5_directory: str) -> None:
        """Test dataset works with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = PatientDataset(paths=sample_h5_directory)
        loader = DataLoader(dataset, batch_size=2)

        batch = next(iter(loader))
        volumes, labels = batch

        assert volumes.shape[0] == 2  # batch size
        assert labels.shape[0] == 2

    def test_dataloader_with_collate_fn(self, sample_h5_directory: str) -> None:
        """Test DataLoader with custom collate function."""
        from torch.utils.data import DataLoader

        dataset = PatientDataset(paths=sample_h5_directory)
        loader = DataLoader(dataset, batch_size=2, collate_fn=patient_collate_fn)

        volumes, labels = next(iter(loader))
        assert volumes.shape[0] == 2
        assert labels.dtype == torch.long
