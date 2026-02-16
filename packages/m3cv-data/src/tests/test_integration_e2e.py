"""
End-to-end integration tests for the complete M3CV pipeline.

These tests validate the full workflow from HDF5 data loading through
model inference, using real medical imaging data. Tests are automatically
skipped if test data is not available (not committed to repo).

Test data setup:
    uv run python scripts/setup_e2e_test_data.py

Running tests:
    # With data - runs all E2E tests
    uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v

    # Without data - tests are skipped
    uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -v
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# Check if test data is available
E2E_DATA_DIR = Path(__file__).parent / "fixtures/e2e_test_patients"
HAS_E2E_DATA = E2E_DATA_DIR.exists() and any(E2E_DATA_DIR.glob("*.h5"))

# Skip marker for tests requiring data
requires_e2e_data = pytest.mark.skipif(
    not HAS_E2E_DATA,
    reason="E2E test data not available. Run scripts/setup_e2e_test_data.py to enable.",
)

# Mark all tests in this file as e2e tests
pytestmark = [pytest.mark.e2e, requires_e2e_data]


@pytest.fixture(scope="module")
def e2e_data_paths():
    """Get paths to E2E test data files."""
    paths = sorted(E2E_DATA_DIR.glob("*.h5"))
    assert len(paths) >= 2, "Need at least 2 test patients for E2E tests"
    return [str(p) for p in paths]


class TestDataLoadingPipeline:
    """Test the complete data loading pipeline."""

    def test_load_patients_from_hdf5(self, e2e_data_paths):
        """Test loading patients from packed HDF5 files."""
        from m3cv_data import load_patient

        patient = load_patient(e2e_data_paths[0])

        # Verify patient loaded successfully
        assert patient.patient_id is not None
        assert patient.ct is not None
        assert patient.ct.shape[0] > 0  # Has slices
        assert patient.ct.shape[1] > 0  # Has rows
        assert patient.ct.shape[2] > 0  # Has columns

        # Verify structures loaded
        assert len(patient.structures) > 0
        assert "Parotid L" in patient.structures or "Parotid_L" in patient.structures
        assert "Parotid R" in patient.structures or "Parotid_R" in patient.structures

    def test_inspect_hdf5_files(self, e2e_data_paths):
        """Test HDF5 inspection utilities."""
        from m3cv_data import inspect_h5

        info = inspect_h5(e2e_data_paths[0])

        assert info.patient_id is not None
        assert info.ct_shape is not None
        assert len(info.ct_shape) == 3
        assert len(info.structure_names) > 0


class TestAnatomicalCroppingPipeline:
    """Test anatomical cropping with real patient data."""

    def test_bilateral_parotid_cropping(self, e2e_data_paths):
        """Test cropping around bilateral parotid midpoint."""
        from m3cv_data import load_patient
        from m3cv_data.transforms import AnatomicalCrop, BilateralStructureMidpoint

        patient = load_patient(e2e_data_paths[0])

        # Create parotid-centered crop
        # Note: Structure names might be "Parotid L" or "Parotid_L"
        structure_names = list(patient.structures.keys())
        parotid_l = [
            s for s in structure_names if "parotid" in s.lower() and "l" in s.lower()
        ][0]
        parotid_r = [
            s for s in structure_names if "parotid" in s.lower() and "r" in s.lower()
        ][0]

        strategy = BilateralStructureMidpoint(parotid_l, parotid_r)
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        cropped_patient = crop(patient)

        # Verify crop succeeded
        assert cropped_patient.ct.shape == (90, 256, 256)
        assert cropped_patient.patient_id == patient.patient_id
        assert parotid_l in cropped_patient.structures
        assert parotid_r in cropped_patient.structures

        # Verify reference point calculation
        reference = strategy.calculate(patient)
        assert reference is not None
        assert len(reference) == 3

    def test_volume_center_cropping(self, e2e_data_paths):
        """Test fallback to volume center cropping."""
        from m3cv_data import load_patient
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        patient = load_patient(e2e_data_paths[0])

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(80, 200, 200), reference_strategy=strategy)

        cropped_patient = crop(patient)

        assert cropped_patient.ct.shape == (80, 200, 200)

    def test_cropping_with_edge_padding(self, e2e_data_paths):
        """Test cropping handles edge cases requiring padding."""
        from m3cv_data import load_patient
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        patient = load_patient(e2e_data_paths[0])

        # Request a very large crop that will require padding
        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(
            crop_shape=(patient.ct.shape[0] + 20, 300, 300),
            reference_strategy=strategy,
        )

        cropped_patient = crop(patient)

        # Should have padding with correct void value
        assert cropped_patient.ct.shape == (patient.ct.shape[0] + 20, 300, 300)
        # Check that padding used CT void value (-1000)
        assert (cropped_patient.ct == -1000.0).any()


class TestDatasetIntegration:
    """Test PatientDataset with real data and transforms."""

    def test_dataset_with_anatomical_cropping(self, e2e_data_paths):
        """Test PatientDataset with anatomical crop transform."""
        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        # Use volume center to work with any patient
        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths[:2],  # Use first 2 patients
            channels=None,  # CT only
            include_ct=True,
            patient_transform=crop,
        )

        assert len(dataset) == 2

        # Test __getitem__
        volume, label = dataset[0]
        assert isinstance(volume, torch.Tensor)
        assert volume.shape == (1, 90, 256, 256)  # (C, Z, Y, X)
        assert volume.dtype == torch.float32

    def test_dataset_with_bilateral_structures(self, e2e_data_paths):
        """Test dataset with bilateral parotid structures and GTV."""
        from m3cv_data import PatientDataset, load_patient
        from m3cv_data.transforms import AnatomicalCrop, BilateralStructureMidpoint

        # Get structure names from first patient
        patient = load_patient(e2e_data_paths[0])
        structure_names = list(patient.structures.keys())
        parotid_l = [
            s for s in structure_names if "parotid" in s.lower() and "l" in s.lower()
        ][0]
        parotid_r = [
            s for s in structure_names if "parotid" in s.lower() and "r" in s.lower()
        ][0]
        gtv = [s for s in structure_names if "gtv" in s.lower()][0]

        strategy = BilateralStructureMidpoint(parotid_l, parotid_r)
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths,
            channels=[parotid_l, parotid_r, gtv],
            include_ct=True,
            patient_transform=crop,
        )

        volume, label = dataset[0]
        # Should have 4 channels: CT + Parotid L + Parotid R + GTV
        assert volume.shape == (4, 90, 256, 256)

    def test_dataloader_batching(self, e2e_data_paths):
        """Test DataLoader batching with real data."""
        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths,
            include_ct=True,
            patient_transform=crop,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Test batching
        batch, labels = next(iter(loader))
        assert batch.shape[0] <= 2  # Batch size
        assert batch.shape[1:] == (1, 90, 256, 256)  # (C, Z, Y, X)
        assert labels.shape[0] == batch.shape[0]

    def test_dataset_random_split(self, e2e_data_paths):
        """Test dataset splitting with real data."""
        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths,
            include_ct=True,
            patient_transform=crop,
        )

        # Split into train/val
        splits = dataset.random_split(
            fractions={"train": 0.75, "val": 0.25},
            seed=42,
        )

        assert "train" in splits
        assert "val" in splits
        assert len(splits["train"]) + len(splits["val"]) == len(dataset)


class TestModelIntegration:
    """Test integration with m3cv-models."""

    def test_resnet3d_forward_pass(self, e2e_data_paths):
        """Test ResNet3D forward pass with real medical imaging data."""
        from m3cv_models import ResNet3DBuilder

        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        # Create dataset
        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths[:2],
            include_ct=True,
            patient_transform=crop,
        )

        # Create model
        model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
        model.eval()

        # Test forward pass
        volume, _ = dataset[0]
        volume = volume.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(volume)

        assert output.shape == (1, 2)  # (batch, num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_resnet3d_with_batch(self, e2e_data_paths):
        """Test ResNet3D with batched data."""
        from m3cv_models import ResNet3DBuilder

        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths,
            include_ct=True,
            patient_transform=crop,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
        model.eval()

        batch, _ = next(iter(loader))

        with torch.no_grad():
            output = model(batch)

        assert output.shape == (batch.shape[0], 2)

    def test_gradient_flow_through_pipeline(self, e2e_data_paths):
        """Test that gradients flow through the complete pipeline."""
        from m3cv_models import ResNet3DBuilder

        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths[:1],
            include_ct=True,
            patient_transform=crop,
        )

        model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
        model.train()

        volume, _ = dataset[0]
        volume = volume.unsqueeze(0).requires_grad_(True)

        # Forward pass
        output = model(volume)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert volume.grad is not None
        assert not torch.isnan(volume.grad).any()

        # Verify model parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestMemoryAndPerformance:
    """Sanity checks for memory usage and performance."""

    def test_realistic_batch_size(self, e2e_data_paths):
        """Test that realistic batch sizes work without OOM."""
        from m3cv_models import ResNet3DBuilder

        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths,
            include_ct=True,
            patient_transform=crop,
        )

        # Test with batch size of 4 (typical for medical imaging)
        loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)

        model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
        model.eval()

        batch, _ = next(iter(loader))

        with torch.no_grad():
            output = model(batch)

        # Should complete without errors
        assert output.shape[0] == batch.shape[0]

    def test_shape_consistency_across_patients(self, e2e_data_paths):
        """Test that cropping produces consistent shapes across all patients."""
        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, VolumeCenterStrategy

        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=e2e_data_paths,
            include_ct=True,
            patient_transform=crop,
        )

        # Get all shapes
        shapes = [dataset[i][0].shape for i in range(len(dataset))]

        # All should be identical
        assert len(set(shapes)) == 1
        assert shapes[0] == (1, 90, 256, 256)


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_missing_structure_with_strict_mode(self, e2e_data_paths):
        """Test that missing structures raise clear errors."""
        from m3cv_data import PatientDataset
        from m3cv_data.transforms import (
            AnatomicalCrop,
            ReferenceNotFoundError,
            SingleStructureCOM,
        )

        strategy = SingleStructureCOM("NonExistentStructure")
        crop = AnatomicalCrop(
            crop_shape=(90, 256, 256),
            reference_strategy=strategy,
            allow_fallback=False,
        )

        dataset = PatientDataset(
            paths=e2e_data_paths[:1],
            include_ct=True,
            patient_transform=crop,
        )

        with pytest.raises(ReferenceNotFoundError) as exc_info:
            _ = dataset[0]

        assert "NonExistentStructure" in str(exc_info.value)

    def test_missing_structure_with_fallback(self, e2e_data_paths):
        """Test that fallback mode handles missing structures gracefully."""
        import warnings

        from m3cv_data import PatientDataset
        from m3cv_data.transforms import AnatomicalCrop, SingleStructureCOM

        strategy = SingleStructureCOM("NonExistentStructure")
        crop = AnatomicalCrop(
            crop_shape=(90, 256, 256),
            reference_strategy=strategy,
            allow_fallback=True,
        )

        dataset = PatientDataset(
            paths=e2e_data_paths[:1],
            include_ct=True,
            patient_transform=crop,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            volume, _ = dataset[0]

            # Should issue a warning
            assert len(w) > 0
            assert "NonExistentStructure" in str(w[0].message)

        # Should still produce valid output
        assert volume.shape == (1, 90, 256, 256)
