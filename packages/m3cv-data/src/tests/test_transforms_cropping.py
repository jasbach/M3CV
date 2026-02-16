"""Tests for anatomical cropping transforms."""

from __future__ import annotations

import warnings

import pytest

from m3cv_data import load_patient
from m3cv_data.transforms import (
    AnatomicalCrop,
    BilateralStructureMidpoint,
    CombinedStructuresCOM,
    FallbackStrategy,
    ReferenceNotFoundError,
    SingleStructureCOM,
    VolumeCenterStrategy,
)


class TestSingleStructureCOM:
    """Tests for SingleStructureCOM reference strategy."""

    def test_calculate_com_valid_structure(self, sample_h5_path: str) -> None:
        """Test COM calculation with present structure."""
        patient = load_patient(sample_h5_path)
        strategy = SingleStructureCOM("GTV")

        result = strategy.calculate(patient)

        assert result is not None
        z, y, x = result
        # GTV is at slice 5, center around (32, 32)
        assert z == 5
        assert 31 <= y <= 33
        assert 31 <= x <= 33

    def test_calculate_com_missing_structure(self, sample_h5_path: str) -> None:
        """Test COM returns None for missing structure."""
        patient = load_patient(sample_h5_path)
        strategy = SingleStructureCOM("NonExistent")

        result = strategy.calculate(patient)

        assert result is None

    def test_get_required_structures(self) -> None:
        """Test required structures list."""
        strategy = SingleStructureCOM("GTV")

        required = strategy.get_required_structures()

        assert required == ["GTV"]


class TestBilateralStructureMidpoint:
    """Tests for BilateralStructureMidpoint reference strategy."""

    def test_calculate_midpoint_valid_structures(
        self, sample_h5_with_bilateral_structures: str
    ) -> None:
        """Test midpoint calculation with both structures present."""
        patient = load_patient(sample_h5_with_bilateral_structures)
        strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")

        result = strategy.calculate(patient)

        assert result is not None
        z, y, x = result
        # Parotid_L at ~(20, 40, 25), Parotid_R at ~(20, 40, 55)
        # Midpoint should be ~(20, 40, 40)
        assert 19 <= z <= 21
        assert 39 <= y <= 41
        assert 39 <= x <= 41

    def test_calculate_midpoint_one_missing(self, sample_h5_path: str) -> None:
        """Test midpoint returns None if one structure is missing."""
        patient = load_patient(sample_h5_path)
        strategy = BilateralStructureMidpoint("GTV", "NonExistent")

        result = strategy.calculate(patient)

        assert result is None

    def test_calculate_midpoint_both_missing(self, sample_h5_path: str) -> None:
        """Test midpoint returns None if both structures are missing."""
        patient = load_patient(sample_h5_path)
        strategy = BilateralStructureMidpoint("Missing1", "Missing2")

        result = strategy.calculate(patient)

        assert result is None

    def test_get_required_structures(self) -> None:
        """Test required structures list."""
        strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")

        required = strategy.get_required_structures()

        assert set(required) == {"Parotid_L", "Parotid_R"}


class TestCombinedStructuresCOM:
    """Tests for CombinedStructuresCOM reference strategy."""

    def test_calculate_combined_com(self, sample_h5_path: str) -> None:
        """Test COM of combined structures."""
        patient = load_patient(sample_h5_path)
        strategy = CombinedStructuresCOM(["GTV", "PTV"])

        result = strategy.calculate(patient)

        assert result is not None
        z, y, x = result
        # Both structures at slice 5, COM should be there
        assert z == 5

    def test_calculate_partial_structures_available(self, sample_h5_path: str) -> None:
        """Test COM when only some structures are available."""
        patient = load_patient(sample_h5_path)
        strategy = CombinedStructuresCOM(["GTV", "NonExistent"])

        result = strategy.calculate(patient)

        # Should still work with just GTV
        assert result is not None

    def test_calculate_all_missing(self, sample_h5_path: str) -> None:
        """Test COM returns None when all structures are missing."""
        patient = load_patient(sample_h5_path)
        strategy = CombinedStructuresCOM(["Missing1", "Missing2"])

        result = strategy.calculate(patient)

        assert result is None

    def test_get_required_structures(self) -> None:
        """Test required structures list."""
        strategy = CombinedStructuresCOM(["GTV", "PTV", "Brain"])

        required = strategy.get_required_structures()

        assert set(required) == {"GTV", "PTV", "Brain"}


class TestFallbackStrategy:
    """Tests for FallbackStrategy reference strategy."""

    def test_fallback_first_succeeds(self, sample_h5_path: str) -> None:
        """Test fallback returns first successful result."""
        patient = load_patient(sample_h5_path)
        primary = SingleStructureCOM("GTV")
        fallback = VolumeCenterStrategy()
        strategy = FallbackStrategy([primary, fallback])

        result = strategy.calculate(patient)

        # Should use GTV COM, not volume center
        assert result is not None
        z, y, x = result
        assert z == 5  # GTV is at slice 5

    def test_fallback_second_succeeds(self, sample_h5_path: str) -> None:
        """Test fallback uses second strategy when first fails."""
        patient = load_patient(sample_h5_path)
        primary = SingleStructureCOM("NonExistent")
        fallback = VolumeCenterStrategy()
        strategy = FallbackStrategy([primary, fallback])

        result = strategy.calculate(patient)

        # Should use volume center
        assert result is not None
        z, y, x = result
        # Volume is (10, 64, 64), center is (5, 32, 32)
        assert z == 5
        assert y == 32
        assert x == 32

    def test_fallback_all_fail(self, sample_h5_path: str) -> None:
        """Test fallback returns None when all strategies fail."""
        patient = load_patient(sample_h5_path)
        strategy = FallbackStrategy(
            [
                SingleStructureCOM("Missing1"),
                SingleStructureCOM("Missing2"),
            ]
        )

        result = strategy.calculate(patient)

        assert result is None

    def test_get_required_structures(self) -> None:
        """Test required structures from all strategies."""
        strategy = FallbackStrategy(
            [
                SingleStructureCOM("GTV"),
                BilateralStructureMidpoint("Parotid_L", "Parotid_R"),
                VolumeCenterStrategy(),
            ]
        )

        required = strategy.get_required_structures()

        assert set(required) == {"GTV", "Parotid_L", "Parotid_R"}


class TestVolumeCenterStrategy:
    """Tests for VolumeCenterStrategy reference strategy."""

    def test_calculate_center_with_ct(self, sample_h5_path: str) -> None:
        """Test volume center calculation with CT present."""
        patient = load_patient(sample_h5_path)
        strategy = VolumeCenterStrategy()

        result = strategy.calculate(patient)

        assert result is not None
        z, y, x = result
        # Volume is (10, 64, 64)
        assert z == 5
        assert y == 32
        assert x == 32

    def test_get_required_structures(self) -> None:
        """Test required structures (none for volume center)."""
        strategy = VolumeCenterStrategy()

        required = strategy.get_required_structures()

        assert required == []


class TestAnatomicalCrop:
    """Tests for AnatomicalCrop transform."""

    def test_basic_crop_with_valid_reference(
        self, sample_h5_with_bilateral_structures: str
    ) -> None:
        """Test basic cropping with valid reference structure."""
        patient = load_patient(sample_h5_with_bilateral_structures)
        strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")
        crop = AnatomicalCrop(crop_shape=(20, 40, 40), reference_strategy=strategy)

        cropped_patient = crop(patient)

        # Check shapes
        assert cropped_patient.ct.shape == (20, 40, 40)
        assert cropped_patient.dose.shape == (20, 40, 40)
        assert cropped_patient.structures["GTV"].shape == (20, 40, 40)
        assert cropped_patient.structures["Parotid_L"].shape == (20, 40, 40)
        assert cropped_patient.structures["Parotid_R"].shape == (20, 40, 40)

    def test_crop_missing_no_fallback_raises_error(self, sample_h5_path: str) -> None:
        """Test cropping raises error when reference missing and no fallback."""
        patient = load_patient(sample_h5_path)
        strategy = SingleStructureCOM("NonExistent")
        crop = AnatomicalCrop(
            crop_shape=(10, 32, 32), reference_strategy=strategy, allow_fallback=False
        )

        with pytest.raises(ReferenceNotFoundError) as exc_info:
            crop(patient)

        assert "NonExistent" in str(exc_info.value)
        assert "TEST001" in str(exc_info.value)

    def test_crop_missing_with_fallback_uses_center(self, sample_h5_path: str) -> None:
        """Test cropping falls back to center when reference missing."""
        patient = load_patient(sample_h5_path)
        strategy = SingleStructureCOM("NonExistent")
        crop = AnatomicalCrop(
            crop_shape=(8, 30, 30),
            reference_strategy=strategy,
            allow_fallback=True,
            warn_on_fallback=True,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cropped_patient = crop(patient)

            # Check warning was issued
            assert len(w) == 1
            assert "NonExistent" in str(w[0].message)
            assert "volume center" in str(w[0].message)

        # Should still get cropped output
        assert cropped_patient.ct.shape == (8, 30, 30)

    def test_crop_with_padding_at_edges(self, sample_h5_path: str) -> None:
        """Test cropping handles edge cases requiring padding."""
        patient = load_patient(sample_h5_path)
        # GTV is at slice 5, volume is (10, 64, 64)
        # Crop of size (12, 30, 30) centered at slice 5 will need padding
        # since it would need slices -1 to 11
        strategy = SingleStructureCOM("GTV")
        crop = AnatomicalCrop(crop_shape=(12, 30, 30), reference_strategy=strategy)

        cropped_patient = crop(patient)

        # Should have correct shape with padding
        assert cropped_patient.ct.shape == (12, 30, 30)
        # Check that padding uses correct voidval
        # First slice should be padded (since we need slice -1)
        assert cropped_patient.ct[0, 0, 0] == -1000.0
        # Last slices should also be padded (we only have 10 slices but need 12)
        assert cropped_patient.ct[-1, 0, 0] == -1000.0

    def test_crop_preserves_patient_metadata(
        self, sample_h5_with_bilateral_structures: str
    ) -> None:
        """Test cropping preserves patient_id and source_path."""
        patient = load_patient(sample_h5_with_bilateral_structures)
        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(20, 40, 40), reference_strategy=strategy)

        cropped_patient = crop(patient)

        assert cropped_patient.patient_id == patient.patient_id
        assert cropped_patient.source_path == patient.source_path

    def test_crop_handles_no_dose(self, sample_h5_minimal: str) -> None:
        """Test cropping works when dose is None."""
        patient = load_patient(sample_h5_minimal)
        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(4, 20, 20), reference_strategy=strategy)

        cropped_patient = crop(patient)

        assert cropped_patient.ct.shape == (4, 20, 20)
        assert cropped_patient.dose is None

    def test_crop_disable_fallback_warning(self, sample_h5_path: str) -> None:
        """Test disabling fallback warning."""
        patient = load_patient(sample_h5_path)
        strategy = SingleStructureCOM("NonExistent")
        crop = AnatomicalCrop(
            crop_shape=(8, 30, 30),
            reference_strategy=strategy,
            allow_fallback=True,
            warn_on_fallback=False,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cropped_patient = crop(patient)

            # No warning should be issued
            assert len(w) == 0

        assert cropped_patient.ct.shape == (8, 30, 30)


class TestTransformsIntegration:
    """Integration tests for transforms with PatientDataset."""

    def test_patient_dataset_with_anatomical_crop(
        self, sample_h5_with_bilateral_structures: str
    ) -> None:
        """Test PatientDataset with anatomical crop transform."""
        from m3cv_data import PatientDataset

        strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")
        crop = AnatomicalCrop(crop_shape=(20, 40, 40), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=[sample_h5_with_bilateral_structures],
            channels=["GTV"],
            include_ct=True,
            include_dose=True,
            patient_transform=crop,
        )

        volume, label = dataset[0]

        # Should have 3 channels: CT, dose, GTV
        assert volume.shape == (3, 20, 40, 40)

    def test_patient_dataset_crop_with_fallback(self, sample_h5_path: str) -> None:
        """Test PatientDataset with crop fallback."""
        from m3cv_data import PatientDataset

        strategy = SingleStructureCOM("NonExistent")
        crop = AnatomicalCrop(
            crop_shape=(8, 30, 30), reference_strategy=strategy, allow_fallback=True
        )

        dataset = PatientDataset(
            paths=[sample_h5_path],
            channels=["GTV"],
            include_ct=True,
            patient_transform=crop,
        )

        volume, label = dataset[0]

        # Should successfully fall back to center
        assert volume.shape == (2, 8, 30, 30)  # CT + GTV

    def test_dataloader_batching_with_crop(
        self, sample_h5_directory: str, sample_h5_with_bilateral_structures: str
    ) -> None:
        """Test DataLoader batching with anatomical crop."""
        from torch.utils.data import DataLoader

        from m3cv_data import PatientDataset

        # Use volume center for compatibility across all patients
        strategy = VolumeCenterStrategy()
        crop = AnatomicalCrop(crop_shape=(8, 32, 32), reference_strategy=strategy)

        dataset = PatientDataset(
            paths=sample_h5_directory,
            include_ct=True,
            patient_transform=crop,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        for batch, _labels in loader:
            # All samples should have same shape after cropping
            assert batch.shape[1:] == (1, 8, 32, 32)  # (C, Z, Y, X)
            break  # Just test first batch
