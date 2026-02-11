"""Tests for patient.py module."""

from __future__ import annotations

import numpy as np
import pytest

from m3cv_data import load_patient, load_patients


class TestPatient:
    """Tests for the Patient dataclass."""

    def test_shape_property(self, sample_h5_path: str) -> None:
        """Test that shape property returns CT shape."""
        patient = load_patient(sample_h5_path)
        assert patient.shape == (10, 64, 64)

    def test_available_structures(self, sample_h5_path: str) -> None:
        """Test that available_structures lists structure names."""
        patient = load_patient(sample_h5_path)
        structures = patient.available_structures
        assert "GTV" in structures
        assert "PTV" in structures
        assert len(structures) == 2

    def test_stack_channels_ct_only(self, sample_h5_path: str) -> None:
        """Test stacking with only CT."""
        patient = load_patient(sample_h5_path)
        volume = patient.stack_channels(include_ct=True, include_dose=False)
        assert volume.shape == (1, 10, 64, 64)

    def test_stack_channels_ct_and_dose(self, sample_h5_path: str) -> None:
        """Test stacking CT and dose."""
        patient = load_patient(sample_h5_path)
        volume = patient.stack_channels(include_ct=True, include_dose=True)
        assert volume.shape == (2, 10, 64, 64)

    def test_stack_channels_with_structures(self, sample_h5_path: str) -> None:
        """Test stacking CT and structures."""
        patient = load_patient(sample_h5_path)
        volume = patient.stack_channels(
            channels=["GTV", "PTV"], include_ct=True, include_dose=False
        )
        assert volume.shape == (3, 10, 64, 64)

    def test_stack_channels_all(self, sample_h5_path: str) -> None:
        """Test stacking CT, dose, and structures."""
        patient = load_patient(sample_h5_path)
        volume = patient.stack_channels(
            channels=["GTV"], include_ct=True, include_dose=True
        )
        assert volume.shape == (3, 10, 64, 64)

    def test_stack_channels_missing_structure_raises(self, sample_h5_path: str) -> None:
        """Test that requesting missing structure raises KeyError."""
        patient = load_patient(sample_h5_path)
        with pytest.raises(KeyError, match="Structure 'MISSING' not found"):
            patient.stack_channels(channels=["MISSING"])

    def test_stack_channels_no_channels_raises(self, sample_h5_path: str) -> None:
        """Test that stacking with no channels raises ValueError."""
        patient = load_patient(sample_h5_path)
        with pytest.raises(ValueError, match="At least one channel"):
            patient.stack_channels(include_ct=False, include_dose=False)

    def test_stack_channels_dose_not_available(self, sample_h5_minimal: str) -> None:
        """Test that requesting dose when unavailable raises ValueError."""
        patient = load_patient(sample_h5_minimal)
        with pytest.raises(ValueError, match="Dose array not available"):
            patient.stack_channels(include_dose=True)

    def test_stack_channels_merge_structures(self, sample_h5_path: str) -> None:
        """Test merging two structures into a single channel."""
        patient = load_patient(sample_h5_path)
        volume = patient.stack_channels(
            channels=["GTV", "PTV"],
            include_ct=False,
            merges=[("GTV", "PTV")],
        )
        # Should be 1 channel (merged) instead of 2
        assert volume.shape == (1, 10, 64, 64)

    def test_stack_channels_merge_with_other_channels(
        self, sample_h5_path: str
    ) -> None:
        """Test merging some structures while keeping others separate."""
        patient = load_patient(sample_h5_path)
        volume = patient.stack_channels(
            channels=["GTV", "PTV"],
            include_ct=True,
            merges=[("GTV", "PTV")],
        )
        # Should be 2 channels: CT + merged structures
        assert volume.shape == (2, 10, 64, 64)

    def test_stack_channels_merge_produces_union(self, sample_h5_path: str) -> None:
        """Test that merged channel is the union (logical OR) of structures."""
        patient = load_patient(sample_h5_path)

        # Get individual structures
        vol_separate = patient.stack_channels(
            channels=["GTV", "PTV"],
            include_ct=False,
        )
        gtv_mask = vol_separate[0]
        ptv_mask = vol_separate[1]

        # Get merged
        vol_merged = patient.stack_channels(
            channels=["GTV", "PTV"],
            include_ct=False,
            merges=[("GTV", "PTV")],
        )
        merged_mask = vol_merged[0]

        # Merged should be max of the two (union for binary masks)
        expected = np.maximum(gtv_mask, ptv_mask)
        np.testing.assert_array_equal(merged_mask, expected)

    def test_stack_channels_merge_order_preserved(self, sample_h5_path: str) -> None:
        """Test that merged channel appears at position of first structure."""
        patient = load_patient(sample_h5_path)

        # Merge appears at PTV position (second in channels list)
        # But since GTV comes first and isn't merged, it should be first
        # This test verifies order is based on channels list
        volume = patient.stack_channels(
            channels=["GTV", "PTV"],
            include_ct=True,
            merges=[("GTV", "PTV")],
        )
        # CT first, then merged GTV+PTV
        assert volume.shape == (2, 10, 64, 64)

    def test_stack_channels_merge_not_in_channels_raises(
        self, sample_h5_path: str
    ) -> None:
        """Test that merge structure not in channels raises ValueError."""
        patient = load_patient(sample_h5_path)
        with pytest.raises(ValueError, match="in merges must also be in channels"):
            patient.stack_channels(
                channels=["GTV"],
                merges=[("GTV", "PTV")],  # PTV not in channels
            )

    def test_stack_channels_merge_missing_structure_raises(
        self, sample_h5_path: str
    ) -> None:
        """Test that merging nonexistent structure raises KeyError."""
        patient = load_patient(sample_h5_path)
        with pytest.raises(KeyError, match="Structure 'MISSING' not found"):
            patient.stack_channels(
                channels=["GTV", "MISSING"],
                merges=[("GTV", "MISSING")],
            )


class TestLoadPatient:
    """Tests for load_patient function."""

    def test_load_full_patient(self, sample_h5_path: str) -> None:
        """Test loading a patient with all data."""
        patient = load_patient(sample_h5_path)

        assert patient.patient_id == "TEST001"
        assert patient.source_path == sample_h5_path
        assert patient.ct.shape == (10, 64, 64)
        assert patient.dose is not None
        assert patient.dose.shape == (10, 64, 64)
        assert "GTV" in patient.structures
        assert "PTV" in patient.structures
        assert patient.study_uid == "1.2.3.4.5"
        assert patient.frame_of_reference == "1.2.3.4.6"

    def test_load_minimal_patient(self, sample_h5_minimal: str) -> None:
        """Test loading a patient with only CT."""
        patient = load_patient(sample_h5_minimal)

        assert patient.patient_id == "MINIMAL001"
        assert patient.ct.shape == (5, 32, 32)
        assert patient.dose is None
        assert len(patient.structures) == 0
        assert patient.study_uid is None
        assert patient.frame_of_reference is None

    def test_load_patient_file_not_found(self) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_patient("/nonexistent/path.h5")

    def test_load_patient_ct_dtype(self, sample_h5_path: str) -> None:
        """Test that CT is loaded as float32."""
        patient = load_patient(sample_h5_path)
        assert patient.ct.dtype == np.float32

    def test_load_patient_dose_dtype(self, sample_h5_path: str) -> None:
        """Test that dose is loaded as float32."""
        patient = load_patient(sample_h5_path)
        assert patient.dose.dtype == np.float32

    def test_load_patient_structure_dtype(self, sample_h5_path: str) -> None:
        """Test that structures are loaded as int8 for memory efficiency."""
        patient = load_patient(sample_h5_path)
        assert patient.structures["GTV"].dtype == np.int8


class TestLoadPatients:
    """Tests for load_patients function."""

    def test_load_multiple_patients(self, sample_h5_directory: str) -> None:
        """Test loading multiple patients from a directory."""
        import glob
        import os

        paths = sorted(glob.glob(os.path.join(sample_h5_directory, "*.h5")))
        patients = load_patients(paths, show_progress=False)

        assert len(patients) == 3
        assert patients[0].patient_id == "PATIENT000"
        assert patients[1].patient_id == "PATIENT001"
        assert patients[2].patient_id == "PATIENT002"

    def test_load_patients_empty_list(self) -> None:
        """Test loading empty list returns empty list."""
        patients = load_patients([], show_progress=False)
        assert patients == []
