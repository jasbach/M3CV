"""Shared utilities for CLI commands."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pydicom
from pydicom.errors import InvalidDicomError

from m3cv_prep.log_util import get_logger

logger = get_logger("cli")


@dataclass
class PatientSummary:
    """Summary of DICOM files for a single patient."""

    patient_id: str
    ct_count: int = 0
    dose_files: list[dict] = field(default_factory=list)
    struct_files: list[dict] = field(default_factory=list)
    other_modalities: dict[str, int] = field(default_factory=dict)

    @property
    def dose_count(self) -> int:
        return len(self.dose_files)

    @property
    def struct_count(self) -> int:
        return len(self.struct_files)


def scan_dicom_directory(
    directory: str,
    recursive: bool = True,
) -> dict[str, PatientSummary]:
    """Scan a directory for DICOM files and summarize by patient.

    Args:
        directory: Root directory to scan.
        recursive: Whether to scan subdirectories.

    Returns:
        Dictionary mapping patient ID to PatientSummary.
    """
    patients: dict[str, PatientSummary] = {}

    if recursive:
        walker = os.walk(directory)
    else:
        walker = [(directory, [], os.listdir(directory))]

    for root, _dirs, files in walker:
        for filename in files:
            filepath = os.path.join(root, filename)
            if not os.path.isfile(filepath):
                continue

            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            except (InvalidDicomError, Exception):
                continue

            patient_id = str(getattr(dcm, "PatientID", "UNKNOWN"))
            modality = str(getattr(dcm, "Modality", "UNKNOWN"))

            if patient_id not in patients:
                patients[patient_id] = PatientSummary(patient_id=patient_id)

            summary = patients[patient_id]

            if modality == "CT":
                summary.ct_count += 1
            elif modality == "RTDOSE":
                dose_info = {
                    "path": filepath,
                    "type": str(getattr(dcm, "DoseSummationType", "UNKNOWN")),
                }
                summary.dose_files.append(dose_info)
            elif modality == "RTSTRUCT":
                # Extract structure names
                struct_names = []
                if hasattr(dcm, "StructureSetROISequence"):
                    for roi in dcm.StructureSetROISequence:
                        struct_names.append(str(roi.ROIName))
                struct_info = {
                    "path": filepath,
                    "structures": struct_names,
                }
                summary.struct_files.append(struct_info)
            else:
                if modality not in summary.other_modalities:
                    summary.other_modalities[modality] = 0
                summary.other_modalities[modality] += 1

    return patients
