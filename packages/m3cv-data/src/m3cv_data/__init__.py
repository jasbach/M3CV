"""M3CV Data - Data loading, augmentation, and batching for medical imaging."""

from m3cv_data.datasets import PatientDataset, patient_collate_fn
from m3cv_data.inspect import H5FileInfo, inspect_directory, inspect_h5, summary_table
from m3cv_data.patient import Patient, load_patient, load_patients

__version__ = "0.1.0"

__all__ = [
    # Patient loading
    "Patient",
    "load_patient",
    "load_patients",
    # Inspection utilities
    "H5FileInfo",
    "inspect_h5",
    "inspect_directory",
    "summary_table",
    # PyTorch datasets
    "PatientDataset",
    "patient_collate_fn",
]
