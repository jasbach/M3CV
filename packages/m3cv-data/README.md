# m3cv-data

Data loading and PyTorch integration for M3CV medical imaging pipelines.

## Status

Under development. Core loading functionality is implemented; data augmentation is not yet available.

## Installation

```bash
uv sync
```

## Usage

### Inspecting HDF5 Files

Before loading data, you can quickly inspect HDF5 files to understand their contents without loading the full arrays into memory:

```python
from m3cv_data import inspect_h5, inspect_directory, summary_table

# Inspect a single file
info = inspect_h5("patient.h5")
print(info.patient_id)        # "PATIENT001"
print(info.ct_shape)          # (120, 512, 512)
print(info.structure_names)   # ["GTV", "PTV", "Parotid_L", "Parotid_R"]
print(info.has_dose)          # True

# Inspect all files in a directory
infos = inspect_directory("data/processed/", recursive=True)

# Print a formatted summary table
summary_table(infos)
```

### Loading Patient Data

For direct access to patient data (useful for exploration and debugging):

```python
from m3cv_data import load_patient, load_patients

# Load a single patient
patient = load_patient("patient.h5")
print(patient.patient_id)           # "PATIENT001"
print(patient.shape)                # (120, 512, 512) - CT shape (Z, Y, X)
print(patient.available_structures) # ["GTV", "PTV", "Parotid_L", "Parotid_R"]

# Access raw arrays
ct_array = patient.ct           # Shape: (Z, Y, X), dtype: float32
dose_array = patient.dose       # Shape: (Z, Y, X), dtype: float32 (or None)
gtv_mask = patient.structures["GTV"]  # Shape: (Z, Y, X), dtype: int8

# Stack into multi-channel array for model input
volume = patient.stack_channels(
    channels=["GTV", "PTV"],
    include_ct=True,
    include_dose=True,
)
# Result shape: (4, Z, Y, X) - CT, dose, GTV, PTV

# Merge bilateral structures into a single channel
volume = patient.stack_channels(
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    merges=[("Parotid_L", "Parotid_R")],  # Combine via logical OR
)
# Result shape: (3, Z, Y, X) - CT, merged parotids, GTV

# Load multiple patients
patients = load_patients(["patient1.h5", "patient2.h5"], show_progress=True)
```

### PyTorch Dataset

For training pipelines, use `PatientDataset` which integrates with PyTorch DataLoader:

```python
from torch.utils.data import DataLoader
from m3cv_data import PatientDataset, patient_collate_fn

# Create dataset from a directory of HDF5 files
dataset = PatientDataset(
    paths="data/processed/",      # Directory or list of file paths
    channels=["GTV", "PTV"],      # Structures to include
    include_ct=True,
    include_dose=True,
    preload=False,                # False = lazy loading (default)
)

# Get a sample
volume, label = dataset[0]        # volume: torch.Tensor (C, Z, Y, X)

# Access the underlying Patient object for debugging
patient = dataset.get_patient(0)

# Split into train/val/test
splits = dataset.random_split(
    fractions={"train": 0.7, "val": 0.15, "test": 0.15},
    seed=42,
)
train_dataset = splits["train"]
val_dataset = splits["val"]
test_dataset = splits["test"]

# Stratified split (maintain class distribution)
splits = dataset.random_split(
    fractions={"train": 0.8, "val": 0.2},
    seed=42,
    stratify_by=lambda p: p.patient_id[:2],  # Group by ID prefix
)

# Use with DataLoader
loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=patient_collate_fn,
)

for volumes, labels in loader:
    # volumes: (B, C, Z, Y, X)
    # labels: (B,)
    pass
```

### Custom Labels

By default, `PatientDataset` returns `-1` for all labels. Provide a `label_fn` to extract labels from patient data:

```python
# Example: binary classification based on patient ID pattern
dataset = PatientDataset(
    paths="data/processed/",
    channels=["GTV"],
    label_fn=lambda patient: 1 if "high_risk" in patient.patient_id else 0,
)
```

### Transforms

Apply transforms to volumes before returning:

```python
def normalize(tensor):
    # Custom normalization
    return (tensor - tensor.mean()) / tensor.std()

dataset = PatientDataset(
    paths="data/processed/",
    channels=["GTV"],
    transform=normalize,
)
```

## API Reference

### Patient Loading

| Function | Description |
|----------|-------------|
| `load_patient(path)` | Load a single patient from an HDF5 file |
| `load_patients(paths)` | Load multiple patients from a list of paths |

### Inspection

| Function | Description |
|----------|-------------|
| `inspect_h5(path)` | Inspect an HDF5 file without loading arrays |
| `inspect_directory(path)` | Inspect all HDF5 files in a directory |
| `summary_table(infos)` | Print a formatted table of file information |

### PyTorch Integration

| Class/Function | Description |
|----------------|-------------|
| `PatientDataset` | PyTorch Dataset with lazy/preload modes |
| `patient_collate_fn` | Collate function for DataLoader |

## Not Yet Implemented

- Data augmentation transforms (rotation, shifting, scaling, intensity)
- Tabular data integration

We are intentionally deferring these features until the end-to-end training flow is verified working correctly.
