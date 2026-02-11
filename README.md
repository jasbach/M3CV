# M3CV

**Medical Multi-Modal Computer Vision** - A framework for deep learning research on medical imaging data, with a focus on radiotherapy applications.

## Overview

M3CV provides tools for the complete deep learning pipeline on medical imaging data:

1. **Data Preparation**: Convert DICOM files to ML-ready HDF5 format
2. **Data Loading**: Efficient loading, augmentation, and batching
3. **Model Architectures**: Ready-to-use neural networks with multimodal support

A key feature is **multimodal fusion** - combining volumetric imaging data (CT, dose, structures) with tabular clinical data in a single model.

## Packages

| Package | Status | Description |
|---------|--------|-------------|
| [m3cv-dataprep](packages/m3cv-dataprep/) | ✅ Ready | DICOM to HDF5 conversion with CLI |
| [m3cv-data](packages/m3cv-data/) | 🚧 In Development | Data loading and PyTorch integration |
| [m3cv-models](packages/m3cv-models/) | 📋 Planned | Neural network architectures (ResNet3D, ViT, UNet) |

### m3cv-data Current State

The data loading package provides the core functionality needed to load HDF5 patient files and feed them into PyTorch training pipelines:

**Implemented:**
- `Patient` dataclass - Container for CT, dose, and structure mask arrays
- `load_patient()` / `load_patients()` - Load data from HDF5 files
- `stack_channels()` - Combine modalities into multi-channel arrays with optional structure merging
- `PatientDataset` - PyTorch Dataset with lazy loading or preload modes
- `random_split()` - Split datasets with optional stratification
- `inspect_h5()` / `inspect_directory()` - Explore HDF5 files without loading full arrays
- `patient_collate_fn` - Batch collation for DataLoader

**Not yet implemented:**
- Data augmentation transforms (rotation, shifting, scaling, intensity augmentation)
- Tabular data integration (design complete, implementation pending)

We are intentionally deferring augmentation until the end-to-end training flow is verified working correctly. The augmentation logic from the legacy codebase is preserved in `_migrated_logic.py` for future integration.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jasbach/M3CV.git
cd M3CV

# Install with uv
uv sync
```

### Convert DICOM to HDF5

```bash
# Inspect your DICOM data
uv run m3cv-prep inspect /path/to/dicom/data --details

# Pack into HDF5
uv run m3cv-prep pack /path/to/patient --out-path patient.h5 --structures "PTV,Parotid_L,Parotid_R"
```

See [m3cv-dataprep README](packages/m3cv-dataprep/README.md) for full CLI documentation.

### Load Data for Training

```python
from m3cv_data import load_patient, PatientDataset

# Load a single patient
patient = load_patient("patient.h5")
print(patient.shape)  # (Z, Y, X)
print(patient.available_structures)  # ['PTV', 'Parotid_L', 'Parotid_R']

# Stack into multi-channel tensor
volume = patient.stack_channels(
    channels=["PTV", "Parotid_L", "Parotid_R"],
    include_ct=True,
    include_dose=True,
    merges=[("Parotid_L", "Parotid_R")],  # Combine bilateral structures
)
# Result: (4, Z, Y, X) - CT, dose, PTV, merged parotids

# Create PyTorch Dataset
dataset = PatientDataset(
    paths="data/processed/",
    channels=["GTV", "PTV"],
    include_ct=True,
    include_dose=True,
)
volume, label = dataset[0]  # Returns (C, Z, Y, X) tensor

# Split into train/val/test
splits = dataset.random_split(
    fractions={"train": 0.7, "val": 0.15, "test": 0.15},
    seed=42,
)
train_dataset = splits["train"]
```

## Project Structure

```
M3CV/
├── packages/
│   ├── m3cv-dataprep/    # ✅ DICOM → HDF5 conversion
│   ├── m3cv-data/        # 🚧 Data loading & PyTorch integration
│   └── m3cv-models/      # 📋 Neural network architectures
├── _legacy/              # Old implementation (reference only)
└── _oldfiles/            # Archived files
```

## Legacy Code

The `_legacy/` directory contains an earlier implementation of this framework. It is **not intended for use** but is preserved as a reference during the refactoring process. Key patterns from the legacy code (multimodal fusion, data augmentation) are being migrated into the new modular package structure.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Development

```bash
# Run tests
uv run pytest packages/m3cv-dataprep/src/tests/
uv run pytest packages/m3cv-data/src/tests/

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## License

See [LICENSE](LICENSE) for details.
