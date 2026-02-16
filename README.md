# M3CV - Medical Multi-Modal Computer Vision

A comprehensive framework for deep learning research on medical imaging data, with a focus on radiotherapy applications.

[![Status](https://img.shields.io/badge/status-active%20development-blue)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Tests](https://img.shields.io/badge/tests-201%20passing-brightgreen)]()

## Overview

M3CV provides a complete toolkit for training deep learning models on medical imaging data:

1. **📦 Data Preparation** (`m3cv-dataprep`) - Convert DICOM files to ML-ready HDF5 format
2. **🔄 Data Loading** (`m3cv-data`) - Efficient loading, anatomical cropping, and batching
3. **🧠 Model Architectures** (`m3cv-models`) - 3D neural networks with multimodal fusion

**Key Feature:** Multimodal fusion - combine volumetric imaging (CT, dose, structure masks) with tabular clinical data in a single model.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jasbach/M3CV.git
cd M3CV

# Install all packages (recommended)
make install

# Or install with uv directly
uv sync
```

### End-to-End Training Example

Here's a complete workflow from DICOM to trained model:

#### 1. Prepare Your Data

```bash
# Inspect DICOM data to see what's available
uv run m3cv-prep inspect /path/to/dicom_data/ --details

# Pack DICOM into HDF5 format
uv run m3cv-prep pack /path/to/dicom_data/ \
    --out-path ./data/packed/ \
    --recursive \
    --structures "Parotid_L,Parotid_R,GTV,Brainstem"
```

#### 2. Load and Transform Data

```python
from m3cv_data import PatientDataset
from m3cv_data.transforms import AnatomicalCrop, BilateralStructureMidpoint
from torch.utils.data import DataLoader

# Define anatomical cropping strategy (centers volumes on parotid glands)
strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")
crop_transform = AnatomicalCrop(
    crop_shape=(90, 256, 256),  # Standard size for head-and-neck
    reference_strategy=strategy,
    allow_fallback=True,  # Fall back to volume center if structures missing
)

# Create dataset with transforms
dataset = PatientDataset(
    paths="./data/packed/",
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    include_dose=True,
    patient_transform=crop_transform,  # Apply anatomical cropping
    label_fn=lambda p: get_label_from_patient(p),  # Your label function
)

# Split into train/val/test
splits = dataset.random_split(
    fractions={"train": 0.7, "val": 0.15, "test": 0.15},
    seed=42,
)

# Create DataLoader
train_loader = DataLoader(
    splits["train"],
    batch_size=4,
    shuffle=True,
    num_workers=4,
)
```

#### 3. Build and Train Model

```python
from m3cv_models import ResNet3DBuilder, FusionConfig, FusionPoint
import torch
import torch.nn as nn
import torch.optim as optim

# Build ResNet3D model with optional tabular fusion
model = ResNet3DBuilder.build_resnet_18(
    in_channels=5,  # CT + dose + 3 structures
    num_classes=2,  # Binary classification
)

# Or with tabular data fusion
fusion = FusionConfig(
    late=FusionPoint(tabular_dim=10, mode="concat")
)
model = ResNet3DBuilder.build_resnet_50(
    in_channels=5,
    num_classes=2,
    fusion_config=fusion,
)

# Standard PyTorch training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(num_epochs):
    for volumes, labels in train_loader:
        volumes = volumes.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(volumes)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation...
```

See [`examples/`](examples/) for complete training scripts.

## Package Breakdown

### 📦 m3cv-dataprep - DICOM Data Preparation

**Status:** ✅ Production Ready

Converts DICOM radiotherapy data into HDF5 format suitable for deep learning.

**Features:**
- CT, dose, and structure mask processing
- Automatic spatial alignment and resampling
- Sparse mask encoding for efficient storage
- CLI tools for inspection and batch processing

**See:** [packages/m3cv-dataprep/README.md](packages/m3cv-dataprep/README.md)

### 🔄 m3cv-data - Data Loading & PyTorch Integration

**Status:** ✅ Production Ready

Efficient data loading, anatomical cropping, and PyTorch dataset integration.

**Features:**
- `PatientDataset` - PyTorch Dataset with lazy/preload modes
- Anatomical cropping transforms (bilateral structures, COM-based)
- Channel stacking and structure merging
- Dataset splitting with stratification
- HDF5 file inspection utilities

**See:** [packages/m3cv-data/README.md](packages/m3cv-data/README.md)

### 🧠 m3cv-models - Neural Network Architectures

**Status:** ✅ Production Ready

3D neural networks optimized for medical imaging with multimodal fusion.

**Features:**
- ResNet3D (18, 34, 50, 101, 152 layers)
- Early and late fusion for tabular data
- Builder API for easy construction
- Pre-configured for medical imaging dimensions

**See:** [packages/m3cv-models/README.md](packages/m3cv-models/README.md)

## Project Structure

```
M3CV/
├── packages/
│   ├── m3cv-dataprep/    # ✅ DICOM → HDF5 conversion
│   ├── m3cv-data/        # ✅ Data loading & transforms
│   └── m3cv-models/      # ✅ Neural network architectures
├── examples/             # Complete training examples
├── Makefile              # Convenient installation and testing
├── TESTING_QUICK_START.md
└── README.md             # This file
```

## Development

### Running Tests

```bash
# All tests (unit + integration)
make test

# Or run package tests individually
uv run pytest packages/m3cv-dataprep/src/tests/ -v
uv run pytest packages/m3cv-data/src/tests/ -v
uv run pytest packages/m3cv-models/src/tests/ -v

# Skip E2E integration tests (useful for CI)
uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -v
```

See [TESTING_QUICK_START.md](TESTING_QUICK_START.md) for detailed testing information.

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Or use ruff directly
uv run ruff format .
uv run ruff check .
```

## Key Concepts

### Anatomical Cropping

Standard cropping around volume centers can misalign anatomical structures. M3CV provides **anatomical reference cropping** to center volumes on landmarks:

```python
from m3cv_data.transforms import BilateralStructureMidpoint, AnatomicalCrop

# Crop around bilateral parotid glands (head-and-neck)
strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")
crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

# This ensures parotids are consistently positioned across patients
```

### Multimodal Fusion

Combine volumetric imaging with tabular clinical data:

```python
from m3cv_models import FusionConfig, FusionPoint

# Late fusion - tabular features added before classification head
fusion = FusionConfig(
    late=FusionPoint(tabular_dim=10, mode="concat")
)

model = ResNet3DBuilder.build_resnet_50(
    in_channels=2,
    num_classes=2,
    fusion_config=fusion,
)

# Forward pass with tabular data
output = model(volume, tabular={"late": clinical_features})
```

### Channel Stacking

Efficiently combine multiple modalities and structures:

```python
# Stack CT, dose, and structure masks into single tensor
volume = patient.stack_channels(
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    include_dose=True,
    merges=[("Parotid_L", "Parotid_R")],  # Combine bilateral structures
)
# Result: (4, Z, Y, X) tensor - CT, dose, GTV, merged parotids
```

## Citation

M3CV was developed to support radiotherapy outcome prediction research. If you use this framework in your research, please cite:

**Published Research:**

This package supported the publication:

> Asbach JC, Singh AK, Iovoli AJ, Farrugia M, Le AH. Novel pre-spatial data fusion deep learning approach for multimodal volumetric outcome prediction models in radiotherapy. *Med Phys*. 2025; 52: 2675–2687. https://doi.org/10.1002/mp.17672

```bibtex
@article{asbach2025prespatial,
  title={Novel pre-spatial data fusion deep learning approach for multimodal volumetric outcome prediction models in radiotherapy},
  author={Asbach, J. C. and Singh, A. K. and Iovoli, A. J. and Farrugia, M. and Le, A. H.},
  journal={Medical Physics},
  year={2025},
  volume={52},
  pages={2675--2687},
  doi={10.1002/mp.17672},
  url={https://doi.org/10.1002/mp.17672}
}
```

**Software Citation:**

```bibtex
@software{m3cv2026,
  title={M3CV: Medical Multi-Modal Computer Vision},
  author={Asbach, J. C.},
  year={2026},
  url={https://github.com/jasbach/M3CV}
}
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- PyTorch 2.0+
- For DICOM processing: pydicom, numpy, scipy
- For data loading: h5py, torch

## License

See [LICENSE](LICENSE) for details.

## Support

- **Issues:** [GitHub Issues](https://github.com/jasbach/M3CV/issues)
- **Discussions:** [GitHub Discussions](https://github.com/jasbach/M3CV/discussions)
- **Documentation:** See individual package READMEs and docstrings
