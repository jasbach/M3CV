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
| [m3cv-data](packages/m3cv-data/) | 🚧 In Development | Data loading, augmentation, PyTorch integration |
| [m3cv-models](packages/m3cv-models/) | 📋 Planned | Neural network architectures (ResNet3D, ViT, UNet) |

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

## Project Structure

```
M3CV/
├── packages/
│   ├── m3cv-dataprep/    # ✅ DICOM → HDF5 conversion
│   ├── m3cv-data/        # 🚧 Data loading & augmentation
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

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## License

See [LICENSE](LICENSE) for details.
