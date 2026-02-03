# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

M3CV (Medical Multi-Modal Computer Vision) is a framework for deep learning on medical imaging data. It focuses on multimodal outcome prediction from radiotherapy planning data (CT, dose, structures), with built-in support for combining volumetric imaging with tabular clinical data.

**Status**: Under active development with a three-package architecture:
- `m3cv-dataprep` - ✅ Ready for use
- `m3cv-data` - 🚧 In development
- `m3cv-models` - 📋 Planned

Legacy code in `_legacy/m3cv/` is preserved for reference but not intended for use.

## Design Goals

The framework is designed to support two primary use cases:

1. **Interactive exploration**: Academic medical researchers working in notebooks or REPL environments, exploring data and prototyping models
2. **HPC batch training**: Production training runs on cluster infrastructure via SLURM scripts or similar batch systems

When designing interfaces, keep both use cases in mind:
- APIs should work well interactively (sensible defaults, clear feedback)
- But also support non-interactive execution (config files, CLI flags, no prompts)

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest packages/m3cv-dataprep/src/tests/

# CLI - inspect DICOM data
uv run m3cv-prep inspect [PATH] [--details]

# CLI - pack DICOM to HDF5
uv run m3cv-prep pack [SOURCE_DIR] --out-path [OUTPUT.h5] [--recursive] [--structures "ROI1,ROI2"]

# Lint/format
uv run ruff check .
uv run ruff format .
```

## Architecture

### Package Structure

- **`packages/m3cv-dataprep/`** - DICOM data preparation (ready for use)
  - `src/m3cv_prep/arrays/` - Core classes for medical imaging arrays
  - `src/m3cv_prep/cli/` - Typer-based CLI with `pack` and `inspect` commands
  - `src/m3cv_prep/array_tools.py` - Array construction and sparse packing
  - `src/m3cv_prep/file_handling.py` - HDF5 I/O operations
  - `src/m3cv_prep/dicom_utils.py` - DICOM file utilities
  - `src/m3cv_prep/nonvolume_logic_reference/` - Reference code for tabular data (not active)

- **`packages/m3cv-data/`** - Data loading and augmentation (in development)
  - `src/m3cv_data/_migrated_logic.py` - Preserved augmentation logic for future implementation

- **`packages/m3cv-models/`** - Neural network architectures (planned)
  - Will include ResNet3D, ViT3D, UNet with multimodal fusion support

- **`_legacy/m3cv/`** - Legacy code (reference only, excluded from linting)

### Arrays Subpackage

The `arrays` module uses an inheritance hierarchy for medical imaging data:

- **`PatientArray`** (base) - Spatial operations (rescale, align, bounding_box)
- **`PatientCT`** - CT images in Hounsfield Units, `voidval=-1000.0`
- **`PatientDose`** - Radiation dose arrays, handles PLAN/BEAM types, `voidval=0.0`
- **`PatientMask`** - Binary structure masks from RTSTRUCT, `voidval=0.0`

Key conventions:
- Array ordering is (Z, Y, X) - slices, rows, columns
- `SpatialMetadata` dataclass holds coordinate system info (immutable)
- `Alignable` protocol defines interface for spatial alignment
- Factory methods: `from_dicom_files()`, `from_dicom()`, `from_rtstruct()`
- Augmentation methods (rotate, shift, zoom) are in m3cv-data, not on PatientArray

### Slice Compatibility

When aligning arrays (e.g., dose to CT), slice compatibility is validated:
- Each source slice must align with a target slice (within 10% of slice thickness)
- Aligned slices must be contiguous (no gaps that would leave intermediate slices empty)
- Source extending beyond target is trimmed; target extending beyond source is padded

### Exception Hierarchy

Custom exceptions under `m3cv_prep.arrays.exceptions`:
- `ArrayError` (base)
- `MetadataMismatchError` - Incompatible DICOM metadata across files
- `AlignmentError` - Array alignment failures
- `SliceCompatibilityError` - Slice position/contiguity issues
- `UnevenSpacingError` - Non-uniform slice spacing
- `ROINotFoundError` - Structure not found in RTSTRUCT
- `DoseTypeError` - Unexpected dose file type

## Code Style

- Python 3.11+, line length 88
- Ruff for linting (E, W, F, I, B, C4, UP rules) and formatting
- Type hints throughout, Google-style docstrings
- First-party imports: `m3cv`, `m3cv_prep`, `m3cv_data`, `m3cv_models`
