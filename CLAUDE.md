# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

M3CV (Medical Multi-Modal Computer Vision) is a framework for deep learning on medical imaging data. It focuses on multimodal outcome prediction from radiotherapy planning data (CT, dose, structures).

**Status**: Under active development. The `m3cv-dataprep` package is actively maintained; the root `src/m3cv/` contains legacy deep learning code pending refactor.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest packages/m3cv-dataprep/src/tests/

# Run CLI for DICOM packing
uv run m3cv-prep [SOURCE_DIR] --out-path [OUTPUT.h5] [--recursive] [--structures "ROI1,ROI2"]

# Lint/format
uv run ruff check .
uv run ruff format .
```

## Architecture

### Package Structure

- **`packages/m3cv-dataprep/`** - Active package for DICOM data preparation
  - `src/m3cv_prep/arrays/` - Core modular classes for medical imaging arrays
  - `src/m3cv_prep/cli.py` - Typer-based CLI entry point
  - `src/m3cv_prep/array_tools.py` - Array construction and sparse packing
  - `src/m3cv_prep/file_handling.py` - HDF5 I/O operations
  - `src/m3cv_prep/dicom_utils.py` - DICOM file utilities

- **`src/m3cv/`** - Legacy deep learning pipeline (excluded from linting)

### Arrays Subpackage

The `arrays` module uses an inheritance hierarchy for medical imaging data:

- **`PatientArray`** (base) - Spatial operations, augmentation methods
- **`PatientCT`** - CT images in Hounsfield Units, `voidval=-1000.0`
- **`PatientDose`** - Radiation dose arrays, handles PLAN/BEAM types, `voidval=0.0`
- **`PatientMask`** - Binary structure masks from RTSTRUCT, `voidval=0.0`

Key conventions:
- Array ordering is (Z, Y, X) - slices, rows, columns
- `SpatialMetadata` dataclass holds coordinate system info (immutable)
- `Alignable` protocol defines interface for spatial alignment
- Factory methods: `from_dicom_files()`, `from_dicom()`, `from_rtstruct()`

### Exception Hierarchy

Custom exceptions under `m3cv_prep.arrays.exceptions`:
- `ArrayError` (base), `MetadataMismatchError`, `AlignmentError`, `UnevenSpacingError`, `ROINotFoundError`, `DoseTypeError`

## Code Style

- Python 3.11+, line length 88
- Ruff for linting (E, W, F, I, B, C4, UP rules) and formatting
- Type hints throughout, Google-style docstrings
- First-party imports: `m3cv`, `m3cv_prep`
