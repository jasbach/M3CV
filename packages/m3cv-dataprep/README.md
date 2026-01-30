# m3cv-dataprep

Data preparation package for M3CV - converts DICOM radiotherapy data into HDF5 format for machine learning pipelines.

## Status

Under active development. Core DICOM processing is functional. CLI refinements and config-based execution coming soon.

## Features

- **CT Processing**: Load and stack CT slices with proper spatial metadata
- **Dose Processing**: Handle PLAN and BEAM dose files with automatic alignment to CT grid
- **Structure Masks**: Rasterize RTSTRUCT contours into binary masks
- **Spatial Alignment**: Automatic resampling and alignment with slice compatibility validation
- **HDF5 Output**: Compact storage with sparse mask encoding and preserved metadata

## Installation

```bash
uv sync
```

## CLI Usage

Pack DICOM files from a directory into an HDF5 file:

```bash
# Single directory
uv run m3cv-prep /path/to/dicoms --out-path output.h5

# With structure masks
uv run m3cv-prep /path/to/dicoms --out-path output.h5 --structures "PTV,Parotid_L,Parotid_R"

# Recursive (process multiple patient directories)
uv run m3cv-prep /path/to/dataset --out-path /path/to/output_dir --recursive
```

### Requirements

- Source directory should contain one patient's DICOM files
- Supported modalities: CT, RTDOSE, RTSTRUCT
- Only one RTSTRUCT file per directory

## Package Structure

```
m3cv_prep/
├── arrays/          # Core array classes (PatientCT, PatientDose, PatientMask)
├── cli.py           # Typer-based CLI
├── array_tools.py   # Array construction and sparse packing
├── dicom_utils.py   # DICOM file utilities
└── file_handling.py # HDF5 I/O operations
```

## Roadmap

- [ ] Improved CLI with better error messages and progress feedback
- [ ] Config-based batch processing
- [ ] Non-volumetric data attachment (clinical variables, surveys)
