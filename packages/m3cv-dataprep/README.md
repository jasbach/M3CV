# m3cv-dataprep

Data preparation package for M3CV - converts DICOM radiotherapy data into HDF5 format for machine learning pipelines.

## Status

✅ **Ready for use.** Core DICOM processing and CLI are functional.

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

### Inspect DICOM Data

Use `inspect` to explore your DICOM files before processing:

```bash
# Summarize patients and modalities in a directory
uv run m3cv-prep inspect /path/to/dicoms

# Show detailed info (structure names, dose types)
uv run m3cv-prep inspect /path/to/dicoms --details
```

### Pack DICOM to HDF5

Use `pack` to convert DICOM files into HDF5 format:

```bash
# Single patient directory
uv run m3cv-prep pack /path/to/dicoms --out-path output.h5

# With structure masks
uv run m3cv-prep pack /path/to/dicoms --out-path output.h5 --structures "PTV,Parotid_L,Parotid_R"

# Recursive (process multiple patient directories)
uv run m3cv-prep pack /path/to/dataset --out-path /path/to/output_dir --recursive
```

### Requirements

- **Single mode**: Source directory should contain one patient's DICOM files
- **Recursive mode**: Each subdirectory with DICOM files is treated as one patient
- Supported modalities: CT, RTDOSE, RTSTRUCT
- Only one RTSTRUCT file per patient directory

## Package Structure

```
m3cv_prep/
├── arrays/           # Core array classes
│   ├── base.py       # PatientArray base class
│   ├── ct.py         # PatientCT
│   ├── dose.py       # PatientDose
│   ├── mask.py       # PatientMask
│   ├── protocols.py  # Alignable, SpatialMetadata
│   └── exceptions.py # Custom exception types
├── cli/              # Typer-based CLI
│   ├── pack.py       # DICOM → HDF5 conversion
│   ├── inspect.py    # DICOM discovery and summary
│   └── _utils.py     # Shared CLI utilities
├── array_tools.py    # Array construction and sparse packing
├── dicom_utils.py    # DICOM file utilities
└── file_handling.py  # HDF5 I/O operations
```

## Roadmap

- [ ] Config-based batch processing
- [ ] Non-volumetric data attachment (clinical variables, surveys)
