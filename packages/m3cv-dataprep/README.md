# m3cv-dataprep

**DICOM Data Preparation for Machine Learning**

Converts DICOM radiotherapy data (CT, dose, structures) into HDF5 format optimized for deep learning pipelines.

**Status:** ✅ Production Ready

## Features

- ✅ **CT Processing** - Load and stack CT slices with spatial metadata preservation
- ✅ **Dose Processing** - Handle PLAN and BEAM dose files with automatic CT grid alignment
- ✅ **Structure Masks** - Rasterize RTSTRUCT contours into binary masks with sparse encoding
- ✅ **Spatial Alignment** - Automatic resampling and alignment with slice compatibility validation
- ✅ **HDF5 Output** - Compact storage with sparse mask encoding and preserved metadata
- ✅ **CLI Tools** - Command-line interface for inspection and batch processing

## Installation

```bash
# From repository root
make install

# Or with uv directly
cd packages/m3cv-dataprep
uv sync
```

## Quick Start

### Inspect DICOM Data

Before processing, inspect your DICOM files to see what's available:

```bash
# Basic summary
uv run m3cv-prep inspect /path/to/dicom/patient_001/

# Output:
# Patient: patient_001
#   CT: 197 files
#   RTDOSE: 3 files
#   RTSTRUCT: 1 file

# Detailed information (shows structure names, dose types)
uv run m3cv-prep inspect /path/to/dicom/patient_001/ --details

# Output includes:
#   - Structure names: GTV, PTV, Parotid_L, Parotid_R, etc.
#   - Dose types: PLAN, BEAM
#   - Patient identifiers
```

### Pack DICOM to HDF5

Convert DICOM files into HDF5 format:

```bash
# Single patient
uv run m3cv-prep pack /path/to/dicom/patient_001/ \
    --out-path ./output/patient_001.h5 \
    --structures "Parotid_L,Parotid_R,GTV,Brainstem"

# Multiple patients (recursive processing)
uv run m3cv-prep pack /path/to/dicom_dataset/ \
    --out-path ./output/ \
    --recursive \
    --structures "Parotid_L,Parotid_R,GTV,Brainstem"

# Without dose (CT and structures only)
uv run m3cv-prep pack /path/to/dicom/patient_001/ \
    --out-path ./output/patient_001.h5 \
    --structures "Parotid_L,Parotid_R"
    # Dose is included by default if RTDOSE files are present
```

### Command Line Options

#### `inspect` command

```bash
uv run m3cv-prep inspect [PATH] [OPTIONS]

Arguments:
  PATH                    Path to DICOM directory or file

Options:
  --details              Show detailed structure names and metadata
  --help                 Show help message
```

#### `pack` command

```bash
# Single patient directory
uv run m3cv-prep pack /path/to/dicoms --out-path output.h5

# With structure masks
uv run m3cv-prep pack /path/to/dicoms --out-path output.h5 --structures "PTV,Parotid_L,Parotid_R"

<<<<<<< Updated upstream
# Recursive (process multiple patient directories)
uv run m3cv-prep pack /path/to/dataset --out-path /path/to/output_dir --recursive
=======
# Without dose (CT and structures only)
uv run m3cv-prep pack /path/to/dicom/patient_001/ \
    --out-path ./output/patient_001.h5 \
    --structures "Parotid_L,Parotid_R"
    # Dose is included by default if RTDOSE files are present

# With an alias file (handles non-standardized ROI names across institutions)
uv run m3cv-prep pack /path/to/dicom_dataset/ \
    --out-path ./output/ \
    --recursive \
    --alias-file structures.json
>>>>>>> Stashed changes
```

### Requirements

<<<<<<< Updated upstream
- **Single mode**: Source directory should contain one patient's DICOM files
- **Recursive mode**: Each subdirectory with DICOM files is treated as one patient
- Supported modalities: CT, RTDOSE, RTSTRUCT
- Only one RTSTRUCT file per patient directory
=======
#### `inspect` command

```bash
uv run m3cv-prep inspect [PATH] [OPTIONS]

Arguments:
  PATH                    Path to DICOM directory or file

Options:
  --details              Show detailed structure names and metadata
  --help                 Show help message
```

#### `pack` command

```bash
uv run m3cv-prep pack [SOURCE] [OPTIONS]

Arguments:
  SOURCE                 Path to DICOM directory

Options:
  --out-path PATH        Output path (file or directory)
  --structures TEXT      Comma-separated list of structure names to include
  --alias-file PATH      JSON file mapping canonical names to DICOM aliases
  --recursive            Process subdirectories as separate patients
  --help                Show help message

Note: `--structures` and `--alias-file` are mutually exclusive.
```

## HDF5 Output Format

The packed HDF5 files have the following structure:

```
patient_001.h5
├── ct                      # CT array (Z, Y, X) - float32
├── dose                    # Dose array (Z, Y, X) - float32 (optional)
├── structures/             # Structure masks (sparse encoding)
│   ├── Parotid_L/
│   │   ├── rows            # Y indices
│   │   ├── cols            # X indices
│   │   └── slices          # Z indices
│   ├── Parotid_R/
│   └── GTV/
├── spatial_metadata/       # Spatial reference information
│   ├── position_x          # Patient position
│   ├── position_y
│   ├── position_z
│   ├── pixel_size_row      # Pixel spacing
│   ├── pixel_size_col
│   ├── slice_thickness
│   ├── slice_ref           # Z coordinates of each slice
│   └── even_spacing        # Whether slices are evenly spaced
└── attributes              # Patient identifiers
    ├── patient_id
    ├── study_uid
    └── frame_of_reference
```

**Sparse Structure Encoding:** Structure masks are stored as coordinate lists (rows, cols, slices) rather than full 3D arrays, dramatically reducing file size for masks with small volumes.

## Requirements and Expected Input

### DICOM Directory Structure

**Single mode** (default):
```
patient_001/
├── CT.*.dcm              # CT image slices
├── RD.*.dcm              # Dose files (optional)
└── RS.*.dcm              # Structure set (optional)
```

**Recursive mode** (`--recursive`):
```
dataset/
├── patient_001/          # Each subdirectory = one patient
│   ├── CT.*.dcm
│   ├── RD.*.dcm
│   └── RS.*.dcm
├── patient_002/
└── patient_003/
```

### Supported Modalities

- ✅ **CT** - Computed Tomography (required)
- ✅ **RTDOSE** - Radiation dose (optional, can be PLAN or BEAM type)
- ✅ **RTSTRUCT** - Structure sets/ROIs (optional)

### Constraints

- One patient per directory (non-recursive mode)
- Only one RTSTRUCT file per patient
- Multiple RTDOSE files are merged if compatible
- CT slice spacing must be uniform
- Dose and structures are automatically aligned to CT grid

## Python API

While the CLI is recommended, you can use the Python API directly:

```python
from m3cv_prep.array_tools import construct_arrays
from m3cv_prep.dicom_utils import group_dcms_by_modality
from m3cv_prep.file_handling import load_dicom_files_from_directory, save_array_to_h5

# Load DICOM files
dcm_files = load_dicom_files_from_directory("/path/to/patient/")

# Group by modality
grouped = group_dcms_by_modality(dcm_files)

# Construct arrays (exact-match names)
ct_array, dose_array, structure_masks = construct_arrays(
    grouped,
    structure_names=["Parotid_L", "Parotid_R", "GTV"]
)

# Or use alias maps for non-standardized ROI names
ct_array, dose_array, structure_masks = construct_arrays(
    grouped,
    structure_aliases={
        "Parotid_L": ["Parotid_L", "parotid_lt", "Lt_Parotid"],
        "Parotid_R": ["Parotid_R", "parotid_rt", "Rt_Parotid"],
        "GTV":       ["GTV", "GTV_70", "gtv_primary"],
    }
)

# Save to HDF5
save_array_to_h5(
    output_path="output.h5",
    ct=ct_array,
    dose=dose_array,
    structures=structure_masks
)
```
>>>>>>> Stashed changes

## Package Structure

```
m3cv_prep/
├── arrays/                 # Core array classes
│   ├── base.py            # PatientArray base class with spatial operations
│   ├── ct.py              # PatientCT (Hounsfield Units)
│   ├── dose.py            # PatientDose (Gy)
│   ├── mask.py            # PatientMask (binary masks)
│   ├── protocols.py       # Alignable protocol, SpatialMetadata
│   └── exceptions.py      # Custom exception types
├── cli/                   # Command-line interface
│   ├── pack.py            # DICOM → HDF5 conversion
│   ├── inspect.py         # DICOM discovery and summary
│   └── _utils.py          # Shared CLI utilities
├── array_tools.py         # Array construction and sparse packing
├── dicom_utils.py         # DICOM file utilities and validation
└── file_handling.py       # HDF5 I/O operations
```

## Error Handling

Common errors and solutions:

**`MetadataMismatchError: Mismatched shape attributes in dose files`**
- Multiple dose files with incompatible metadata (e.g., different grid scaling)
- Solution: Use only compatible dose files or process separately

**`ROINotFoundError: ROI 'StructureName' not found`**
- Requested structure doesn't exist in RTSTRUCT
- Solution: Check available structures with `inspect --details`

**`SliceCompatibilityError`**
- Dose/structure slices don't align with CT slices
- Solution: Check that all DICOM files are from the same study

**`UnevenSpacingError`**
- CT slices have non-uniform spacing
- Solution: This is currently not supported; use uniformly-spaced CT scans

## Advanced Usage

### Structure Name Patterns

Structure names often vary between institutions. Common patterns:

```bash
# Different naming conventions
--structures "Parotid_L,Parotid_R"      # Underscore
--structures "Parotid L,Parotid R"      # Space
--structures "L_Parotid,R_Parotid"      # Prefix

# Always check available names first
uv run m3cv-prep inspect /path/to/data/ --details
```

### Batch Processing Script

For processing many patients:

```bash
#!/bin/bash
for patient_dir in /data/patients/*/; do
    patient_id=$(basename "$patient_dir")
    uv run m3cv-prep pack "$patient_dir" \
        --out-path "/output/${patient_id}.h5" \
        --structures "Parotid_L,Parotid_R,GTV"
done
```

## Testing

```bash
# Run tests
uv run pytest src/tests/ -v

# Test coverage includes:
# - Array construction and alignment (75 tests)
# - DICOM utilities
# - HDF5 I/O
# - Sparse mask encoding
```

## Performance

Typical processing times (Intel i7, 16GB RAM):

- Single patient (197 CT slices, dose, 5 structures): ~3-5 seconds
- HDF5 file size: ~400 MB per patient (CT-dominated)
- Sparse mask encoding reduces structure size by >95%

## Troubleshooting

### DICOM Files Not Found

```bash
# Check file extensions
ls -la /path/to/patient/*.dcm
ls -la /path/to/patient/*.DCM

# Verify DICOM format
uv run m3cv-prep inspect /path/to/patient/
```

### Memory Issues with Large Datasets

```bash
# Process patients individually instead of recursive mode
for dir in /data/patients/*/; do
    uv run m3cv-prep pack "$dir" --out-path "/output/$(basename $dir).h5"
done
```

## Integration with m3cv-data

The HDF5 files produced by m3cv-prep are designed to be loaded by m3cv-data:

```python
from m3cv_data import load_patient, PatientDataset

# Load single patient
patient = load_patient("patient_001.h5")

# Create PyTorch dataset
dataset = PatientDataset(
    paths="./output/",
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    include_dose=True,
)
```

See [m3cv-data documentation](../m3cv-data/README.md) for data loading details.

## Error Handling

Common errors and solutions:

**`MetadataMismatchError: Mismatched shape attributes in dose files`**
- Multiple dose files with incompatible metadata (e.g., different grid scaling)
- Solution: Use only compatible dose files or process separately

**`ROINotFoundError: ROI 'StructureName' not found`**
- Requested structure doesn't exist in RTSTRUCT
- Solution: Check available structures with `inspect --details`

**`SliceCompatibilityError`**
- Dose/structure slices don't align with CT slices
- Solution: Check that all DICOM files are from the same study

**`UnevenSpacingError`**
- CT slices have non-uniform spacing
- Solution: This is currently not supported; use uniformly-spaced CT scans

## Advanced Usage

### Structure Name Patterns

Structure names often vary between institutions. Common patterns:

```bash
# Different naming conventions
--structures "Parotid_L,Parotid_R"      # Underscore
--structures "Parotid L,Parotid R"      # Space
--structures "L_Parotid,R_Parotid"      # Prefix

# Always check available names first
uv run m3cv-prep inspect /path/to/data/ --details
```

### Batch Processing Script

For processing many patients:

```bash
#!/bin/bash
for patient_dir in /data/patients/*/; do
    patient_id=$(basename "$patient_dir")
    uv run m3cv-prep pack "$patient_dir" \
        --out-path "/output/${patient_id}.h5" \
        --structures "Parotid_L,Parotid_R,GTV"
done
```

## Testing

```bash
# Run tests
uv run pytest src/tests/ -v

# Test coverage includes:
# - Array construction and alignment (75 tests)
# - DICOM utilities
# - HDF5 I/O
# - Sparse mask encoding
```

## Performance

Typical processing times (Intel i7, 16GB RAM):

- Single patient (197 CT slices, dose, 5 structures): ~3-5 seconds
- HDF5 file size: ~400 MB per patient (CT-dominated)
- Sparse mask encoding reduces structure size by >95%

## Troubleshooting

### DICOM Files Not Found

```bash
# Check file extensions
ls -la /path/to/patient/*.dcm
ls -la /path/to/patient/*.DCM

# Verify DICOM format
uv run m3cv-prep inspect /path/to/patient/
```

### Memory Issues with Large Datasets

```bash
# Process patients individually instead of recursive mode
for dir in /data/patients/*/; do
    uv run m3cv-prep pack "$dir" --out-path "/output/$(basename $dir).h5"
done
```

## Integration with m3cv-data

The HDF5 files produced by m3cv-prep are designed to be loaded by m3cv-data:

```python
from m3cv_data import load_patient, PatientDataset

# Load single patient
patient = load_patient("patient_001.h5")

# Create PyTorch dataset
dataset = PatientDataset(
    paths="./output/",
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    include_dose=True,
)
```

See [m3cv-data documentation](../m3cv-data/README.md) for data loading details.

<<<<<<< Updated upstream
## Roadmap

- [ ] Config-based batch processing
- [ ] Non-volumetric data attachment (clinical variables, surveys)
=======
## Error Handling

Common errors and solutions:

**`MetadataMismatchError: Mismatched shape attributes in dose files`**
- Multiple dose files with incompatible metadata (e.g., different grid scaling)
- Solution: Use only compatible dose files or process separately

**`ROINotFoundError: ROI 'StructureName' not found`**
- Requested structure doesn't exist in RTSTRUCT
- Solution: Check available structures with `inspect --details`

**`SliceCompatibilityError`**
- Dose/structure slices don't align with CT slices
- Solution: Check that all DICOM files are from the same study

**`UnevenSpacingError`**
- CT slices have non-uniform spacing
- Solution: This is currently not supported; use uniformly-spaced CT scans

## Advanced Usage

### Structure Alias Maps

ROI names in DICOM files are not standardized — the same anatomical structure may be called
`"Parotid_L"`, `"parotid_lt"`, `"Lt_Parotid"`, or `"PAROTID_LEFT"` across institutions.
The `--alias-file` option lets you define canonical names with lists of possible aliases so
the same pack job works across a heterogeneous dataset.

Create a JSON file mapping each canonical name to a list of aliases to try (first match wins):

```json
{
  "Parotid_L": ["Parotid_L", "parotid_lt", "Lt_Parotid", "PAROTID_LEFT"],
  "Parotid_R": ["Parotid_R", "parotid_rt", "Rt_Parotid", "PAROTID_RIGHT"],
  "GTV":       ["GTV", "GTV_70", "gtv_primary"]
}
```

```bash
# Always check available structure names first
uv run m3cv-prep inspect /path/to/data/ --details

# Pack with alias resolution — output HDF5 keys are always the canonical names
uv run m3cv-prep pack /path/to/dicom_dataset/ \
    --out-path ./output/ \
    --recursive \
    --alias-file structures.json
```

The canonical names become the HDF5 group keys (`structures/Parotid_L`, etc.), so
downstream code can use consistent names regardless of the source institution.
If no alias matches a patient's RTSTRUCT, an error is raised for that patient
(in recursive mode the patient is skipped and processing continues).

You can also use `structure_aliases` directly via the Python API:

```python
ct_array, dose_array, structure_masks = construct_arrays(
    grouped,
    structure_aliases={
        "Parotid_L": ["Parotid_L", "parotid_lt", "Lt_Parotid"],
        "GTV":       ["GTV", "GTV_70"],
    }
)
# Keys in structure_masks are always the canonical names
```

### Batch Processing Script

For processing many patients:

```bash
#!/bin/bash
for patient_dir in /data/patients/*/; do
    patient_id=$(basename "$patient_dir")
    uv run m3cv-prep pack "$patient_dir" \
        --out-path "/output/${patient_id}.h5" \
        --structures "Parotid_L,Parotid_R,GTV"
done
```

## Testing

```bash
# Run tests
uv run pytest src/tests/ -v

# Test coverage includes:
# - Array construction and alignment (75 tests)
# - DICOM utilities
# - HDF5 I/O
# - Sparse mask encoding
```

## Performance

Typical processing times (Intel i7, 16GB RAM):

- Single patient (197 CT slices, dose, 5 structures): ~3-5 seconds
- HDF5 file size: ~400 MB per patient (CT-dominated)
- Sparse mask encoding reduces structure size by >95%

## Troubleshooting

### DICOM Files Not Found

```bash
# Check file extensions
ls -la /path/to/patient/*.dcm
ls -la /path/to/patient/*.DCM

# Verify DICOM format
uv run m3cv-prep inspect /path/to/patient/
```

### Memory Issues with Large Datasets

```bash
# Process patients individually instead of recursive mode
for dir in /data/patients/*/; do
    uv run m3cv-prep pack "$dir" --out-path "/output/$(basename $dir).h5"
done
```

## Integration with m3cv-data

The HDF5 files produced by m3cv-prep are designed to be loaded by m3cv-data:

```python
from m3cv_data import load_patient, PatientDataset

# Load single patient
patient = load_patient("patient_001.h5")

# Create PyTorch dataset
dataset = PatientDataset(
    paths="./output/",
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    include_dose=True,
)
```

See [m3cv-data documentation](../m3cv-data/README.md) for data loading details.

## Roadmap

Future enhancements:

- [x] Structure alias maps for non-standardized ROI names (`--alias-file`)
- [ ] Config file support for batch processing
- [ ] Support for non-uniform slice spacing
- [ ] Multi-RTSTRUCT handling (merge from multiple files)
- [ ] Tabular clinical data attachment
- [ ] Progress bars for batch processing

## Contributing

Contributions welcome! Areas for improvement:
- Additional DICOM modality support
- Performance optimizations
- Enhanced error messages
- Documentation improvements

## Citation

If you use this package in your research, please cite the published work:

> Asbach JC, Singh AK, Iovoli AJ, Farrugia M, Le AH. Novel pre-spatial data fusion deep learning approach for multimodal volumetric outcome prediction models in radiotherapy. *Med Phys*. 2025; 52: 2675–2687. https://doi.org/10.1002/mp.17672

See the [main README](../../README.md#citation) for full citation details.

## License

See [LICENSE](../../LICENSE) for details.
>>>>>>> Stashed changes
