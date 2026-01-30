# M3CV-Data: Capabilities Migration Checklist

This document tracks data augmentation and loading capabilities that need to be migrated
from various locations in the codebase into this unified package.

## Package Scope

**m3cv-data** is responsible for:
- Loading patient data from HDF5 files produced by m3cv-dataprep
- Data augmentation (transforms applied during training)
- PyTorch Dataset/DataLoader integration
- Batching and streaming for training pipelines

## Augmentation Operations to Implement

### Core Spatial Transforms

| Operation | Status | Source Implementations | Notes |
|-----------|--------|----------------------|-------|
| **Rotate** | TODO | `PatientArray.rotate()`, `pending_refactor/data_augmentation.py`, `src/m3cv/dl/preprocessing/data_augmentation.py` | Rotation about Z-axis. Parameters: degree_range (default 15), explicit degrees |
| **Shift** | TODO | `PatientArray.shift()`, `pending_refactor/data_augmentation.py` | Translation in Y/X plane (no Z shift). Parameters: max_shift fraction (default 0.2), explicit pixelshift tuple |
| **Zoom** | TODO | `PatientArray.zoom()`, `pending_refactor/data_augmentation.py` | Uniform scaling with crop/pad. Parameters: max_zoom_factor (default 0.2 = 0.8-1.2x range) |
| **Downsample** | TODO | `src/m3cv/dl/preprocessing/data_augmentation.py` | Per-axis downsampling. Parameters: factor (default 0.5), axis list. **Only in legacy code** |
| **Bounding Box / Crop** | TODO | `PatientArray.bounding_box()`, `pending_refactor/data_augmentation.py` | Extract region around center point |

### Intensity Transforms (CT-specific)

| Operation | Status | Source | Notes |
|-----------|--------|--------|-------|
| **Window/Level** | TODO | `PatientCT.window_level()`, `dicom_utils.window_level()` | Radiological windowing. Parameters: window, level, normalize flag |

## Design Decisions

### Seeding & Reproducibility

**Current implementations use two patterns:**

1. **Legacy (avoid)**: `np.random.seed()` - global state, not thread-safe
2. **Modern (preferred)**: `np.random.Generator` - local state, thread-safe

**Recommendation**: Use `np.random.Generator` throughout. Accept optional `rng` parameter,
create default generator if not provided.

```python
def rotate(array, degree_range=15.0, rng=None, degrees=None):
    if rng is None:
        rng = np.random.default_rng()
    # ...
```

### Multi-Modality Coordination

Medical imaging augmentation requires **spatial consistency** across modalities (CT, dose, mask
must transform identically). Current approaches:

1. **Shared seed**: Pass same seed to each array's transform method
2. **Preprocessor orchestration**: `handler.py` coordinates transforms with shared seed

**Recommendation**: Implement transforms that accept multiple arrays and apply coordinated
transforms, OR use a transform pipeline that generates parameters once and applies to all.

```python
# Option A: Transform accepts multiple arrays
def rotate_multi(arrays: list[np.ndarray], degree_range=15.0, rng=None):
    degrees = _sample_degrees(degree_range, rng)
    return [_apply_rotation(arr, degrees) for arr in arrays]

# Option B: Separate parameter generation from application
params = RotateParams.sample(degree_range=15.0, rng=rng)
ct_aug = params.apply(ct_array, voidval=-1000)
dose_aug = params.apply(dose_array, voidval=0)
```

### Void Values by Modality

| Modality | Void Value | Rationale |
|----------|------------|-----------|
| CT | -1000.0 | Air in Hounsfield Units |
| Dose | 0.0 | No radiation |
| Mask | 0.0 | Outside structure |

Transforms that introduce padding must use the appropriate void value.

## Data Loading Capabilities

### From Legacy Code (`src/m3cv/dl/io/`)

| Capability | Status | Source | Notes |
|------------|--------|--------|-------|
| HDF5 file reading | TODO | `DataLoading.py`, `_datautils.py` | Load CT, dose, masks from packed files |
| Sparse mask unpacking | TODO | `array_tools.py`, `_preprocess_util.py` | COO format to dense array |
| Label loading | TODO | `DataLoading.py` | Binary/multiclass labels from CSV or HDF5 attributes |
| Supplemental data | TODO | `DataLoading.py` | Clinical variables, survey responses |
| Train/val/test splits | TODO | `DataLoading.py` | Stratified splitting |
| Batch generation | TODO | `_datautils.py` (`gen_inputs`) | Generator-based batching with augmentation |

### PyTorch Integration (New)

| Capability | Status | Notes |
|------------|--------|-------|
| `PatientDataset` | TODO | PyTorch Dataset wrapping HDF5 files |
| Lazy loading | TODO | Memory-map large arrays, load on demand |
| Collate functions | TODO | Custom batching for variable-size inputs |
| DataLoader config | TODO | num_workers, prefetch, pin_memory |

## Batch Augmentation Workflow

The `build_augments()` function in `pending_refactor/data_augmentation.py` provides a
batch workflow for pre-computing augmented versions:

```python
def build_augments(filelist, iterations=3, augments=2):
    """
    For each HDF5 file:
    1. Load CT, dose, masks
    2. Stack into 4D array (Z, Y, X, channels)
    3. For each iteration:
       a. Randomly select `augments` operations from [zoom, rotate, shift]
       b. Apply with shared seed
       c. Save to HDF5 group with metadata (ops, seed)
    """
```

**Decision needed**: Pre-compute augmentations vs on-the-fly during training?
- Pre-compute: Faster training, but uses more disk space
- On-the-fly: More variety, less disk, but slower

## File Structure Plan

```
packages/m3cv-data/
├── src/
│   └── m3cv_data/
│       ├── __init__.py
│       ├── transforms/           # Augmentation operations
│       │   ├── __init__.py
│       │   ├── spatial.py        # rotate, shift, zoom, crop
│       │   ├── intensity.py      # window_level, normalize
│       │   └── compose.py        # Transform pipelines
│       ├── datasets/             # PyTorch Dataset implementations
│       │   ├── __init__.py
│       │   ├── patient.py        # Single-patient dataset
│       │   └── multi.py          # Multi-patient dataset
│       └── utils/                # Loading utilities
│           ├── __init__.py
│           └── hdf5.py           # HDF5 reading helpers
└── tests/
```

## Migration Priority

1. **High**: Core spatial transforms (rotate, shift, zoom) - needed for any training
2. **High**: HDF5 loading with sparse mask support - needed to read dataprep output
3. **Medium**: PyTorch Dataset wrapper - enables standard training loops
4. **Medium**: Transform composition/pipelines - cleaner API
5. **Low**: Batch pre-augmentation workflow - optimization
6. **Low**: Downsample - rarely used

## Dependencies

**Required:**
- `numpy` - array operations
- `scipy.ndimage` - interpolation for transforms
- `h5py` - HDF5 file reading
- `torch` - PyTorch integration

**Optional:**
- `opencv-python` - alternative resize implementations (if needed)

## References

Source files for migration:
- `packages/m3cv-dataprep/src/m3cv_prep/arrays/base.py` (lines 350-475) - modern implementation
- `packages/m3cv-dataprep/src/m3cv_prep/handler.py` - orchestration pattern
- `packages/m3cv-dataprep/src/m3cv_prep/pending_refactor/data_augmentation.py` - batch workflow
- `src/m3cv/dl/preprocessing/data_augmentation.py` - includes downsample
- `src/m3cv/dl/io/_datautils.py` - data loading and generation
- `src/m3cv/dl/io/DataLoading.py` - DataLoader class
