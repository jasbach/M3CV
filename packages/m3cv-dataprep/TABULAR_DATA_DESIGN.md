# Tabular Data Integration Design Document

This document specifies the design for attaching tabular clinical data to HDF5 patient files created by `m3cv-dataprep`. The feature enables multimodal fusion workflows where volumetric imaging data (CT, dose, structures) is combined with tabular clinical variables in a single model.

## Table of Contents

1. [Overview](#overview)
2. [Workflow](#workflow)
3. [CLI Interface](#cli-interface)
4. [Field Type Inference](#field-type-inference)
5. [Date Handling](#date-handling)
6. [HDF5 Storage Format](#hdf5-storage-format)
7. [Behavior Specifications](#behavior-specifications)
8. [Integration with m3cv-data](#integration-with-m3cv-data)
9. [Examples](#examples)
10. [Implementation Notes](#implementation-notes)

---

## Overview

### Purpose

Clinical outcomes in medical imaging often depend on both imaging features and tabular clinical variables (demographics, staging, treatment, etc.). This feature allows users to attach tabular data from a CSV file to existing HDF5 patient files, enabling downstream multimodal fusion in `m3cv-data` and `m3cv-models`.

### Design Principles

1. **Two-step workflow**: DICOM packing and tabular attachment are separate operations
2. **Key-based matching**: User specifies which CSV column maps to HDF5 `patient_id`
3. **Automatic type inference**: Parse column names to detect encodings and field types
4. **Relative dates**: All date fields converted to days-from-reference (user-specified)
5. **Clean storage**: Field names are cleaned; original names and mappings stored as metadata
6. **No HIPAA handling**: Anonymization is the user's responsibility

---

## Workflow

### Expected User Flow

```
Step 1: Pack DICOM files (existing functionality)
─────────────────────────────────────────────────
$ m3cv-prep pack /dicom/patient001 --out-path packed/patient001.h5
$ m3cv-prep pack /dicom/patient002 --out-path packed/patient002.h5
...
(or batch with --recursive)

Step 2: Attach tabular data (new functionality)
───────────────────────────────────────────────
$ m3cv-prep attach-tabular clinical_data.csv \
    --key MRN \
    --reference-date "Date of Diagnosis" \
    --h5-dir packed/
```

### Why Two Steps?

- Tabular data typically comes as a single CSV with many patients
- DICOM data comes as separate directories per patient
- Separating the steps allows flexibility (re-run tabular without re-packing, update clinical data independently, etc.)

---

## CLI Interface

### Command Signature

```bash
m3cv-prep attach-tabular <csv_path> \
    --key <column_name> \
    --reference-date <date_column_name> \
    --h5-dir <directory> \
    [--h5-files <file1.h5> <file2.h5> ...] \
    [--force] \
    [--dry-run] \
    [--exclude <column1> <column2> ...] \
    [--verbose]
```

### Arguments and Options

| Argument/Option | Required | Description |
|-----------------|----------|-------------|
| `csv_path` | Yes | Path to the CSV file containing tabular data |
| `--key` | Yes | Column name in CSV that matches HDF5 `patient_id` attribute |
| `--reference-date` | Yes | Column name to use as the reference for relative date calculations |
| `--h5-dir` | No* | Directory containing HDF5 files to update |
| `--h5-files` | No* | Explicit list of HDF5 files to update |
| `--force` | No | Overwrite existing `/tabular` group if present (default: error) |
| `--dry-run` | No | Report what would be done without modifying files |
| `--exclude` | No | Column names to exclude from storage (e.g., the key column itself) |
| `--verbose` | No | Print detailed progress and inference decisions |

*One of `--h5-dir` or `--h5-files` must be provided.

### Output

The command should produce clear console output:

```
Attaching tabular data from clinical_data.csv
  Key column: MRN → patient_id
  Reference date: Date of Diagnosis
  Found 150 rows in CSV
  Found 142 HDF5 files in packed/

Field type inference:
  Sex (1=F, 2=M)                    → Sex [categorical]
  Age at diagnosis                  → Age_at_diagnosis [continuous]
  Race (1=B, 2=W, 3=A, 4=O, 99=Unkn) → Race [categorical]
  Date of Diagnosis                 → (reference date, excluded)
  Date of Death/Last Follow-Up      → Date_of_Death_Last_Follow_Up [date → days]
  Deceased (0=N, 1=Y)               → Deceased [binary]
  cT-Classification (0=T0, ...)     → cT_Classification [categorical]
  ...

Processing:
  ✓ patient001.h5 (MRN: 12345)
  ✓ patient002.h5 (MRN: 12346)
  ⚠ patient003.h5 - no matching row in CSV, skipping
  ✓ patient004.h5 (MRN: 12348)
  ...

Summary:
  Updated: 138 files
  Skipped (no CSV match): 4 files
  CSV rows not matched to HDF5: 12
```

---

## Field Type Inference

The system automatically infers field types by analyzing column names and values. This section specifies the inference algorithm in detail.

### Inference Priority Order

Apply these rules in order; first match wins:

1. **Explicit exclusion**: If column is in `--exclude` list or is the `--key` column, skip it
2. **Reference date column**: If column matches `--reference-date`, mark as reference (don't store as field)
3. **Embedded categorical pattern**: Parse column name for `(value=label, ...)` patterns
4. **Date detection**: Column name contains "date" or "DOB" (case-insensitive)
5. **Numeric detection**: All non-null values are numeric
6. **Fallback**: Treat as string/categorical

### Pattern Parsing for Categoricals

#### Binary Pattern

Pattern: `(0=N, 1=Y)` or `(0=N, 1=Y, 99=Unkn)` or similar

```python
# Regex pattern for binary fields
BINARY_PATTERN = r'\(0\s*=\s*N.*1\s*=\s*Y.*\)'
```

Examples:
- `Deceased (0=N, 1=Y)` → binary
- `HPV+ (0=N, 1=Y, 99=Unkn)` → binary with missing code
- `Chemotherapy (0=N, 1=Y, 99=Unkn)` → binary with missing code

#### General Categorical Pattern

Pattern: `(value1=label1, value2=label2, ...)` where values are typically integers

```python
# Regex pattern for categorical mappings
CATEGORICAL_PATTERN = r'\(([^)]+)\)'
MAPPING_PATTERN = r'(\d+)\s*=\s*([^,\)]+)'
```

Examples:
- `Sex (1=F, 2=M)` → categorical, mapping: {1: "F", 2: "M"}
- `Race (1=B, 2=W, 3=A, 4=O, 99=Unkn)` → categorical, mapping: {1: "B", 2: "W", 3: "A", 4: "O", 99: "Unkn"}

#### Complex Ordinal Pattern

Pattern: Values like `T0, T1, T1a, T1b, T2, T3, T4, T4a, T4b` or `N0, N1, N2a, N2b, N2c, N3`

```python
# Regex for staging patterns
STAGING_PATTERN = r'(\d+)\s*=\s*(T\d[ab]?|N\d[abc]?|TX|NX)'
```

Examples:
- `cT-Classification (0=T0, 10=T1, 11=T1a, 12=T1b, 20=T2, 30=T3, 40=T4, 41=T4a, 42=T4b, 99=TX)`
- `cN-Classification AJCC 7th Ed (0=N0, 10=N1, 20=N2, 21=N2a, 22=N2b, 23=N2c, 30=N3, 99=NX)`

These are stored as categorical with the full mapping. The ordinal nature is preserved in the numeric codes (0 < 10 < 11 < 12 < 20 < ...).

### Field Name Cleaning

Original column names are cleaned for storage:

1. Remove the parenthetical encoding description
2. Replace spaces with underscores
3. Replace special characters (`/`, `-`, `+`) with underscores
4. Remove consecutive underscores
5. Strip leading/trailing underscores

Examples:
| Original | Cleaned |
|----------|---------|
| `Sex (1=F, 2=M)` | `Sex` |
| `Age at diagnosis` | `Age_at_diagnosis` |
| `Date of Death/Last Follow-Up` | `Date_of_Death_Last_Follow_Up` |
| `HPV+ (0=N, 1=Y, 99=Unkn)` | `HPV` |
| `cT-Classification (0=T0, ...)` | `cT_Classification` |

### Inference Output Structure

For each field, the inference produces:

```python
@dataclass
class FieldSpec:
    original_name: str          # "Sex (1=F, 2=M)"
    clean_name: str             # "Sex"
    field_type: FieldType       # categorical, binary, continuous, date
    mapping: dict[int, str] | None  # {1: "F", 2: "M"} or None
    missing_codes: list[int]    # [99] if 99=Unkn pattern found
    is_ordinal: bool            # True for staging variables
```

### Handling Pre-Anonymized Data

If a column name contains "date" but the values are already numeric (floats/ints), treat it as continuous rather than attempting date parsing. This supports workflows where dates have already been converted to days-from-reference before export.

```python
def infer_date_field(column_name: str, values: pd.Series) -> FieldType:
    if "date" not in column_name.lower() and "dob" not in column_name.lower():
        return None  # Not a date field

    # Check if values are already numeric
    if pd.api.types.is_numeric_dtype(values.dropna()):
        return FieldType.CONTINUOUS  # Pre-anonymized

    # Try parsing as dates
    try:
        pd.to_datetime(values.dropna().head(10))
        return FieldType.DATE
    except:
        return FieldType.CONTINUOUS  # Unparseable, treat as continuous
```

---

## Date Handling

### Reference Date Requirement

The `--reference-date` argument is **required** (no default). This ensures users make an explicit choice about the temporal reference point.

Common reference date choices:
- `Date of Diagnosis` - for disease-centric analyses
- `Rad Start Date` - for treatment-centric analyses

### Conversion Algorithm

For each date field (excluding the reference date itself):

```python
def convert_to_relative_days(
    date_value: str | datetime,
    reference_date: str | datetime,
) -> float | None:
    """Convert a date to days from reference date.

    Returns:
        Float days (can be negative if before reference).
        None if either date is missing/unparseable.
    """
    if pd.isna(date_value) or pd.isna(reference_date):
        return None

    date_dt = pd.to_datetime(date_value)
    ref_dt = pd.to_datetime(reference_date)

    delta = date_dt - ref_dt
    return delta.total_seconds() / 86400  # Convert to days as float
```

### Storage of Date Fields

- Original date columns are **not stored** (PII concern, user handles anonymization)
- Relative days values are stored as float datasets
- The reference date column name is stored in metadata
- The reference date values themselves are **not stored** (only used for calculation)

### Example

| Original Column | Reference Date | Stored Value |
|-----------------|----------------|--------------|
| Date of Diagnosis | (this is the reference) | Not stored |
| Date of Death/Last Follow-Up | 2020-01-15 → 2021-06-20 | 522.0 |
| Date of Local Recurrence | 2020-01-15 → 2020-08-30 | 228.0 |
| Date of Local Recurrence | 2020-01-15 → (missing) | NaN |

---

## HDF5 Storage Format

### Group Structure

```
/tabular/
│
├── Sex                          # Dataset: int array [1, 2, 1, 2, ...]
├── Age_at_diagnosis             # Dataset: float array [54.0, 67.0, ...]
├── Race                         # Dataset: int array [2, 1, 3, 4, ...]
├── Deceased                     # Dataset: int array [0, 1, 0, 0, ...]
├── Date_of_Death_Last_Follow_Up # Dataset: float array [522.0, NaN, ...]
├── cT_Classification            # Dataset: int array [10, 20, 41, ...]
└── ...
```

### Attributes on `/tabular` Group

```python
{
    # Mapping from clean names to original names
    "original_names": {
        "Sex": "Sex (1=F, 2=M)",
        "Age_at_diagnosis": "Age at diagnosis",
        "Race": "Race (1=B, 2=W, 3=A, 4=O, 99=Unkn)",
        ...
    },

    # Field type for each field
    "field_types": {
        "Sex": "categorical",
        "Age_at_diagnosis": "continuous",
        "Race": "categorical",
        "Deceased": "binary",
        "Date_of_Death_Last_Follow_Up": "continuous",  # stored as days
        "cT_Classification": "categorical",
        ...
    },

    # Categorical mappings (only for categorical/binary fields)
    "mappings": {
        "Sex": {"1": "F", "2": "M"},
        "Race": {"1": "B", "2": "W", "3": "A", "4": "O", "99": "Unkn"},
        "Deceased": {"0": "N", "1": "Y"},
        "cT_Classification": {"0": "T0", "10": "T1", "11": "T1a", ...},
        ...
    },

    # Missing value codes (for fields that use sentinel values)
    "missing_codes": {
        "Race": [99],
        "HPV": [99],
        "Clinical_ENE": [99],
        ...
    },

    # Which fields are ordinal (vs nominal) categoricals
    "ordinal_fields": ["cT_Classification", "cN_Classification_AJCC_7th_Ed", ...],

    # Reference date field (for documentation, values not stored)
    "reference_date_field": "Date of Diagnosis",

    # Fields that were converted from dates to relative days
    "date_derived_fields": [
        "Date_of_Death_Last_Follow_Up",
        "Date_of_Local_Recurrence",
        ...
    ],

    # Key column used for matching (for documentation)
    "key_column": "MRN",

    # Timestamp of when tabular data was attached
    "attached_at": "2025-02-10T14:30:00Z",
}
```

### Data Types for Datasets

| Field Type | HDF5 dtype | Notes |
|------------|------------|-------|
| Categorical | `int32` | Uses the numeric codes from CSV |
| Binary | `int32` | 0 or 1 (or missing code like 99) |
| Continuous | `float32` | NaN for missing values |
| Date-derived | `float32` | Days from reference, NaN for missing |

### Handling Missing Values

- **Numeric fields**: Use `NaN` (stored as IEEE float NaN)
- **Categorical with missing code**: Keep the sentinel value (e.g., 99) and document in `missing_codes`
- **Categorical without missing code**: Use -1 as missing indicator, document in metadata

---

## Behavior Specifications

### Patient Matching

```
CSV Key Column (e.g., MRN)  ←→  HDF5 patient_id attribute
```

Both values are converted to strings for comparison. Leading/trailing whitespace is stripped.

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| CSV row has no matching HDF5 file | Skip silently (not an error) |
| HDF5 file has no matching CSV row | Warn to console, skip `/tabular` creation for that file |
| `/tabular` group already exists | Error with message, unless `--force` flag provided |
| `--force` with existing `/tabular` | Delete existing group, create new one |
| CSV has duplicate keys | Error: "Duplicate key values found in CSV: [list]" |
| HDF5 has no `patient_id` attribute | Error: "HDF5 file missing patient_id attribute: {path}" |
| Reference date column not found | Error: "Reference date column '{name}' not found in CSV" |
| Reference date value missing for a patient | Warn, store NaN for all date-derived fields for that patient |

### Validation

Before processing, validate:

1. CSV file exists and is readable
2. Key column exists in CSV
3. Reference date column exists in CSV
4. No duplicate keys in CSV
5. At least one HDF5 file found
6. All HDF5 files have `patient_id` attribute

### Dry Run Mode

With `--dry-run`:
- Parse and validate CSV
- Run field type inference
- Report which files would be updated
- Report any warnings (missing matches, etc.)
- Do not modify any files

---

## Integration with m3cv-data

### Patient Dataclass Updates

The `Patient` dataclass in `m3cv-data` should be extended:

```python
@dataclass
class Patient:
    ct: NDArray[np.floating]
    patient_id: str
    source_path: str
    dose: NDArray[np.floating] | None = None
    structures: dict[str, NDArray] = field(default_factory=dict)
    study_uid: str | None = None
    frame_of_reference: str | None = None
    spatial_metadata: Any | None = None

    # New field for tabular data
    tabular: TabularData | None = None
```

### TabularData Class

```python
@dataclass
class TabularData:
    """Container for tabular clinical data loaded from HDF5.

    Attributes:
        fields: Dictionary mapping field names to values.
        field_types: Dictionary mapping field names to their types.
        mappings: Dictionary of categorical mappings.
        missing_codes: Dictionary of missing value codes.
        reference_date_field: Name of the reference date column used.
    """
    fields: dict[str, float | int]
    field_types: dict[str, str]
    mappings: dict[str, dict[str, str]]
    missing_codes: dict[str, list[int]]
    reference_date_field: str

    def get_field(self, name: str) -> float | int:
        """Get a field value by name."""
        return self.fields[name]

    def get_decoded(self, name: str) -> str | float | int:
        """Get a field value, decoding categoricals to their labels."""
        value = self.fields[name]
        if name in self.mappings:
            return self.mappings[name].get(str(int(value)), value)
        return value

    def is_missing(self, name: str) -> bool:
        """Check if a field value is missing."""
        value = self.fields[name]
        if np.isnan(value):
            return True
        if name in self.missing_codes:
            return int(value) in self.missing_codes[name]
        return False

    def to_tensor(
        self,
        fields: list[str] | None = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Convert tabular data to a tensor for model input.

        Args:
            fields: List of field names to include. If None, include all.
            normalize: If True, normalize continuous fields (requires fitted scaler).

        Returns:
            1D tensor of field values.
        """
        ...
```

### Loading Changes

Update `load_patient()` to load tabular data:

```python
def load_patient(path: str, load_tabular: bool = True) -> Patient:
    with h5py.File(path, "r") as f:
        # ... existing loading code ...

        # Load tabular data if present
        tabular = None
        if load_tabular and "tabular" in f:
            tabular = _load_tabular_data(f["tabular"])

    return Patient(..., tabular=tabular)
```

### PatientDataset Updates

The `PatientDataset` should support including tabular data in the output:

```python
class PatientDataset(Dataset):
    def __init__(
        self,
        ...,
        tabular_fields: list[str] | None = None,  # New parameter
    ):
        self._tabular_fields = tabular_fields

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        # Returns (volume, tabular_tensor, label)
        # tabular_tensor is None if tabular_fields not specified
        ...
```

---

## Examples

### Example 1: Basic Usage

```bash
# After packing DICOM files
$ m3cv-prep attach-tabular data/clinical.csv \
    --key MRN \
    --reference-date "Date of Diagnosis" \
    --h5-dir output/packed/
```

### Example 2: With Exclusions and Force

```bash
$ m3cv-prep attach-tabular data/clinical.csv \
    --key MRN \
    --reference-date "Date of Diagnosis" \
    --h5-dir output/packed/ \
    --exclude "DOB" "Date of Diagnosis" \
    --force \
    --verbose
```

### Example 3: Dry Run to Preview

```bash
$ m3cv-prep attach-tabular data/clinical.csv \
    --key MRN \
    --reference-date "Date of Diagnosis" \
    --h5-dir output/packed/ \
    --dry-run
```

### Example 4: Using in m3cv-data

```python
from m3cv_data import load_patient, PatientDataset

# Load single patient
patient = load_patient("packed/patient001.h5")
print(patient.tabular.fields)
# {'Sex': 1, 'Age_at_diagnosis': 54.0, 'Race': 2, ...}

print(patient.tabular.get_decoded("Sex"))
# 'F'

print(patient.tabular.is_missing("HPV"))
# True (if value is 99)

# Use in dataset with tabular fields
dataset = PatientDataset(
    paths="packed/",
    channels=["GTV"],
    include_ct=True,
    include_dose=True,
    tabular_fields=["Age_at_diagnosis", "Sex", "cT_Classification"],
)

volume, tabular, label = dataset[0]
# volume: (C, Z, Y, X) tensor
# tabular: (3,) tensor with [age, sex, t_stage]
```

---

## Implementation Notes

### Suggested Module Structure

```
packages/m3cv-dataprep/src/m3cv_prep/
├── tabular/
│   ├── __init__.py
│   ├── inference.py      # Field type inference logic
│   ├── parsing.py        # Column name parsing, cleaning
│   ├── storage.py        # HDF5 read/write for tabular data
│   └── exceptions.py     # TabularError, InferenceError, etc.
├── cli/
│   ├── attach_tabular.py # New CLI command
│   └── ...
```

### Key Functions to Implement

```python
# inference.py
def infer_field_specs(df: pd.DataFrame, exclude: list[str]) -> list[FieldSpec]:
    """Infer field specifications from a DataFrame."""
    ...

def parse_categorical_mapping(column_name: str) -> dict[int, str] | None:
    """Parse categorical mapping from column name like 'Sex (1=F, 2=M)'."""
    ...

def clean_field_name(column_name: str) -> str:
    """Clean a column name for storage."""
    ...

# storage.py
def write_tabular_group(
    h5_file: h5py.File,
    row: pd.Series,
    field_specs: list[FieldSpec],
    reference_date_value: Any,
    force: bool = False,
) -> None:
    """Write tabular data to HDF5 file."""
    ...

def read_tabular_group(group: h5py.Group) -> TabularData:
    """Read tabular data from HDF5 group."""
    ...

# cli/attach_tabular.py
def attach_tabular_command(
    csv_path: Path,
    key: str,
    reference_date: str,
    h5_dir: Path | None,
    h5_files: list[Path] | None,
    force: bool,
    dry_run: bool,
    exclude: list[str],
    verbose: bool,
) -> None:
    """CLI command implementation."""
    ...
```

### Testing Strategy

1. **Unit tests for inference**: Test pattern parsing, field type detection
2. **Unit tests for storage**: Test HDF5 read/write round-trip
3. **Integration tests**: Test full CLI workflow with sample data
4. **Edge case tests**: Missing values, pre-anonymized dates, duplicate keys

### Dependencies

No new dependencies required. Uses:
- `pandas` (already a dependency)
- `h5py` (already a dependency)
- `typer` (already a dependency for CLI)

---

## Open Questions / Future Considerations

1. **One-hot encoding**: Should we provide an option to store categoricals as one-hot encoded arrays instead of integer codes? (Defer to m3cv-data transforms)

2. **Normalization metadata**: Should we compute and store normalization parameters (mean, std) for continuous fields during attachment? (Defer to m3cv-data)

3. **Validation against existing data**: Should we validate that attached tabular data is consistent across a cohort (same fields, same mappings)? (Nice to have)

4. **CSV alternatives**: Support for Excel files, Parquet, etc.? (Defer, CSV is sufficient for now)
