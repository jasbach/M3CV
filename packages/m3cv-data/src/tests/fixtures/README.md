# Test Data Fixtures

This directory can contain real DICOM data for end-to-end integration testing. The E2E tests will automatically skip if this data is not present.

## Setup (Optional)

The E2E integration tests (`test_integration_e2e.py`) require real medical imaging data to test the complete pipeline. Since this data cannot be committed to the repository due to data sharing restrictions, it must be set up locally.

### Required Data Structure

Place HDF5 files packed from DICOM data in this directory:

```
fixtures/
├── README.md (this file)
├── e2e_test_patients/
│   ├── patient_001.h5
│   ├── patient_002.h5
│   ├── patient_003.h5
│   └── patient_004.h5
└── .gitignore (excludes *.h5 files)
```

### Creating Test Data

1. **From DICOM files:**
   ```bash
   # Pack your DICOM data with bilateral structures
   uv run m3cv-prep pack /path/to/dicom/patient_001/ \
       --out-path fixtures/e2e_test_patients/patient_001.h5 \
       --structures "Parotid L,Parotid R,GTV,BrainStem"
   ```

2. **Requirements:**
   - At least 2 patients (4+ recommended for robust testing)
   - Head-and-neck CT scans preferred (for bilateral structure tests)
   - Must include bilateral structures (e.g., Parotid L/R)
   - CT volumes should be at least 90 slices

### Running E2E Tests

```bash
# With test data present - runs all tests
uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v

# Without test data - tests are automatically skipped
uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v
# Output: "SKIPPED [1] ... E2E test data not available"

# Run only E2E tests (when data is present)
uv run pytest packages/m3cv-data/src/tests/ -m e2e -v

# Exclude E2E tests (useful for CI without test data)
uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -v
```

## What Gets Tested

The E2E tests validate the complete pipeline:
1. **Data Loading** - Load patients from HDF5 files
2. **Anatomical Cropping** - Apply bilateral parotid-centered cropping
3. **Dataset Integration** - Use PatientDataset with transforms
4. **DataLoader Batching** - Batch loading with consistent shapes
5. **Model Forward Pass** - Pass through ResNet3D model
6. **Gradient Flow** - Verify backpropagation works
7. **Memory/Shape Verification** - Ensure realistic dimensions work

## CI/CD Considerations

In continuous integration environments without test data:
- E2E tests will be skipped automatically
- Unit tests continue to run normally
- No test failures will occur

To run E2E tests in CI, set up test data as a CI artifact or use a dedicated test data repository.

## Data Privacy

**Important:** Do not commit HDF5 files containing patient data to version control, even if anonymized, unless you have appropriate data sharing agreements.

The `.gitignore` in this directory is configured to exclude all `.h5` files.
