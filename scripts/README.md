# Development Scripts

Utility scripts for development, testing, and data preparation.

## Scripts

### `pack_test_data.py`

Pack test DICOM data to HDF5 format, excluding dose files to avoid multi-beam compatibility issues.

```bash
uv run python scripts/pack_test_data.py
```

**Input:** DICOM files in `./dicom/` directory
**Output:** HDF5 files in `./test_packed/` directory

**Note:** This script is specific to the test dataset. For general DICOM packing, use:
```bash
uv run m3cv-prep pack [SOURCE] --out-path [OUTPUT]
```

### `setup_e2e_test_data.py`

Copy packed HDF5 files to test fixtures directory to enable E2E integration tests.

```bash
uv run python scripts/setup_e2e_test_data.py
```

**Input:** HDF5 files in `./test_packed/`
**Output:** Files copied to `packages/m3cv-data/src/tests/fixtures/e2e_test_patients/`

After running, E2E tests can be executed with:
```bash
make test-e2e
# or
uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v
```

### `test_anatomical_cropping.py`

Interactive demonstration and validation of anatomical cropping functionality.

```bash
uv run python scripts/test_anatomical_cropping.py
```

**Requirements:** Test data must be packed first (see `pack_test_data.py`)

**Features:**
- Compare volume center vs bilateral parotid cropping
- Demonstrate PatientDataset integration
- Show fallback behavior when structures are missing

This complements the automated tests with visual demonstrations.

## Typical Workflow

1. **Pack test data:**
   ```bash
   uv run python scripts/pack_test_data.py
   ```

2. **Setup E2E test data:**
   ```bash
   uv run python scripts/setup_e2e_test_data.py
   ```

3. **Run tests:**
   ```bash
   make test          # All tests including E2E
   make test-unit     # Unit tests only
   make test-e2e      # E2E tests only
   ```

4. **Optional - Run demonstrations:**
   ```bash
   uv run python scripts/test_anatomical_cropping.py
   ```

## See Also

- **Makefile** - Convenient test targets (`make test`, `make test-e2e`, etc.)
- **TESTING_QUICK_START.md** - Complete testing guide
- **packages/m3cv-data/src/tests/fixtures/README.md** - E2E test data setup details
