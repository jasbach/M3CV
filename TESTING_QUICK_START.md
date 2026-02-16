# Testing Quick Start Guide

## Current Status

✅ **201 tests** - All passing
- m3cv-dataprep: 75 tests
- m3cv-data: 100 tests (84 unit + 16 E2E)
- m3cv-models: 26 tests

## Running Tests

### All Tests (Per Package)

```bash
# m3cv-dataprep
uv run pytest packages/m3cv-dataprep/src/tests/ -v

# m3cv-data (includes E2E if data present)
uv run pytest packages/m3cv-data/src/tests/ -v

# m3cv-models
uv run pytest packages/m3cv-models/src/tests/ -v
```

### Unit Tests Only (Skip E2E)

```bash
uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -v
```

### E2E Tests Only

```bash
# First-time setup (local only, don't commit)
uv run python scripts/setup_e2e_test_data.py

# Run E2E tests
uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v
```

### Quick Validation

```bash
# Run all tests quickly (no verbose output)
uv run pytest packages/m3cv-dataprep/src/tests/ -q
uv run pytest packages/m3cv-data/src/tests/ -q
uv run pytest packages/m3cv-models/src/tests/ -q
```

## CI/CD Usage

For continuous integration without test data:

```bash
# This will skip E2E tests automatically
uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -v
```

## Test Data Notes

- **E2E test data is NOT committed** to the repository
- E2E tests automatically skip if data is absent
- No test failures when data is missing
- See `packages/m3cv-data/src/tests/fixtures/README.md` for setup

## Test Coverage by Area

| Area | Coverage | Tests |
|------|----------|-------|
| DICOM data prep | ✅ Excellent | 75 |
| Data loading | ✅ Excellent | 18 |
| Transforms | ✅ Excellent | 27 |
| Dataset/DataLoader | ✅ Excellent | 24 |
| Inspection utils | ✅ Good | 10 |
| Model blocks | ✅ Good | 15 |
| Model builders | ✅ Good | 6 |
| Fusion config | ✅ Adequate | 5 |
| **E2E integration** | ✅ **Comprehensive** | **16** |

## What E2E Tests Validate

Real end-to-end workflows:
- ✅ Load real patient data from HDF5
- ✅ Apply anatomical cropping (bilateral parotids)
- ✅ Batch loading with consistent shapes
- ✅ Model forward pass with realistic dimensions
- ✅ Gradient flow through complete pipeline

See `E2E_TESTS_SUMMARY.md` for details.
