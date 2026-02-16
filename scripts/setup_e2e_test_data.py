"""Setup script for E2E integration test data.

This script copies packed HDF5 test data to the test fixtures directory, enabling
E2E integration tests. The test data is not committed to the repository due to
data sharing restrictions, so tests automatically skip when data is absent.

Usage:
    # After packing test data with pack_test_data.py or m3cv-prep:
    uv run python scripts/setup_e2e_test_data.py

    # Then run E2E tests:
    make test-e2e
    # or
    uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v

Input:
    - HDF5 files in ./test_packed/ directory

Output:
    - Copies HDF5 files to packages/m3cv-data/src/tests/fixtures/e2e_test_patients/
    - Enables E2E integration tests that validate the full data pipeline

Note:
    - Do not commit the copied test files (they're in .gitignore)
    - E2E tests will automatically skip if data is not present
    - See packages/m3cv-data/src/tests/fixtures/README.md for details
"""

import shutil
from pathlib import Path


def setup_test_data():
    """Copy test data from ./test_packed/ to test fixtures directory.

    Returns:
        bool: True if successful, False if source data not found
    """
    source_dir = Path("test_packed")
    fixtures_dir = Path("packages/m3cv-data/src/tests/fixtures/e2e_test_patients")

    # Check source directory exists
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        print("   Run the data packing first:")
        print("     uv run python scripts/pack_test_data.py")
        print("   Or use m3cv-prep to pack your own DICOM data.")
        return False

    # Get HDF5 files
    h5_files = list(source_dir.glob("*.h5"))
    if not h5_files:
        print(f"❌ No HDF5 files found in {source_dir}")
        print("   Pack DICOM data first with:")
        print("     uv run python scripts/pack_test_data.py")
        return False

    # Create fixtures directory
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"Copying {len(h5_files)} HDF5 files to {fixtures_dir}...")
    for src_file in h5_files:
        dst_file = fixtures_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"  ✓ {src_file.name}")

    print("\n✅ E2E test data setup complete!")
    print(f"   Files copied: {len(h5_files)}")
    print(f"   Location: {fixtures_dir}")
    print("\nRun E2E tests with:")
    print("  make test-e2e")
    print("  # or")
    print("  uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v")

    return True


if __name__ == "__main__":
    success = setup_test_data()
    exit(0 if success else 1)
