"""Pack test DICOM data for development and testing.

This script processes DICOM files from the test dataset and converts them to HDF5
format suitable for use with m3cv-data. It specifically excludes dose files to
avoid DoseGridScaling mismatches that can occur with multi-beam dose files.

Usage:
    uv run python scripts/pack_test_data.py

Input:
    - DICOM files in ./dicom/ directory (one subdirectory per patient)

Output:
    - HDF5 files in ./test_packed/ directory
    - Each file contains: CT array + structure masks
    - Dose is intentionally excluded due to multi-beam compatibility issues

Note:
    This script is specific to the test dataset and serves as a workaround for
    known DoseGridScaling metadata mismatches. For general DICOM packing, use
    the m3cv-prep CLI tool:
        uv run m3cv-prep pack [SOURCE] --out-path [OUTPUT]
"""

from pathlib import Path

from m3cv_prep.array_tools import construct_arrays
from m3cv_prep.dicom_utils import group_dcms_by_modality
from m3cv_prep.file_handling import load_dicom_files_from_directory, save_array_to_h5

# Define structure names to extract from test dataset
STRUCTURE_NAMES = ["Parotid L", "Parotid R", "70 GTV - Tumour", "BrainStem"]


def pack_patient_without_dose(dicom_dir: str, output_path: str) -> None:
    """Pack a single patient, excluding dose files.

    Args:
        dicom_dir: Path to directory containing patient DICOM files
        output_path: Path where HDF5 file should be written

    Raises:
        Various exceptions from m3cv_prep if DICOM files are invalid
    """
    print(f"Processing {dicom_dir}...")

    # Load DICOM files
    dcm_files = load_dicom_files_from_directory(dicom_dir)
    grouped = group_dcms_by_modality(dcm_files)

    # Remove dose files to avoid the DoseGridScaling mismatch issue
    if "RTDOSE" in grouped:
        print(f"  Skipping {len(grouped['RTDOSE'])} dose files (multi-beam issue)")
        del grouped["RTDOSE"]

    # Construct arrays
    ct_array, dose_array, structure_masks = construct_arrays(
        grouped, structure_names=STRUCTURE_NAMES
    )

    # Write to HDF5
    save_array_to_h5(output_path, ct_array, None, structure_masks)

    # Report what was packed
    print(f"  ✓ Packed CT: {ct_array.array.shape}")
    print(f"  ✓ Structures: {', '.join(structure_masks.keys())}")
    print(f"  ✓ Saved to: {output_path}")


def main():
    """Pack all test patients from ./dicom/ to ./test_packed/."""
    dicom_base = Path("/home/johna/repos/M3CV/dicom")
    output_dir = Path("/home/johna/repos/M3CV/test_packed")
    output_dir.mkdir(exist_ok=True)

    if not dicom_base.exists():
        print(f"❌ DICOM directory not found: {dicom_base}")
        print("   Place DICOM data in ./dicom/ directory first.")
        return

    # Process each patient directory
    for patient_dir in sorted(dicom_base.iterdir()):
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            output_path = output_dir / f"{patient_id}.h5"

            try:
                pack_patient_without_dose(str(patient_dir), str(output_path))
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

    print(f"\nDone! Packed {len(list(output_dir.glob('*.h5')))} patients")


if __name__ == "__main__":
    main()
