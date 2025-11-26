import os
import typer
import json
from typing_extensions import Annotated
from rich import print
from rich.progress import track

from m3cv_prep.file_handling import (
    load_dicom_files_from_directory, 
    save_array_to_h5
)
from m3cv_prep.dicom_utils import (
    validate_fields,
    validate_patientid,
    group_dcms_by_modality
)
from m3cv_prep.array_tools import construct_arrays
from m3cv_prep.arrayclasses import PatientCT, PatientDose

app = typer.Typer(help="Data Preparation CLI for M3CV")

"""
How to use the PACK command:

CURRENTLY pack is the entrypoint into the app, so you don't need to explicitly call it,
just call the m3cv-dataprep CLI and it will run `pack` by default.

uv run m3cv-dataprep [SOURCE] --out-path [OUT_PATH] --recursive --structures [STRUCTURES]

Arguments:
    SOURCE: Path to the directory containing DICOM files. If not provided, defaults to the current working directory.
Options:
    --out-path: Path to save the output HDF5 file. If not provided, defaults to "packed_dicom.h5" in the source directory.
    --recursive: If set, the tool will recursively search through subdirectories for DICOM files.
    --structures: JSON string or path to JSON file mapping desired structure names to lists of possible ROI names in the RTSTRUCT.

The structures JSON format is meant to standardize structure names across different naming conventions. It should be formatted as follows:
{
    "Standard_Structure_Name1": ["Possible_ROI_Name1", "Possible_ROI_Name2"],
    "Standard_Structure_Name2": ["Possible_ROI_Name3", "Possible_ROI_Name4"]
}
All DICOM files with ROIs that match any of the names in the lists will be stored with the standard structure name in the HDF5 output.

Example usage:
    uv run m3cv-dataprep /path/to/dicom/files --out-path /path/to/output/packed_data.h5 
        --recursive --structures '{"PTV": ["PTV1", "PTV"], "parotid_l": ["Parotid (Left)", "Left Parotid"]}'
"""

@app.command()
def pack(
    source: Annotated[str, typer.Argument()] = None,
    out_path: Annotated[str, typer.Option()] = None,
    recursive: Annotated[bool, typer.Option()] = False,
    structures: Annotated[str, typer.Option()] = None,
):
    if source is None:
        source = os.getcwd()
    
    if out_path is None:
        out_path = os.path.join(source, "packed_dicom.h5") # TODO - read patient ID from DICOMs for default filename

    if structures is not None:
        print(f"[blue]Evaluating structures argument...[/blue]")
        try:
            if "'" in structures:
                structures = structures.replace("'", '"')
            structure_dict = json.loads(structures)
            print(f"[green]Parsed structures JSON successfully.[/green]")
        except json.JSONDecodeError as e:
            print(f"[blue]Attempting to read structure name map as JSON file:[/blue] {structures}")
            with open(structures, 'r') as f:
                structure_dict = json.load(f)
        

    if not recursive:
        dcm_files = load_dicom_files_from_directory(source)
        if not dcm_files:
            print("[yellow]No valid DICOM files found.[/yellow]")
            raise typer.Exit(code=1)
        pid = validate_patientid(dcm_files)
        grouped = group_dcms_by_modality(dcm_files)
        ct_array, dose_array, structure_masks = construct_arrays(
            grouped,
            structure_dict=structure_dict if structures else None
        )
        save_array_to_h5(out_path, ct_array, dose_array, structure_masks, patient_id=pid)
    else:
        for root, dirs, files in os.walk(source):
            dcm_files = load_dicom_files_from_directory(root)
            if not dcm_files:
                continue
            print(f"[blue]Processing directory: {root}[/blue]")
            pid =validate_patientid(dcm_files)
            grouped = group_dcms_by_modality(dcm_files)
            ct_array, dose_array, structure_masks = construct_arrays(
                grouped,
                structure_dict=structure_dict if structures else None
            )
            save_array_to_h5(out_path, ct_array, dose_array, structure_masks, patient_id=pid)

def main():
    app()

if __name__ == "__main__":
    app()