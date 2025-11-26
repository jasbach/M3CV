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
        validate_patientid(dcm_files)
        grouped = group_dcms_by_modality(dcm_files)
        ct_array, dose_array, structure_masks = construct_arrays(
            grouped,
            structure_dict=structure_dict if structures else None
        )
        save_array_to_h5(out_path, ct_array, dose_array, structure_masks)
    else:
        for root, dirs, files in os.walk(source):
            dcm_files = load_dicom_files_from_directory(root)
            if not dcm_files:
                continue
            print(f"[blue]Processing directory: {root}[/blue]")
            validate_patientid(dcm_files)
            grouped = group_dcms_by_modality(dcm_files)
            ct_array, dose_array, structure_masks = construct_arrays(
                grouped,
                structure_dict=structure_dict if structures else None
            )
            save_array_to_h5(out_path, ct_array, dose_array, structure_masks)

def main():
    app()

if __name__ == "__main__":
    app()