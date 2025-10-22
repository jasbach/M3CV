import sys
import os
import typer
from typing_extensions import Annotated
from rich import print
from rich.progress import track

import pydicom
from m3cv_prep.file_handling import load_dicom_files_from_directory

app = typer.Typer(help="Data Preparation CLI for M3CV")

@app.command()
def pack(
    source: Annotated[str, typer.Argument()] = None,
    out_path: Annotated[str, typer.Option()] = None,
    recursive: Annotated[bool, typer.Option()] = False,
):
    if source is None:
        source = os.getcwd()
    
    if out_path is None:
        out_path = os.path.join(source, "packed_dicom.h5") # TODO - read patient ID from DICOMs for default filename

    if not recursive:
        dcm_files = load_dicom_files_from_directory(source)
        if not dcm_files:
            print("[yellow]No valid DICOM files found.[/yellow]")
            raise typer.Exit(code=1)
        validate_dcm_files(dcm_files) # TODO - implement validation function
        array = placeholder(dcm_files) # TODO - implement array constructor
        save_array_to_h5(array, out_path) # TODO - implement save function
    else:
        for root, dirs, files in os.walk(source):
            dcm_files = load_dicom_files_from_directory(root)
            if not dcm_files:
                continue
            print(f"[blue]Processing directory: {root}[/blue]")
            validate_dcm_files(dcm_files) # TODO - implement validation function
            array = placeholder(dcm_files) # TODO - implement array constructor
            save_array_to_h5(array, out_path) # TODO - implement save function

if __name__ == "__main__":
    app()