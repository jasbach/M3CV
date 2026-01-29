import os
from typing import Annotated

import typer
from rich import print

from m3cv_prep.array_tools import construct_arrays
from m3cv_prep.dicom_utils import (
    group_dcms_by_modality,
    validate_patientid,
)
from m3cv_prep.file_handling import load_dicom_files_from_directory, save_array_to_h5

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
        out_path = os.path.join(
            source, "packed_dicom.h5"
        )  # TODO - read patient ID from DICOMs for default filename

    if not recursive:
        dcm_files = load_dicom_files_from_directory(source)
        if not dcm_files:
            print("[yellow]No valid DICOM files found.[/yellow]")
            raise typer.Exit(code=1)
        validate_patientid(dcm_files)
        grouped = group_dcms_by_modality(dcm_files)
        ct_array, dose_array, structure_masks = construct_arrays(
            grouped, structure_names=structures.split(",") if structures else None
        )
        save_array_to_h5(out_path, ct_array, dose_array, structure_masks)
    else:
        for root, _dirs, _files in os.walk(source):
            dcm_files = load_dicom_files_from_directory(root)
            if not dcm_files:
                continue
            print(f"[blue]Processing directory: {root}[/blue]")
            validate_patientid(dcm_files)
            grouped = group_dcms_by_modality(dcm_files)
            ct_array, dose_array, structure_masks = construct_arrays(
                grouped, structure_names=structures.split(",") if structures else None
            )
            save_array_to_h5(out_path, ct_array, dose_array, structure_masks)


def main():
    app()


if __name__ == "__main__":
    app()
