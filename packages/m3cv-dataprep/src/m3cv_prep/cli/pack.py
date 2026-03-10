"""Pack command for converting DICOM files to HDF5."""

from __future__ import annotations

import json
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


def _load_alias_file(alias_file: str) -> dict[str, list[str]]:
    """Load and validate a structure alias JSON file.

    Args:
        alias_file: Path to the JSON file.

    Returns:
        Validated alias map (canonical name -> list of alias strings).

    Raises:
        typer.BadParameter: If the file cannot be read, parsed, or has invalid schema.
    """
    try:
        with open(alias_file) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise typer.BadParameter(f"Alias file not found: {alias_file}") from e
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON in alias file: {e}") from e

    if not isinstance(data, dict):
        raise typer.BadParameter("Alias file must be a JSON object at the top level.")
    for key, value in data.items():
        if not isinstance(key, str):
            raise typer.BadParameter(f"Alias file keys must be strings, got: {key!r}")
        if not isinstance(value, list) or not value:
            raise typer.BadParameter(
                f"Alias file values must be non-empty lists of strings, "
                f"got {value!r} for key {key!r}"
            )
        for alias in value:
            if not isinstance(alias, str):
                raise typer.BadParameter(
                    f"All aliases must be strings, got {alias!r} in key {key!r}"
                )
    return data


def pack(
    source: Annotated[
        str,
        typer.Argument(help="Source directory containing DICOM files"),
    ] = None,
    out_path: Annotated[
        str,
        typer.Option("--out-path", "-o", help="Output HDF5 file or directory path"),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Process subdirectories recursively"),
    ] = False,
    structures: Annotated[
        str,
        typer.Option(
            "--structures",
            "-s",
            help="Comma-separated list of structure names to extract",
        ),
    ] = None,
    alias_file: Annotated[
        str,
        typer.Option(
            "--alias-file",
            "-a",
            help="JSON file mapping canonical structure names to DICOM aliases",
        ),
    ] = None,
):
    """Pack DICOM files into HDF5 format.

    Converts CT, RTDOSE, and RTSTRUCT files from a directory into a single
    HDF5 file suitable for machine learning pipelines.
    """
    if structures is not None and alias_file is not None:
        raise typer.BadParameter(
            "--structures and --alias-file are mutually exclusive. Use one or the other."
        )

    if source is None:
        source = os.getcwd()

    if not os.path.isdir(source):
        print(f"[red]Error: Source directory not found: {source}[/red]")
        raise typer.Exit(code=1)

    if not recursive:
        _pack_single_directory(source, out_path, structures, alias_file)
    else:
        _pack_recursive(source, out_path, structures, alias_file)


def _pack_single_directory(
    source: str,
    out_path: str | None,
    structures: str | None,
    alias_file: str | None,
) -> None:
    """Pack a single directory of DICOM files."""
    if out_path is None:
        out_path = os.path.join(source, "packed_dicom.h5")

    dcm_files = load_dicom_files_from_directory(source)
    if not dcm_files:
        print("[yellow]No valid DICOM files found.[/yellow]")
        raise typer.Exit(code=1)

    validate_patientid(dcm_files)
    grouped = group_dcms_by_modality(dcm_files)

    structure_list = structures.split(",") if structures else None
    aliases = _load_alias_file(alias_file) if alias_file else None
    ct_array, dose_array, structure_masks = construct_arrays(
        grouped, structure_names=structure_list, structure_aliases=aliases
    )

    save_array_to_h5(out_path, ct_array, dose_array, structure_masks)
    print(f"[green]Saved: {out_path}[/green]")


def _pack_recursive(
    source: str,
    out_path: str | None,
    structures: str | None,
    alias_file: str | None,
) -> None:
    """Pack multiple directories of DICOM files recursively."""
    if out_path is None:
        out_path = source
    os.makedirs(out_path, exist_ok=True)

    aliases = _load_alias_file(alias_file) if alias_file else None
    processed = 0
    for root, _dirs, _files in os.walk(source):
        dcm_files = load_dicom_files_from_directory(root)
        if not dcm_files:
            continue

        print(f"[blue]Processing directory: {root}[/blue]")
        try:
            validate_patientid(dcm_files)
            grouped = group_dcms_by_modality(dcm_files)

            structure_list = structures.split(",") if structures else None
            ct_array, dose_array, structure_masks = construct_arrays(
                grouped, structure_names=structure_list, structure_aliases=aliases
            )

            patient_id = ct_array.patient_id or os.path.basename(root)
            patient_out_path = os.path.join(out_path, f"{patient_id}.h5")
            save_array_to_h5(patient_out_path, ct_array, dose_array, structure_masks)
            print(f"[green]Saved: {patient_out_path}[/green]")
            processed += 1
        except Exception as e:
            print(f"[red]Error processing {root}: {e}[/red]")
            continue

    if processed == 0:
        print("[yellow]No valid DICOM directories found.[/yellow]")
        raise typer.Exit(code=1)

    print(f"[green]Processed {processed} patient(s).[/green]")
