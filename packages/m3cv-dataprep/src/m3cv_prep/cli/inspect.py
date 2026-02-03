"""Inspect command for summarizing DICOM directory contents."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from m3cv_prep.cli._utils import scan_dicom_directory


def inspect(
    path: Annotated[
        str,
        typer.Argument(help="Directory path to inspect"),
    ],
    details: Annotated[
        bool,
        typer.Option(
            "--details",
            "-d",
            help="Show detailed info (structure names, dose types)",
        ),
    ] = False,
):
    """Inspect a directory tree for DICOM files.

    Scans the directory and its subdirectories for DICOM files, groups them
    by PatientID, and reports a summary of available data.
    """
    if not os.path.isdir(path):
        print(f"[red]Error: Directory not found: {path}[/red]")
        raise typer.Exit(code=1)

    print(f"[blue]Scanning: {path}[/blue]")
    print()

    patients = scan_dicom_directory(path, recursive=True)

    if not patients:
        print("[yellow]No DICOM files found.[/yellow]")
        raise typer.Exit(code=0)

    # Summary table
    console = Console()

    print(f"[green]Found {len(patients)} patient(s)[/green]")
    print()

    # Create summary table
    table = Table(title="Patient Summary")
    table.add_column("Patient ID", style="cyan")
    table.add_column("CT", justify="right")
    table.add_column("RTDOSE", justify="right")
    table.add_column("RTSTRUCT", justify="right")
    table.add_column("Other", justify="right")

    for patient_id in sorted(patients.keys()):
        summary = patients[patient_id]
        other_count = sum(summary.other_modalities.values())
        other_str = str(other_count) if other_count else "-"

        table.add_row(
            patient_id,
            str(summary.ct_count) if summary.ct_count else "-",
            str(summary.dose_count) if summary.dose_count else "-",
            str(summary.struct_count) if summary.struct_count else "-",
            other_str,
        )

    console.print(table)

    # Detailed output if requested
    if details:
        print()
        _print_details(patients)


def _print_details(patients: dict) -> None:
    """Print detailed information for each patient."""
    console = Console()

    for patient_id in sorted(patients.keys()):
        summary = patients[patient_id]

        print(f"[bold cyan]Patient: {patient_id}[/bold cyan]")

        # RTDOSE details
        if summary.dose_files:
            print("  [yellow]RTDOSE files:[/yellow]")
            for dose in summary.dose_files:
                dose_type = dose["type"]
                print(f"    - {dose_type}")

        # RTSTRUCT details
        if summary.struct_files:
            print("  [yellow]RTSTRUCT files:[/yellow]")
            for struct in summary.struct_files:
                struct_count = len(struct["structures"])
                print(f"    - {struct_count} structures:")

                # Create a table for structures
                if struct["structures"]:
                    struct_table = Table(show_header=False, box=None, padding=(0, 2))
                    struct_table.add_column()
                    struct_table.add_column()
                    struct_table.add_column()

                    # Display structures in 3 columns
                    structures = sorted(struct["structures"])
                    rows = (len(structures) + 2) // 3
                    for i in range(rows):
                        row = []
                        for j in range(3):
                            idx = i + j * rows
                            if idx < len(structures):
                                row.append(structures[idx])
                            else:
                                row.append("")
                        struct_table.add_row(*row)

                    console.print(struct_table)

        # Other modalities
        if summary.other_modalities:
            print("  [yellow]Other modalities:[/yellow]")
            for modality, count in sorted(summary.other_modalities.items()):
                print(f"    - {modality}: {count}")

        print()
