"""CLI module for m3cv-dataprep.

This module provides the command-line interface for data preparation tasks.

Commands:
    pack: Convert DICOM files to HDF5 format
    inspect: Summarize DICOM directory contents
"""

import typer

from m3cv_prep.cli.inspect import inspect
from m3cv_prep.cli.pack import pack

app = typer.Typer(
    help="Data Preparation CLI for M3CV",
    no_args_is_help=True,
)

app.command()(pack)
app.command()(inspect)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
