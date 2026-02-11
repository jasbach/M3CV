"""HDF5 inspection utilities for exploring patient data files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path


@dataclass
class H5FileInfo:
    """Summary information about an HDF5 patient data file.

    Attributes:
        path: Path to the HDF5 file.
        patient_id: Patient identifier if available.
        has_ct: Whether the file contains a CT array.
        has_dose: Whether the file contains a dose array.
        structure_names: List of available structure/ROI names.
        ct_shape: Shape of the CT array if available (Z, Y, X).
        file_size_mb: File size in megabytes.
    """

    path: str
    patient_id: str | None
    has_ct: bool
    has_dose: bool
    structure_names: list[str]
    ct_shape: tuple[int, int, int] | None
    file_size_mb: float

    def __repr__(self) -> str:
        struct_str = ", ".join(self.structure_names) if self.structure_names else "none"
        return (
            f"H5FileInfo({self.patient_id or 'unknown'}, "
            f"ct={self.has_ct}, dose={self.has_dose}, "
            f"structures=[{struct_str}])"
        )


def inspect_h5(path: str) -> H5FileInfo:
    """Inspect an HDF5 file without loading the full arrays.

    This function reads only metadata and dataset shapes, making it fast
    for exploring large collections of files.

    Args:
        path: Path to the HDF5 file.

    Returns:
        H5FileInfo with summary information about the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import h5py

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    with h5py.File(path, "r") as f:
        # Patient ID
        patient_id = str(f.attrs["patient_id"]) if "patient_id" in f.attrs else None

        # CT info
        has_ct = "ct" in f
        ct_shape = None
        if has_ct:
            ct_shape = tuple(f["ct"].shape)  # type: ignore[arg-type]

        # Dose info
        has_dose = "dose" in f

        # Structure names
        structure_names: list[str] = []
        if "structures" in f:
            structure_names = list(f["structures"].keys())

    return H5FileInfo(
        path=path,
        patient_id=patient_id,
        has_ct=has_ct,
        has_dose=has_dose,
        structure_names=structure_names,
        ct_shape=ct_shape,
        file_size_mb=round(file_size_mb, 2),
    )


def inspect_directory(
    path: str,
    pattern: str = "*.h5",
    recursive: bool = False,
) -> list[H5FileInfo]:
    """Inspect all HDF5 files in a directory.

    Args:
        path: Path to the directory to search.
        pattern: Glob pattern for matching files (default: "*.h5").
        recursive: Whether to search recursively in subdirectories.

    Returns:
        List of H5FileInfo objects for each matching file.

    Raises:
        NotADirectoryError: If path is not a directory.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Not a directory: {path}")

    if recursive:
        search_pattern = os.path.join(path, "**", pattern)
        files = glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(path, pattern)
        files = glob(search_pattern)

    # Sort by filename for consistent ordering
    files = sorted(files, key=lambda p: Path(p).name)

    infos = []
    for file_path in files:
        try:
            infos.append(inspect_h5(file_path))
        except Exception:
            # Skip files that can't be read as HDF5
            continue

    return infos


def summary_table(infos: list[H5FileInfo]) -> None:
    """Print a summary table of HDF5 file information.

    Uses Rich for formatted output if available, otherwise falls back to
    plain text.

    Args:
        infos: List of H5FileInfo objects to display.
    """
    if not infos:
        print("No files to display.")
        return

    try:
        import rich  # noqa: F401

        _print_rich_table(infos)
    except ImportError:
        _print_plain_table(infos)


def _print_rich_table(infos: list[H5FileInfo]) -> None:
    """Print table using Rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="HDF5 Patient Files")

    table.add_column("Patient ID", style="cyan")
    table.add_column("CT Shape", style="green")
    table.add_column("Dose", style="yellow")
    table.add_column("Structures", style="magenta")
    table.add_column("Size (MB)", style="blue", justify="right")
    table.add_column("File", style="dim")

    for info in infos:
        shape_str = f"{info.ct_shape}" if info.ct_shape else "-"
        dose_str = "Yes" if info.has_dose else "No"
        struct_str = ", ".join(info.structure_names) if info.structure_names else "-"
        filename = Path(info.path).name

        table.add_row(
            info.patient_id or "unknown",
            shape_str,
            dose_str,
            struct_str,
            f"{info.file_size_mb:.2f}",
            filename,
        )

    console.print(table)

    # Summary stats
    total_size = sum(i.file_size_mb for i in infos)
    with_dose = sum(1 for i in infos if i.has_dose)
    console.print(f"\n[bold]Total:[/bold] {len(infos)} files, {total_size:.1f} MB")
    console.print(f"[bold]With dose:[/bold] {with_dose}/{len(infos)}")


def _print_plain_table(infos: list[H5FileInfo]) -> None:
    """Print table using plain text formatting."""
    # Header
    header = f"{'Patient ID':<15} {'CT Shape':<18} {'Dose':<6} {'Structures':<30}"
    print(f"{header} {'MB':>8}")
    print("-" * 80)

    for info in infos:
        shape_str = f"{info.ct_shape}" if info.ct_shape else "-"
        dose_str = "Yes" if info.has_dose else "No"
        if info.structure_names:
            struct_str = ", ".join(info.structure_names[:3])
        else:
            struct_str = "-"
        if len(info.structure_names) > 3:
            struct_str += f" (+{len(info.structure_names) - 3})"

        print(
            f"{(info.patient_id or 'unknown'):<15} "
            f"{shape_str:<18} "
            f"{dose_str:<6} "
            f"{struct_str:<30} "
            f"{info.file_size_mb:>8.2f}"
        )

    # Summary
    total_size = sum(i.file_size_mb for i in infos)
    with_dose = sum(1 for i in infos if i.has_dose)
    print("-" * 80)
    print(f"Total: {len(infos)} files, {total_size:.1f} MB")
    print(f"With dose: {with_dose}/{len(infos)}")
