# CLI use
Eventually we'll be adding more functionality to the CLI. At this time, if you install the package, you can pack a file into an HDF5 file.

1. Ensure you have the `uv` package manager installed.
2. Clone this package repository.
3. From the project directory, run `uv run m3cv-prep [SOURCE_DIR] --out-path [OUTPUT_FILENAME]`

This will package all files in the SOURCE_DIR into an HDF5 file at OUTPUT_FILENAME.

We're going to get better structure set handling in, but for now it will handle dose and CT files.

You need to ensure you only have one patient's files in the SOURCE_DIR, as well as only one dose file.

It's a little fragile right now, I'll keep working on making it more durable.
