# Non-Volumetric Data Logic Reference

This directory contains reference implementations for processing non-volumetric
patient data (clinical variables, surveys, RedCap exports).

**Status: Reference only - not for direct use**

These scripts are preserved as reference implementations showing patterns for:
- PII scrubbing (date anonymization, age capping)
- RedCap grouped field handling (checkbox columns)
- Survey time-binning relative to treatment dates
- Clinical data one-hot encoding

## Files

- `PatientDB.py` - RedCap CSV export processing with PII scrubbing
- `nondicomclasses.py` - Survey and clinical data classes

## Why Reference Only?

These implementations are bespoke to specific RedCap schemas and field names.
They demonstrate useful patterns but require adaptation for other datasets.

When ready to implement generic non-volumetric data attachment:
1. Review these patterns
2. Design a more flexible interface
3. Implement against that interface

## Do Not Import

Nothing in the main m3cv_prep package should import from this directory.
This code is excluded from linting and not part of the public API.
