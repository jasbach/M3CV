"""Compatibility layer for optional m3cv-dataprep imports."""

from __future__ import annotations

# SpatialMetadata is optional - only available if m3cv-dataprep is installed
try:
    from m3cv_prep.arrays import SpatialMetadata

    HAS_DATAPREP = True
except ImportError:
    HAS_DATAPREP = False
    SpatialMetadata = None  # type: ignore[assignment,misc]

__all__ = ["HAS_DATAPREP", "SpatialMetadata"]
