"""Backwards-compatible shim for arrayclasses module.

.. deprecated::
    Import from m3cv_prep.arrays instead. This module will be removed in a future version.
"""

import warnings

warnings.warn(
    "m3cv_prep.arrayclasses is deprecated. Import from m3cv_prep.arrays instead.",
    DeprecationWarning,
    stacklevel=2,
)

from m3cv_prep.arrays import (  # noqa: E402
    PatientArray,
    PatientCT,
    PatientDose,
    PatientMask,
)

__all__ = ["PatientArray", "PatientCT", "PatientDose", "PatientMask"]
