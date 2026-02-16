"""Exceptions for transform operations."""


class TransformError(Exception):
    """Base exception for transform operations."""

    pass


class ReferenceNotFoundError(TransformError):
    """Raised when reference structure(s) not available for cropping.

    Attributes:
        patient_id: Identifier of the patient with missing structures.
        missing_structures: List of structure names that were not found.
        available_structures: List of structure names that are available.
        allow_fallback: Whether fallback to volume center is enabled.
    """

    def __init__(
        self,
        patient_id: str,
        missing_structures: list[str],
        available_structures: list[str],
        allow_fallback: bool = False,
    ) -> None:
        """Initialize ReferenceNotFoundError.

        Args:
            patient_id: Identifier of the patient.
            missing_structures: Structure names that were not found.
            available_structures: Structure names that are available.
            allow_fallback: Whether fallback is enabled.
        """
        self.patient_id = patient_id
        self.missing_structures = missing_structures
        self.available_structures = available_structures
        self.allow_fallback = allow_fallback

        msg = (
            f"Patient '{patient_id}': Required structures {missing_structures} "
            f"not found. Available structures: {available_structures}"
        )
        if not allow_fallback:
            msg += ". Set allow_fallback=True to use volume center as fallback."

        super().__init__(msg)
