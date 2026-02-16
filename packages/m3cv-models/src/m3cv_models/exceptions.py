"""Custom exceptions for m3cv-models."""


class ModelError(Exception):
    """Base exception for model-related errors."""


class FusionConfigError(ModelError):
    """Error in fusion configuration."""


class InvalidBlockError(ModelError):
    """Invalid block index for fusion point."""


class DimensionMismatchError(ModelError):
    """Tensor dimension mismatch during forward pass."""
