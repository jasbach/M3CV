"""M3CV Models - Neural network architectures for medical imaging.

This package provides 3D neural network architectures designed for medical
imaging tasks, with built-in support for multimodal inputs combining
volumetric data (CT, dose, masks) with tabular clinical data.

Example:
    >>> from m3cv_models import ResNet3DBuilder, FusionConfig, FusionPoint
    >>>
    >>> # Basic ResNet-18
    >>> model = ResNet3DBuilder.build_resnet_18(in_channels=2, num_classes=2)
    >>>
    >>> # With late fusion
    >>> fusion = FusionConfig(late=FusionPoint(tabular_dim=10))
    >>> model = ResNet3DBuilder.build_resnet_18(
    ...     in_channels=2, num_classes=2, fusion_config=fusion
    ... )
"""

__version__ = "0.1.0"

from .exceptions import (
    DimensionMismatchError,
    FusionConfigError,
    InvalidBlockError,
    ModelError,
)
from .fusion import (
    EarlyFusionConv,
    EarlyFusionDirect,
    FusionConfig,
    FusionPoint,
    LateFusionModule,
)
from .resnet import BasicBlock3D, Bottleneck3D, ResNet3D, ResNet3DBuilder

__all__ = [
    # Exceptions
    "ModelError",
    "FusionConfigError",
    "InvalidBlockError",
    "DimensionMismatchError",
    # Fusion
    "FusionConfig",
    "FusionPoint",
    "EarlyFusionDirect",
    "EarlyFusionConv",
    "LateFusionModule",
    # ResNet
    "BasicBlock3D",
    "Bottleneck3D",
    "ResNet3D",
    "ResNet3DBuilder",
]
