"""Factory methods for building ResNet3D models."""

from ..fusion.config import FusionConfig
from .blocks import BasicBlock3D, Bottleneck3D
from .resnet3d import ResNet3D


class ResNet3DBuilder:
    """Factory class for building standard ResNet3D architectures."""

    @staticmethod
    def build_resnet_18(
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        fusion_config: FusionConfig | None = None,
    ) -> ResNet3D:
        """Build a ResNet-18 (2 + 2 + 2 + 2 blocks).

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            base_filters: Base number of filters (default 64).
            fusion_config: Optional multimodal fusion configuration.

        Returns:
            ResNet3D model instance.
        """
        return ResNet3D(
            block=BasicBlock3D,
            layers=[2, 2, 2, 2],
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            fusion_config=fusion_config,
        )

    @staticmethod
    def build_resnet_34(
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        fusion_config: FusionConfig | None = None,
    ) -> ResNet3D:
        """Build a ResNet-34 (3 + 4 + 6 + 3 blocks).

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            base_filters: Base number of filters (default 64).
            fusion_config: Optional multimodal fusion configuration.

        Returns:
            ResNet3D model instance.
        """
        return ResNet3D(
            block=BasicBlock3D,
            layers=[3, 4, 6, 3],
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            fusion_config=fusion_config,
        )

    @staticmethod
    def build_resnet_50(
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        fusion_config: FusionConfig | None = None,
    ) -> ResNet3D:
        """Build a ResNet-50 (3 + 4 + 6 + 3 bottleneck blocks).

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            base_filters: Base number of filters (default 64).
            fusion_config: Optional multimodal fusion configuration.

        Returns:
            ResNet3D model instance.
        """
        return ResNet3D(
            block=Bottleneck3D,
            layers=[3, 4, 6, 3],
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            fusion_config=fusion_config,
        )

    @staticmethod
    def build_resnet_101(
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        fusion_config: FusionConfig | None = None,
    ) -> ResNet3D:
        """Build a ResNet-101 (3 + 4 + 23 + 3 bottleneck blocks).

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            base_filters: Base number of filters (default 64).
            fusion_config: Optional multimodal fusion configuration.

        Returns:
            ResNet3D model instance.
        """
        return ResNet3D(
            block=Bottleneck3D,
            layers=[3, 4, 23, 3],
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            fusion_config=fusion_config,
        )

    @staticmethod
    def build_resnet_152(
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        fusion_config: FusionConfig | None = None,
    ) -> ResNet3D:
        """Build a ResNet-152 (3 + 8 + 36 + 3 bottleneck blocks).

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            base_filters: Base number of filters (default 64).
            fusion_config: Optional multimodal fusion configuration.

        Returns:
            ResNet3D model instance.
        """
        return ResNet3D(
            block=Bottleneck3D,
            layers=[3, 8, 36, 3],
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=base_filters,
            fusion_config=fusion_config,
        )
