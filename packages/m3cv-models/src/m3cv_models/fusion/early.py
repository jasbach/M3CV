"""Early fusion modules for combining tabular data with volumetric features."""

import torch
import torch.nn as nn

from .config import FusionPoint


class EarlyFusionDirect(nn.Module):
    """Direct early fusion: project tabular to spatial dims and merge.

    Projects tabular features through a linear layer, reshapes to match
    the spatial dimensions of the volumetric features, and combines them.
    """

    def __init__(
        self,
        tabular_dim: int,
        volume_channels: int,
        mode: str = "concat",
    ) -> None:
        """Initialize early fusion module.

        Args:
            tabular_dim: Input dimension of tabular features.
            volume_channels: Number of channels in the volumetric features.
            mode: Fusion mode - "concat" or "add".
        """
        super().__init__()
        self.mode = mode
        self.volume_channels = volume_channels

        if mode == "concat":
            # Project to volume_channels for concatenation
            self.project = nn.Linear(tabular_dim, volume_channels)
        else:  # add
            # Project to match volume channels exactly
            self.project = nn.Linear(tabular_dim, volume_channels)

    def forward(self, volume: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Fuse tabular features into volumetric feature map.

        Args:
            volume: Volumetric features (N, C, D, H, W)
            tabular: Tabular features (N, tabular_dim)

        Returns:
            Fused features. If mode="concat", shape is (N, C + volume_channels, D, H, W).
            If mode="add", shape is (N, C, D, H, W).
        """
        # Project tabular to channel dimensions
        projected = self.project(tabular)  # (N, volume_channels)

        # Reshape to match spatial dimensions
        # (N, C) -> (N, C, 1, 1, 1) then broadcast
        spatial = projected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        spatial = spatial.expand(-1, -1, volume.size(2), volume.size(3), volume.size(4))

        if self.mode == "concat":
            return torch.cat([volume, spatial], dim=1)
        else:  # add
            return volume + spatial


class EarlyFusionConv(nn.Module):
    """Early fusion with 1x1x1 conv after projection.

    Projects tabular features, broadcasts to spatial dims, applies a 1x1x1
    conv for learned combination, then merges with volumetric features.
    """

    def __init__(
        self,
        tabular_dim: int,
        volume_channels: int,
        mode: str = "concat",
    ) -> None:
        """Initialize early fusion module.

        Args:
            tabular_dim: Input dimension of tabular features.
            volume_channels: Number of channels in the volumetric features.
            mode: Fusion mode - "concat" or "add".
        """
        super().__init__()
        self.mode = mode
        self.volume_channels = volume_channels

        # Project tabular to intermediate representation
        self.project = nn.Linear(tabular_dim, volume_channels)

        # 1x1x1 conv to refine spatial representation
        self.conv = nn.Sequential(
            nn.Conv3d(volume_channels, volume_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(volume_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, volume: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Fuse tabular features into volumetric feature map.

        Args:
            volume: Volumetric features (N, C, D, H, W)
            tabular: Tabular features (N, tabular_dim)

        Returns:
            Fused features. If mode="concat", shape is (N, C + volume_channels, D, H, W).
            If mode="add", shape is (N, C, D, H, W).
        """
        # Project and reshape
        projected = self.project(tabular)  # (N, volume_channels)
        spatial = projected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        spatial = spatial.expand(-1, -1, volume.size(2), volume.size(3), volume.size(4))

        # Apply 1x1x1 conv
        spatial = self.conv(spatial)

        if self.mode == "concat":
            return torch.cat([volume, spatial], dim=1)
        else:  # add
            return volume + spatial


def create_early_fusion(
    fusion_point: FusionPoint,
    volume_channels: int,
) -> nn.Module:
    """Factory function to create an early fusion module.

    Args:
        fusion_point: Configuration for this fusion point.
        volume_channels: Number of channels in the volumetric features.

    Returns:
        Early fusion module.
    """
    if fusion_point.method == "direct":
        return EarlyFusionDirect(
            tabular_dim=fusion_point.tabular_dim,
            volume_channels=volume_channels,
            mode=fusion_point.mode,
        )
    else:  # conv
        return EarlyFusionConv(
            tabular_dim=fusion_point.tabular_dim,
            volume_channels=volume_channels,
            mode=fusion_point.mode,
        )
