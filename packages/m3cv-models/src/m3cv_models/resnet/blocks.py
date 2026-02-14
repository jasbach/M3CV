"""3D ResNet residual blocks."""

import torch
import torch.nn as nn

from ..layers.common import conv3d_bn, conv3d_bn_relu


class BasicBlock3D(nn.Module):
    """Basic residual block for ResNet-18 and ResNet-34.

    Structure:
        x -> Conv3D(3x3x3) -> BN -> ReLU -> Conv3D(3x3x3) -> BN -> (+x) -> ReLU
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3d_bn_relu(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = conv3d_bn(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    """Bottleneck residual block for ResNet-50, -101, and -152.

    Structure:
        x -> Conv3D(1x1x1) -> BN -> ReLU
          -> Conv3D(3x3x3) -> BN -> ReLU
          -> Conv3D(1x1x1) -> BN -> (+x) -> ReLU

    The 1x1x1 convs reduce and then restore channels, with the 3x3x3 conv
    operating on the reduced channel count (bottleneck).
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # First 1x1 conv reduces channels
        self.conv1 = conv3d_bn_relu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        # 3x3 conv (bottleneck)
        self.conv2 = conv3d_bn_relu(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        # Second 1x1 conv restores channels (to out_channels * expansion)
        self.conv3 = conv3d_bn(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out
