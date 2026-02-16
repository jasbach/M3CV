"""Common layer building blocks for 3D neural networks."""

import torch.nn as nn


def conv3d_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int] = 3,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 1,
    bias: bool = False,
) -> nn.Sequential:
    """Create a Conv3D -> BatchNorm3D -> ReLU block."""
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


def bn_relu_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int] = 3,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 1,
    bias: bool = False,
) -> nn.Sequential:
    """Create a BatchNorm3D -> ReLU -> Conv3D block (pre-activation style)."""
    return nn.Sequential(
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
    )


def conv3d_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int] = 3,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 1,
    bias: bool = False,
) -> nn.Sequential:
    """Create a Conv3D -> BatchNorm3D block (no activation)."""
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm3d(out_channels),
    )
