"""3D ResNet implementation with optional multimodal fusion."""

import torch
import torch.nn as nn

from ..exceptions import FusionConfigError, InvalidBlockError
from ..fusion.config import FusionConfig
from ..fusion.early import create_early_fusion
from ..fusion.late import LateFusionModule
from ..layers.common import conv3d_bn
from .blocks import BasicBlock3D, Bottleneck3D


class ResNet3D(nn.Module):
    """3D ResNet with optional multimodal tabular fusion.

    This implementation supports both early fusion (injecting tabular features
    into intermediate layers) and late fusion (combining tabular with pooled
    features before classification).

    Architecture:
        - Initial conv: 7x7x7 with stride 2
        - Max pool: 3x3x3 with stride 2
        - 4 residual stages (layers 1-4), each with multiple blocks
        - Global average pooling
        - Fully connected classification layer

    Early fusion points can be added after any of the 4 residual stages.
    Late fusion is applied after global pooling, before the FC layer.
    """

    def __init__(
        self,
        block: type[BasicBlock3D] | type[Bottleneck3D],
        layers: list[int],
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        """Initialize ResNet3D.

        Args:
            block: Block class to use (BasicBlock3D or Bottleneck3D).
            layers: Number of blocks in each of the 4 stages.
            in_channels: Number of input channels (e.g., 1 for CT, 2 for CT+dose).
            num_classes: Number of output classes.
            base_filters: Number of filters in the first conv layer.
            fusion_config: Optional configuration for multimodal fusion.
        """
        super().__init__()
        self.in_channels = base_filters
        self.fusion_config = fusion_config

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_filters,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_filters * 8, layers[3], stride=2)

        # Store layer output channels for fusion modules
        self._layer_channels = {
            0: base_filters * block.expansion,
            1: base_filters * 2 * block.expansion,
            2: base_filters * 4 * block.expansion,
            3: base_filters * 8 * block.expansion,
        }

        # Create early fusion modules and channel projections
        self.early_fusions: nn.ModuleDict = nn.ModuleDict()
        self.early_projections: nn.ModuleDict = nn.ModuleDict()

        if fusion_config and fusion_config.early:
            for block_idx, fusion_point in fusion_config.early.items():
                if block_idx not in range(4):
                    raise InvalidBlockError(
                        f"Early fusion block index must be 0-3, got {block_idx}"
                    )
                volume_channels = self._layer_channels[block_idx]
                self.early_fusions[f"early_{block_idx}"] = create_early_fusion(
                    fusion_point, volume_channels
                )
                # For concat mode, add a 1x1 conv to project back to expected channels
                if fusion_point.mode == "concat":
                    # After concat: volume_channels + volume_channels = 2 * volume_channels
                    # Need to project back to volume_channels for the next layer
                    self.early_projections[f"proj_{block_idx}"] = nn.Sequential(
                        nn.Conv3d(
                            volume_channels * 2,
                            volume_channels,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.BatchNorm3d(volume_channels),
                        nn.ReLU(inplace=True),
                    )

        # Calculate final feature dimension
        final_features = base_filters * 8 * block.expansion

        # Late fusion
        self.late_fusion: LateFusionModule | None = None
        if fusion_config and fusion_config.late:
            self.late_fusion = LateFusionModule(
                tabular_dim=fusion_config.late.tabular_dim,
                volume_features=final_features,
                mode=fusion_config.late.mode,
            )
            fc_input_dim = self.late_fusion.output_dim
        else:
            fc_input_dim = final_features

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(fc_input_dim, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(
        self,
        block: type[BasicBlock3D] | type[Bottleneck3D],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a residual stage with multiple blocks.

        Args:
            block: Block class to use.
            out_channels: Output channels for blocks in this stage.
            num_blocks: Number of blocks in this stage.
            stride: Stride for the first block (for downsampling).

        Returns:
            Sequential container of residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = conv3d_bn(
                self.in_channels,
                out_channels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        volume: torch.Tensor,
        tabular: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional tabular fusion.

        Args:
            volume: Input volume tensor (N, C, D, H, W).
            tabular: Dictionary of tabular tensors keyed by fusion point.
                - Early fusion keys: "early_0", "early_1", "early_2", "early_3"
                - Late fusion key: "late"

        Returns:
            Class logits (N, num_classes).
        """
        # Initial conv and pool
        x = self.conv1(volume)
        x = self.maxpool(x)

        # Residual stages with optional early fusion
        x = self.layer1(x)
        x = self._apply_early_fusion(x, 0, tabular)

        x = self.layer2(x)
        x = self._apply_early_fusion(x, 1, tabular)

        x = self.layer3(x)
        x = self._apply_early_fusion(x, 2, tabular)

        x = self.layer4(x)
        x = self._apply_early_fusion(x, 3, tabular)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Late fusion
        if self.late_fusion is not None:
            if tabular is None or "late" not in tabular:
                raise FusionConfigError(
                    "Late fusion configured but 'late' key not in tabular dict"
                )
            x = self.late_fusion(x, tabular["late"])

        # Classification
        x = self.fc(x)
        return x

    def _apply_early_fusion(
        self,
        x: torch.Tensor,
        block_idx: int,
        tabular: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Apply early fusion if configured for this block."""
        key = f"early_{block_idx}"
        if key in self.early_fusions:
            if tabular is None or key not in tabular:
                raise FusionConfigError(
                    f"Early fusion at block {block_idx} configured but "
                    f"'{key}' key not in tabular dict"
                )
            x = self.early_fusions[key](x, tabular[key])
            # Apply projection if this was a concat fusion
            proj_key = f"proj_{block_idx}"
            if proj_key in self.early_projections:
                x = self.early_projections[proj_key](x)
        return x
