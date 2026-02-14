"""Tests for ResNet3D model."""

import pytest
import torch
import torch.nn as nn

from m3cv_models import (
    BasicBlock3D,
    Bottleneck3D,
    FusionConfig,
    FusionPoint,
    ResNet3D,
    ResNet3DBuilder,
)
from m3cv_models.exceptions import FusionConfigError, InvalidBlockError


class TestBasicBlock3D:
    """Tests for BasicBlock3D."""

    def test_forward_same_channels(self):
        """Test forward pass when in_channels == out_channels."""
        block = BasicBlock3D(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 8, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_forward_with_downsample(self):
        """Test forward pass with downsampling."""
        downsample = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(64),
        )
        block = BasicBlock3D(
            in_channels=32, out_channels=64, stride=2, downsample=downsample
        )
        x = torch.randn(2, 32, 8, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 4, 8, 8)

    def test_expansion(self):
        """Test expansion factor is 1."""
        assert BasicBlock3D.expansion == 1


class TestBottleneck3D:
    """Tests for Bottleneck3D."""

    def test_forward_with_expansion(self):
        """Test forward pass with channel expansion."""
        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(256),
        )
        block = Bottleneck3D(in_channels=64, out_channels=64, downsample=downsample)
        x = torch.randn(2, 64, 8, 16, 16)
        out = block(x)
        # Output channels = out_channels * expansion = 64 * 4 = 256
        assert out.shape == (2, 256, 8, 16, 16)

    def test_expansion(self):
        """Test expansion factor is 4."""
        assert Bottleneck3D.expansion == 4


class TestResNet3D:
    """Tests for ResNet3D model."""

    def test_basic_forward(self, sample_volume):
        """Test basic forward pass without fusion."""
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
        )
        out = model(sample_volume)
        assert out.shape == (2, 2)

    def test_bottleneck_forward(self, sample_volume):
        """Test forward pass with bottleneck blocks."""
        model = ResNet3D(
            block=Bottleneck3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=3,
            base_filters=16,
        )
        out = model(sample_volume)
        assert out.shape == (2, 3)

    def test_late_fusion_concat(self, sample_volume, sample_tabular):
        """Test late fusion with concatenation."""
        fusion = FusionConfig(late=FusionPoint(tabular_dim=10, mode="concat"))
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        out = model(sample_volume, tabular={"late": sample_tabular})
        assert out.shape == (2, 2)

    def test_late_fusion_add(self, sample_volume, sample_tabular):
        """Test late fusion with addition."""
        fusion = FusionConfig(late=FusionPoint(tabular_dim=10, mode="add"))
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        out = model(sample_volume, tabular={"late": sample_tabular})
        assert out.shape == (2, 2)

    def test_early_fusion(self, sample_volume, sample_tabular):
        """Test early fusion at block 0."""
        fusion = FusionConfig(early={0: FusionPoint(tabular_dim=10)})
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        out = model(sample_volume, tabular={"early_0": sample_tabular})
        assert out.shape == (2, 2)

    def test_combined_fusion(self, sample_volume):
        """Test both early and late fusion."""
        fusion = FusionConfig(
            early={1: FusionPoint(tabular_dim=5)},
            late=FusionPoint(tabular_dim=8),
        )
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        tabular = {
            "early_1": torch.randn(2, 5),
            "late": torch.randn(2, 8),
        }
        out = model(sample_volume, tabular=tabular)
        assert out.shape == (2, 2)

    def test_missing_late_tabular_raises(self, sample_volume):
        """Test that missing late tabular data raises error."""
        fusion = FusionConfig(late=FusionPoint(tabular_dim=10))
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        with pytest.raises(FusionConfigError):
            model(sample_volume)

    def test_missing_early_tabular_raises(self, sample_volume):
        """Test that missing early tabular data raises error."""
        fusion = FusionConfig(early={0: FusionPoint(tabular_dim=10)})
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        with pytest.raises(FusionConfigError):
            model(sample_volume)

    def test_invalid_block_index_raises(self):
        """Test that invalid block index raises error."""
        with pytest.raises(InvalidBlockError):
            fusion = FusionConfig(early={5: FusionPoint(tabular_dim=10)})
            ResNet3D(
                block=BasicBlock3D,
                layers=[1, 1, 1, 1],
                in_channels=2,
                num_classes=2,
                fusion_config=fusion,
            )

    def test_gradients_flow(self, sample_volume, sample_labels):
        """Test that gradients flow through the model."""
        model = ResNet3D(
            block=BasicBlock3D,
            layers=[1, 1, 1, 1],
            in_channels=2,
            num_classes=2,
            base_filters=16,
        )
        criterion = nn.CrossEntropyLoss()

        out = model(sample_volume)
        loss = criterion(out, sample_labels)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestResNet3DBuilder:
    """Tests for ResNet3DBuilder factory methods."""

    def test_build_resnet_18(self, sample_volume_single_channel):
        """Test ResNet-18 construction."""
        model = ResNet3DBuilder.build_resnet_18(
            in_channels=1, num_classes=2, base_filters=16
        )
        out = model(sample_volume_single_channel)
        assert out.shape == (2, 2)

    def test_build_resnet_34(self, sample_volume_single_channel):
        """Test ResNet-34 construction."""
        model = ResNet3DBuilder.build_resnet_34(
            in_channels=1, num_classes=3, base_filters=8
        )
        out = model(sample_volume_single_channel)
        assert out.shape == (2, 3)

    def test_build_resnet_50(self, sample_volume_single_channel):
        """Test ResNet-50 construction."""
        model = ResNet3DBuilder.build_resnet_50(
            in_channels=1, num_classes=2, base_filters=8
        )
        out = model(sample_volume_single_channel)
        assert out.shape == (2, 2)

    def test_build_resnet_101(self, sample_volume_single_channel):
        """Test ResNet-101 construction."""
        model = ResNet3DBuilder.build_resnet_101(
            in_channels=1, num_classes=2, base_filters=8
        )
        out = model(sample_volume_single_channel)
        assert out.shape == (2, 2)

    def test_build_resnet_152(self, sample_volume_single_channel):
        """Test ResNet-152 construction."""
        model = ResNet3DBuilder.build_resnet_152(
            in_channels=1, num_classes=2, base_filters=8
        )
        out = model(sample_volume_single_channel)
        assert out.shape == (2, 2)

    def test_with_fusion_config(self, sample_volume_single_channel, sample_tabular):
        """Test builder with fusion config."""
        fusion = FusionConfig(late=FusionPoint(tabular_dim=10))
        model = ResNet3DBuilder.build_resnet_18(
            in_channels=1,
            num_classes=2,
            base_filters=16,
            fusion_config=fusion,
        )
        out = model(sample_volume_single_channel, tabular={"late": sample_tabular})
        assert out.shape == (2, 2)


class TestFusionConfig:
    """Tests for FusionConfig."""

    def test_from_legacy_dict_late_only(self):
        """Test legacy format with late fusion only."""
        legacy = {"late": 15}
        config = FusionConfig.from_legacy_dict(legacy)
        assert config.late is not None
        assert config.late.tabular_dim == 15
        assert config.early is None

    def test_from_legacy_dict_early_only(self):
        """Test legacy format with early fusion only."""
        legacy = {0: 10, 2: 5}
        config = FusionConfig.from_legacy_dict(legacy)
        assert config.early is not None
        assert 0 in config.early
        assert config.early[0].tabular_dim == 10
        assert 2 in config.early
        assert config.early[2].tabular_dim == 5
        assert config.late is None

    def test_from_legacy_dict_combined(self):
        """Test legacy format with both early and late."""
        legacy = {1: 8, "late": 12}
        config = FusionConfig.from_legacy_dict(legacy)
        assert config.early is not None
        assert config.early[1].tabular_dim == 8
        assert config.late is not None
        assert config.late.tabular_dim == 12

    def test_get_early_keys(self):
        """Test get_early_keys method."""
        config = FusionConfig(early={2: FusionPoint(5), 0: FusionPoint(10)})
        keys = config.get_early_keys()
        assert keys == ["early_0", "early_2"]

    def test_get_all_keys(self):
        """Test get_all_keys method."""
        config = FusionConfig(
            early={1: FusionPoint(5)},
            late=FusionPoint(10),
        )
        keys = config.get_all_keys()
        assert keys == ["early_1", "late"]
