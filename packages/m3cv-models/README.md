# m3cv-models

Neural network architectures for multimodal medical imaging research.

## Status

**Planned** - This package is not yet implemented. See below for planned features.

## Purpose

This package provides ready-to-use neural network architectures designed for medical imaging tasks, with built-in support for **multimodal inputs** combining:

- **Volumetric data**: 3D CT, dose distributions, structure masks
- **Tabular data**: Clinical variables, demographics, survey responses

## Planned Architectures

### ResNet3D

3D ResNet variants (18, 34, 50, 101, 152) for volumetric classification tasks.

- Based on the standard ResNet architecture adapted for 3D medical volumes
- Configurable depth and base filter count
- **Multimodal fusion support** (see below)

### Vision Transformer (ViT3D)

3D Vision Transformer for volumetric classification.

- Patch-based tokenization of 3D volumes
- Configurable transformer depth and attention heads
- **Multimodal fusion support** (see below)

### UNet

2D UNet for segmentation tasks.

- Encoder-decoder architecture with skip connections
- Dice loss and BCE-Dice loss functions included
- Useful for auto-contouring and structure segmentation

## Multimodal Fusion

A key feature of this package is the ability to combine volumetric imaging data with non-volumetric tabular data (clinical variables, demographics, etc.) in a single model. This is achieved through configurable **fusion points**:

### Early Fusion

Tabular data is processed through dense layers, reshaped and upsampled to match spatial dimensions, then merged (concatenate or add) with the volumetric feature maps at a specified network layer.

```python
# Example: Fuse clinical variables at ResNet block 2
model = Resnet3DBuilder.build_resnet_18(
    input_shape=(40, 128, 128, 3),
    num_outputs=1,
    fusions={2: 15}  # Fuse 15-dimensional tabular data at block 2
)
```

### Late Fusion

Tabular data is processed and concatenated with the flattened volumetric features just before the classification head.

```python
# Example: Late fusion of survey responses
model = Resnet3DBuilder.build_resnet_18(
    input_shape=(40, 128, 128, 3),
    num_outputs=1,
    fusions={"late": 20}  # Fuse 20-dimensional data after pooling
)
```

### Multiple Fusion Points

Both early and late fusion can be combined, and multiple early fusion points can be used:

```python
model = Resnet3DBuilder.build_resnet_34(
    input_shape=(40, 128, 128, 3),
    num_outputs=2,
    fusions={
        1: 10,       # Clinical variables at block 1
        3: 8,        # Imaging biomarkers at block 3
        "late": 15,  # Survey responses at late fusion
    }
)
```

## Integration with M3CV

This package is designed to work with:

- **m3cv-dataprep**: Produces HDF5 files with volumetric data
- **m3cv-data**: Loads and augments data, provides PyTorch/Keras data loaders

## Installation

```bash
uv sync
```

## Dependencies

- TensorFlow/Keras for model implementation
- NumPy for array operations

## Reference Implementation

The legacy implementation can be found in `_legacy/m3cv/models/` for reference during development.
