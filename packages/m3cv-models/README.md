# m3cv-models

Neural network architectures for medical imaging with multimodal fusion support.

## Features

- **ResNet3D**: 3D ResNet implementations (18, 34, 50, 101, 152 layers)
- **Multimodal Fusion**: Early and late fusion of volumetric and tabular data
- **Builder API**: Factory methods for easy model construction

## Installation

```bash
uv sync
```

## Usage

```python
from m3cv_models import ResNet3DBuilder

# Basic ResNet-18 for 3D volumes
model = ResNet3DBuilder.build_resnet_18(
    in_channels=2,  # e.g., CT + dose
    num_classes=2,  # binary classification
)

# With tabular fusion
from m3cv_models import FusionConfig, FusionPoint

fusion = FusionConfig(
    late=FusionPoint(tabular_dim=10, mode="concat")
)
model = ResNet3DBuilder.build_resnet_18(
    in_channels=2,
    num_classes=2,
    fusion_config=fusion,
)

# Forward pass
output = model(volume, tabular={"late": tabular_features})
```

## Demo

Run the demo training script with HDF5 data from m3cv-dataprep:

```bash
uv run python packages/m3cv-models/examples/demo_training.py --data path/to/packed.h5
```

## Integration with M3CV

This package is designed to work with:

- **m3cv-dataprep**: Produces HDF5 files with volumetric data
- **m3cv-data**: Loads and augments data, provides PyTorch data loaders
