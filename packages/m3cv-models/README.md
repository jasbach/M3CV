# m3cv-models

**3D Neural Networks for Medical Imaging**

Neural network architectures optimized for medical imaging with built-in support for multimodal fusion.

**Status:** ✅ Production Ready

## Features

- ✅ **ResNet3D** - 3D ResNet implementations (18, 34, 50, 101, 152 layers)
- ✅ **Multimodal Fusion** - Early and late fusion of volumetric and tabular data
- ✅ **Builder API** - Factory methods for easy model construction
- ✅ **Medical Imaging Optimized** - Pre-configured for typical medical imaging dimensions

## Installation

```bash
# From repository root
make install

# Or with uv directly
cd packages/m3cv-models
uv sync
```

## Quick Start

### Basic ResNet3D

```python
from m3cv_models import ResNet3DBuilder

# Create ResNet-18 for binary classification
model = ResNet3DBuilder.build_resnet_18(
    in_channels=2,    # e.g., CT + dose
    num_classes=2,    # binary classification
)

# Forward pass
import torch
volume = torch.randn(4, 2, 90, 256, 256)  # (batch, channels, Z, Y, X)
output = model(volume)  # (4, 2)
```

### With Multimodal Fusion

Combine volumetric imaging with tabular clinical data:

```python
from m3cv_models import ResNet3DBuilder, FusionConfig, FusionPoint

# Configure late fusion (tabular features before final classification)
fusion_config = FusionConfig(
    late=FusionPoint(tabular_dim=10, mode="concat")
)

model = ResNet3DBuilder.build_resnet_50(
    in_channels=5,        # CT + dose + 3 structure masks
    num_classes=2,
    fusion_config=fusion_config,
)

# Forward pass with tabular data
volume = torch.randn(4, 5, 90, 256, 256)
tabular_features = torch.randn(4, 10)

output = model(volume, tabular={"late": tabular_features})
```

## Model Architectures

### ResNet3D Variants

| Model | Layers | Parameters | Block Type | Best For |
|-------|--------|------------|------------|----------|
| ResNet-18 | 18 | ~33M | BasicBlock | Fast prototyping, small datasets |
| ResNet-34 | 34 | ~63M | BasicBlock | Balance of speed and accuracy |
| ResNet-50 | 50 | ~68M | Bottleneck | Production models |
| ResNet-101 | 101 | ~86M | Bottleneck | High accuracy, large datasets |
| ResNet-152 | 152 | ~117M | Bottleneck | Maximum capacity |

All models use:
- 3D convolutions throughout
- Batch normalization after each conv layer
- ReLU activation
- Global average pooling before classification

### Builder API

```python
from m3cv_models import ResNet3DBuilder

# All standard architectures
model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
model = ResNet3DBuilder.build_resnet_34(in_channels=1, num_classes=2)
model = ResNet3DBuilder.build_resnet_50(in_channels=1, num_classes=2)
model = ResNet3DBuilder.build_resnet_101(in_channels=1, num_classes=2)
model = ResNet3DBuilder.build_resnet_152(in_channels=1, num_classes=2)

# With fusion configuration
from m3cv_models import FusionConfig

fusion = FusionConfig(
    early=None,  # No early fusion
    late=FusionPoint(tabular_dim=10, mode="concat")
)

model = ResNet3DBuilder.build_resnet_50(
    in_channels=1,
    num_classes=2,
    fusion_config=fusion,
)
```

## Multimodal Fusion

### Early Fusion

Concatenate tabular features with image features early in the network:

```python
from m3cv_models import FusionConfig, FusionPoint

# Early fusion after the first residual block
fusion = FusionConfig(
    early=FusionPoint(
        block_index=0,      # After which residual block
        tabular_dim=10,     # Number of tabular features
    )
)

model = ResNet3DBuilder.build_resnet_50(
    in_channels=1,
    num_classes=2,
    fusion_config=fusion,
)

# Forward pass
volume = torch.randn(4, 1, 90, 256, 256)
tabular = torch.randn(4, 10)

output = model(volume, tabular={"early": tabular})
```

**How it works:**
1. Image passes through initial conv and first residual block
2. Tabular features are broadcast and concatenated channelwise
3. Combined features continue through remaining blocks

### Late Fusion

Combine features before the final classification layer:

```python
fusion = FusionConfig(
    late=FusionPoint(
        tabular_dim=10,
        mode="concat",  # or "add"
    )
)

model = ResNet3DBuilder.build_resnet_50(
    in_channels=1,
    num_classes=2,
    fusion_config=fusion,
)

# Forward pass
volume = torch.randn(4, 1, 90, 256, 256)
tabular = torch.randn(4, 10)

output = model(volume, tabular={"late": tabular})
```

**Fusion modes:**
- `"concat"` - Concatenate tabular features with image features
- `"add"` - Element-wise addition (requires matching dimensions)

### Combined Fusion

Use both early and late fusion:

```python
fusion = FusionConfig(
    early=FusionPoint(block_index=0, tabular_dim=10),
    late=FusionPoint(tabular_dim=5, mode="concat"),
)

model = ResNet3DBuilder.build_resnet_50(
    in_channels=1,
    num_classes=2,
    fusion_config=fusion,
)

# Forward pass with different tabular features at each fusion point
volume = torch.randn(4, 1, 90, 256, 256)

output = model(
    volume,
    tabular={
        "early": torch.randn(4, 10),  # Clinical variables
        "late": torch.randn(4, 5),     # Derived features
    }
)
```

## Integration with m3cv-data

Complete example using data from m3cv-data:

```python
from torch.utils.data import DataLoader
from m3cv_data import PatientDataset
from m3cv_data.transforms import AnatomicalCrop, BilateralStructureMidpoint
from m3cv_models import ResNet3DBuilder

# Create dataset with anatomical cropping
strategy = BilateralStructureMidpoint("Parotid_L", "Parotid_R")
crop = AnatomicalCrop(crop_shape=(90, 256, 256), reference_strategy=strategy)

dataset = PatientDataset(
    paths="./data/packed/",
    channels=["Parotid_L", "Parotid_R", "GTV"],
    include_ct=True,
    include_dose=True,
    patient_transform=crop,
    label_fn=lambda p: get_label(p),
)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create model
model = ResNet3DBuilder.build_resnet_50(
    in_channels=5,  # CT + dose + 3 structures
    num_classes=2,
)

# Training loop
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
for volumes, labels in train_loader:
    volumes = volumes.to(device)
    labels = labels.to(device)

    outputs = model(volumes)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Training Examples

See the [`examples/`](examples/) directory for complete training scripts:

```bash
# Demo training with head-and-neck data
uv run python examples/demo_training.py --data ./data/packed/

# With hyperparameter configuration
uv run python examples/demo_training.py \
    --data ./data/packed/ \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --epochs 50
```

## Model Configuration

### Input Dimensions

ResNet3D models are flexible with input dimensions but are optimized for:

```python
# Typical medical imaging dimensions
volume.shape = (batch_size, channels, depth, height, width)

# Head-and-neck radiotherapy (typical)
volume.shape = (4, 5, 90, 256, 256)

# Smaller volumes (e.g., single organ)
volume.shape = (8, 3, 64, 128, 128)

# Full CT scans
volume.shape = (2, 2, 200, 512, 512)
```

**Note:** Deeper networks (ResNet-50+) may require larger volumes or lower batch sizes due to memory constraints.

### Memory Usage

Approximate GPU memory requirements (batch_size=4, input=(5, 90, 256, 256)):

| Model | Training Memory | Inference Memory |
|-------|----------------|------------------|
| ResNet-18 | ~6 GB | ~2 GB |
| ResNet-34 | ~8 GB | ~3 GB |
| ResNet-50 | ~10 GB | ~4 GB |
| ResNet-101 | ~14 GB | ~5 GB |
| ResNet-152 | ~18 GB | ~6 GB |

**Tips for reducing memory:**
- Use smaller batch sizes
- Use gradient checkpointing (not currently implemented)
- Use mixed precision training (FP16)
- Reduce input dimensions

## Advanced Usage

### Custom Number of Classes

```python
# Multi-class classification
model = ResNet3DBuilder.build_resnet_50(
    in_channels=5,
    num_classes=4,  # 4-class problem
)

# Regression (single output)
model = ResNet3DBuilder.build_resnet_18(
    in_channels=1,
    num_classes=1,  # Regression task
)
```

### Feature Extraction

Extract features before classification:

```python
model = ResNet3DBuilder.build_resnet_50(in_channels=1, num_classes=2)

# Remove classification head
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# Extract features
with torch.no_grad():
    features = feature_extractor(volume)  # (batch, 2048, 1, 1, 1)
    features = features.squeeze()         # (batch, 2048)
```

### Transfer Learning

```python
# Load pre-trained model (if available)
model = ResNet3DBuilder.build_resnet_50(in_channels=1, num_classes=2)
model.load_state_dict(torch.load("pretrained_resnet50.pth"))

# Freeze early layers
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False

# Fine-tune later layers
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)
```

## Testing

```bash
# Run tests
uv run pytest src/tests/ -v

# Test coverage includes:
# - Model construction (26 tests)
# - Forward pass shapes
# - Fusion configurations
# - Gradient flow
```

## Model Export

### ONNX Export

```python
import torch.onnx

model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
model.eval()

dummy_input = torch.randn(1, 1, 90, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "resnet3d_18.onnx",
    input_names=["volume"],
    output_names=["logits"],
    dynamic_axes={
        "volume": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
)
```

### TorchScript Export

```python
model = ResNet3DBuilder.build_resnet_18(in_channels=1, num_classes=2)
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save("resnet3d_18.pt")

# Load later
loaded_model = torch.jit.load("resnet3d_18.pt")
```

## Performance Optimization

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

model = ResNet3DBuilder.build_resnet_50(in_channels=1, num_classes=2)
model = model.to(device)

scaler = GradScaler()

for volumes, labels in train_loader:
    volumes = volumes.to(device)
    labels = labels.to(device)

    with autocast():
        outputs = model(volumes)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Multi-GPU Training

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=2)  # Instead of 4

# Reduce input dimensions
crop = AnatomicalCrop(crop_shape=(64, 128, 128))  # Smaller crop

# Use smaller model
model = ResNet3DBuilder.build_resnet_18(...)  # Instead of ResNet-50
```

### Gradient Explosion/Vanishing

```python
# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Use learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5
)
```

## Package Structure

```
m3cv_models/
├── resnet/
│   ├── resnet3d.py       # Main ResNet3D model
│   ├── blocks.py         # BasicBlock3D, Bottleneck3D
│   └── builder.py        # ResNet3DBuilder factory
├── fusion/
│   ├── config.py         # FusionConfig, FusionPoint
│   ├── early.py          # EarlyFusion module
│   └── late.py           # LateFusion module
├── layers/
│   └── common.py         # Shared layer utilities
└── exceptions.py         # Custom exceptions
```

## Roadmap

Future enhancements:

- [ ] Pre-trained weights on medical imaging datasets
- [ ] Additional architectures (UNet3D, ViT3D, DenseNet3D)
- [ ] Attention mechanisms
- [ ] Gradient checkpointing for memory efficiency
- [ ] Model ensemble utilities
- [ ] Interpretability tools (GradCAM, attention maps)

## Contributing

Contributions welcome! Areas for improvement:
- New architecture implementations
- Pre-training scripts
- Optimization techniques
- Documentation improvements

## Citation

If you use these models in your research, please cite the published work:

> Asbach JC, Singh AK, Iovoli AJ, Farrugia M, Le AH. Novel pre-spatial data fusion deep learning approach for multimodal volumetric outcome prediction models in radiotherapy. *Med Phys*. 2025; 52: 2675–2687. https://doi.org/10.1002/mp.17672

```bibtex
@article{asbach2025prespatial,
  title={Novel pre-spatial data fusion deep learning approach for multimodal volumetric outcome prediction models in radiotherapy},
  author={Asbach, J. C. and Singh, A. K. and Iovoli, A. J. and Farrugia, M. and Le, A. H.},
  journal={Medical Physics},
  year={2025},
  volume={52},
  pages={2675--2687},
  doi={10.1002/mp.17672}
}
```

Software citation:

```bibtex
@software{m3cv_models2026,
  title={M3CV Models: 3D Neural Networks for Medical Imaging},
  author={Asbach, J. C.},
  year={2026},
  url={https://github.com/jasbach/M3CV}
}
```

## License

See [LICENSE](../../LICENSE) for details.
