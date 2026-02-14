"""Late fusion module for combining tabular data with pooled volumetric features."""

import torch
import torch.nn as nn


class LateFusionModule(nn.Module):
    """Late fusion: combine tabular with pooled volumetric features.

    Applied after global average pooling, before the final classification layer.
    Tabular features are projected and either concatenated or added to the
    pooled volumetric features.
    """

    def __init__(
        self,
        tabular_dim: int,
        volume_features: int,
        mode: str = "concat",
    ) -> None:
        """Initialize late fusion module.

        Args:
            tabular_dim: Input dimension of tabular features.
            volume_features: Number of features after global pooling.
            mode: Fusion mode - "concat" or "add".
        """
        super().__init__()
        self.mode = mode
        self.volume_features = volume_features

        if mode == "add":
            # Project to match pooled features exactly
            self.project = nn.Sequential(
                nn.Linear(tabular_dim, volume_features),
                nn.ReLU(inplace=True),
            )
        else:  # concat
            # Project to a reasonable representation for concatenation
            self.project = nn.Sequential(
                nn.Linear(tabular_dim, tabular_dim),
                nn.ReLU(inplace=True),
            )
            self._output_dim = volume_features + tabular_dim

    @property
    def output_dim(self) -> int:
        """Return output dimension after fusion."""
        if self.mode == "add":
            return self.volume_features
        return self._output_dim

    def forward(self, pooled: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Fuse tabular features with pooled volumetric features.

        Args:
            pooled: Pooled volumetric features (N, volume_features)
            tabular: Tabular features (N, tabular_dim)

        Returns:
            Fused features. If mode="concat", shape is (N, volume_features + tabular_dim).
            If mode="add", shape is (N, volume_features).
        """
        projected = self.project(tabular)

        if self.mode == "concat":
            return torch.cat([pooled, projected], dim=1)
        else:  # add
            return pooled + projected
