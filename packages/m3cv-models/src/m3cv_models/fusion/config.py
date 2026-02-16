"""Fusion configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class FusionPoint:
    """Configuration for a single fusion point.

    Attributes:
        tabular_dim: Dimensionality of the tabular input features.
        mode: How to combine tabular and volumetric features.
            - "concat": Concatenate along channel dimension
            - "add": Add after projecting to same dimensions
        method: How to process tabular features before fusion.
            - "direct": Project directly to spatial dimensions
            - "conv": Apply 1x1x1 conv after projection
    """

    tabular_dim: int
    mode: Literal["concat", "add"] = "concat"
    method: Literal["direct", "conv"] = "conv"


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion in a model.

    Attributes:
        early: Dictionary mapping block indices to FusionPoint configs.
            Tabular data will be fused into the volumetric stream at
            the end of the specified block.
        late: FusionPoint config for late fusion, applied after global
            average pooling but before the classification head.
    """

    early: dict[int, FusionPoint] | None = field(default=None)
    late: FusionPoint | None = field(default=None)

    @classmethod
    def from_legacy_dict(cls, fusions: dict) -> "FusionConfig":
        """Create FusionConfig from legacy mixed-key dictionary format.

        Legacy format used integer keys for early fusion block indices
        and the string "late" for late fusion. Values were just the
        tabular dimension as an integer.

        Example legacy format:
            {0: 10, 2: 5, "late": 15}

        Args:
            fusions: Legacy fusion dictionary.

        Returns:
            FusionConfig with equivalent configuration.
        """
        early: dict[int, FusionPoint] = {}
        late: FusionPoint | None = None

        for key, value in fusions.items():
            if key == "late":
                if isinstance(value, int):
                    late = FusionPoint(tabular_dim=value)
                elif isinstance(value, FusionPoint):
                    late = value
                else:
                    raise TypeError(
                        f"Late fusion value must be int or FusionPoint, got {type(value)}"
                    )
            elif isinstance(key, int):
                if isinstance(value, int):
                    early[key] = FusionPoint(tabular_dim=value)
                elif isinstance(value, FusionPoint):
                    early[key] = value
                else:
                    raise TypeError(
                        f"Early fusion value must be int or FusionPoint, got {type(value)}"
                    )
            else:
                raise TypeError(f"Fusion key must be int or 'late', got {key!r}")

        return cls(early=early if early else None, late=late)

    def get_early_keys(self) -> list[str]:
        """Get the tabular input keys for early fusion points."""
        if not self.early:
            return []
        return [f"early_{i}" for i in sorted(self.early.keys())]

    def get_all_keys(self) -> list[str]:
        """Get all tabular input keys (early + late)."""
        keys = self.get_early_keys()
        if self.late:
            keys.append("late")
        return keys
