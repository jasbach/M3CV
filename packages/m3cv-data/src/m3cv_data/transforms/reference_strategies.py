"""Reference strategies for calculating anatomical reference points."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from m3cv_data.structures import Patient


class ReferenceStrategy(ABC):
    """Abstract base class for reference point calculation strategies.

    Reference strategies calculate a voxel coordinate (Z, Y, X) from a Patient
    object, typically using center-of-mass of structure masks as anatomical
    reference points.
    """

    @abstractmethod
    def calculate(self, patient: "Patient") -> tuple[int, int, int] | None:
        """Calculate reference point from patient data.

        Args:
            patient: Patient object with CT, dose, and structure data.

        Returns:
            Voxel indices (Z, Y, X) as reference point, or None if calculation
            cannot be performed (e.g., required structures missing).
        """
        pass

    @abstractmethod
    def get_required_structures(self) -> list[str]:
        """Get list of structure names required by this strategy.

        Returns:
            List of structure names needed for reference calculation.
        """
        pass


class SingleStructureCOM(ReferenceStrategy):
    """Calculate center-of-mass of a single structure.

    Uses the center-of-mass (COM) of a specified structure mask as the
    reference point. Returns None if the structure is not present or is empty.
    """

    def __init__(self, structure_name: str) -> None:
        """Initialize single structure COM strategy.

        Args:
            structure_name: Name of the structure to use for COM calculation.
        """
        self._structure_name = structure_name

    def calculate(self, patient: "Patient") -> tuple[int, int, int] | None:
        """Calculate COM of the specified structure.

        Args:
            patient: Patient object with structure data.

        Returns:
            Voxel indices (Z, Y, X) of structure COM, or None if structure
            is missing or empty.
        """
        if self._structure_name not in patient.structures:
            return None

        mask = patient.structures[self._structure_name]
        nonzero_indices = np.argwhere(mask > 0)

        if len(nonzero_indices) == 0:
            return None

        # Calculate mean position of all nonzero voxels
        com = nonzero_indices.mean(axis=0)
        return int(com[0]), int(com[1]), int(com[2])

    def get_required_structures(self) -> list[str]:
        """Get required structure name.

        Returns:
            List containing the single required structure name.
        """
        return [self._structure_name]


class BilateralStructureMidpoint(ReferenceStrategy):
    """Calculate midpoint between left and right bilateral structures.

    Calculates the COM of each bilateral structure, then returns the midpoint
    between them. This is useful for anatomical references like bilateral
    parotid glands. Returns None if either structure is missing or empty.
    """

    def __init__(self, left_structure: str, right_structure: str) -> None:
        """Initialize bilateral structure midpoint strategy.

        Args:
            left_structure: Name of the left-side structure.
            right_structure: Name of the right-side structure.
        """
        self._left_structure = left_structure
        self._right_structure = right_structure

    def calculate(self, patient: "Patient") -> tuple[int, int, int] | None:
        """Calculate midpoint between bilateral structure COMs.

        Args:
            patient: Patient object with structure data.

        Returns:
            Voxel indices (Z, Y, X) of midpoint between structures, or None
            if either structure is missing or empty.
        """
        # Check both structures are present
        if (
            self._left_structure not in patient.structures
            or self._right_structure not in patient.structures
        ):
            return None

        # Calculate COM for left structure
        left_mask = patient.structures[self._left_structure]
        left_nonzero = np.argwhere(left_mask > 0)
        if len(left_nonzero) == 0:
            return None
        left_com = left_nonzero.mean(axis=0)

        # Calculate COM for right structure
        right_mask = patient.structures[self._right_structure]
        right_nonzero = np.argwhere(right_mask > 0)
        if len(right_nonzero) == 0:
            return None
        right_com = right_nonzero.mean(axis=0)

        # Calculate midpoint
        midpoint = (left_com + right_com) / 2
        return int(midpoint[0]), int(midpoint[1]), int(midpoint[2])

    def get_required_structures(self) -> list[str]:
        """Get required bilateral structure names.

        Returns:
            List containing both left and right structure names.
        """
        return [self._left_structure, self._right_structure]


class CombinedStructuresCOM(ReferenceStrategy):
    """Calculate center-of-mass of multiple structures combined.

    Merges multiple structure masks and calculates the COM of the combined
    mask. Returns None if all structures are missing or empty.
    """

    def __init__(self, structure_names: list[str]) -> None:
        """Initialize combined structures COM strategy.

        Args:
            structure_names: List of structure names to combine.
        """
        self._structure_names = structure_names

    def calculate(self, patient: "Patient") -> tuple[int, int, int] | None:
        """Calculate COM of combined structures.

        Args:
            patient: Patient object with structure data.

        Returns:
            Voxel indices (Z, Y, X) of combined structures COM, or None if
            all structures are missing or empty.
        """
        # Combine all available structure masks
        combined_mask = None
        for structure_name in self._structure_names:
            if structure_name in patient.structures:
                mask = patient.structures[structure_name]
                if combined_mask is None:
                    combined_mask = mask.copy()
                else:
                    combined_mask = np.logical_or(combined_mask, mask)

        if combined_mask is None:
            return None

        nonzero_indices = np.argwhere(combined_mask > 0)
        if len(nonzero_indices) == 0:
            return None

        # Calculate mean position of all nonzero voxels
        com = nonzero_indices.mean(axis=0)
        return int(com[0]), int(com[1]), int(com[2])

    def get_required_structures(self) -> list[str]:
        """Get required structure names.

        Returns:
            List of structure names to combine.
        """
        return self._structure_names


class FallbackStrategy(ReferenceStrategy):
    """Try multiple strategies sequentially until one succeeds.

    Attempts each strategy in order, returning the first successful result.
    This is useful for implementing fallback chains, e.g., try anatomical
    reference first, fall back to volume center if structures missing.
    """

    def __init__(self, strategies: list[ReferenceStrategy]) -> None:
        """Initialize fallback strategy chain.

        Args:
            strategies: List of strategies to try in order.
        """
        self._strategies = strategies

    def calculate(self, patient: "Patient") -> tuple[int, int, int] | None:
        """Try strategies sequentially until one succeeds.

        Args:
            patient: Patient object with structure data.

        Returns:
            Voxel indices (Z, Y, X) from first successful strategy, or None
            if all strategies fail.
        """
        for strategy in self._strategies:
            result = strategy.calculate(patient)
            if result is not None:
                return result
        return None

    def get_required_structures(self) -> list[str]:
        """Get combined required structures from all strategies.

        Returns:
            Combined list of all required structures from all strategies.
        """
        all_structures = []
        for strategy in self._strategies:
            all_structures.extend(strategy.get_required_structures())
        return list(set(all_structures))  # Remove duplicates


class VolumeCenterStrategy(ReferenceStrategy):
    """Calculate geometric center of the volume.

    Returns the center voxel of the CT volume. This strategy always succeeds
    and is commonly used as a fallback when anatomical references are missing.
    """

    def calculate(self, patient: "Patient") -> tuple[int, int, int] | None:
        """Calculate geometric center of CT volume.

        Args:
            patient: Patient object with CT data.

        Returns:
            Voxel indices (Z, Y, X) of volume center. Always succeeds.
        """
        if patient.ct is None:
            # Fallback to dose if CT not available
            if patient.dose is None:
                return None
            shape = patient.dose.shape
        else:
            shape = patient.ct.shape

        center_z = shape[0] // 2
        center_y = shape[1] // 2
        center_x = shape[2] // 2

        return center_z, center_y, center_x

    def get_required_structures(self) -> list[str]:
        """Get required structures (none for volume center).

        Returns:
            Empty list, as volume center doesn't require structures.
        """
        return []
