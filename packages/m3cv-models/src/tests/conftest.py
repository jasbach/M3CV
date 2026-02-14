"""Pytest fixtures for m3cv-models tests."""

import pytest
import torch


@pytest.fixture
def sample_volume():
    """Create a sample 3D volume tensor."""
    # (batch, channels, depth, height, width)
    return torch.randn(2, 2, 16, 32, 32)


@pytest.fixture
def sample_volume_single_channel():
    """Create a sample single-channel 3D volume tensor."""
    return torch.randn(2, 1, 16, 32, 32)


@pytest.fixture
def sample_tabular():
    """Create sample tabular features."""
    return torch.randn(2, 10)


@pytest.fixture
def sample_labels():
    """Create sample binary labels."""
    return torch.randint(0, 2, (2,))
