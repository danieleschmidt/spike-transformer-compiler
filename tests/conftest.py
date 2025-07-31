"""Pytest configuration and fixtures for Spike-Transformer-Compiler tests."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from spike_transformer_compiler import SpikeCompiler


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_model():
    """Provide a simple PyTorch model for testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()


@pytest.fixture
def spike_compiler():
    """Provide a SpikeCompiler instance for testing."""
    return SpikeCompiler(
        target="simulation",
        optimization_level=1,
        debug=True
    )


@pytest.fixture
def sample_input():
    """Provide sample input tensor for testing."""
    return torch.randn(1, 10)


@pytest.fixture(scope="session")
def performance_baseline():
    """Load performance baselines for regression testing."""
    return {
        "compile_time_ms": 1000,
        "inference_time_ms": 50,
        "memory_usage_mb": 100,
        "energy_per_inference_mj": 2.5
    }


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )