"""
Global pytest configuration and fixtures for Spike-Transformer-Compiler tests.

This module provides shared fixtures, configuration, and utilities used
across all test modules in the project.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import Mock

import pytest
import torch
import numpy as np

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from spike_transformer_compiler import SpikeCompiler
    from spike_transformer_compiler.config import CompilerConfig
except ImportError:
    # Mock for when modules don't exist yet
    class SpikeCompiler:
        def __init__(self, **kwargs):
            self.config = kwargs
    
    class CompilerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def to_dict(self):
            return self.__dict__


# ==============================================================================
# Test Environment Configuration
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Configure test environment settings."""
    # Set environment variables for testing
    os.environ["SPIKE_COMPILER_ENV"] = "testing"
    os.environ["SPIKE_COMPILER_LOG_LEVEL"] = "DEBUG"
    os.environ["HARDWARE_SIMULATION_ENABLED"] = "true"
    os.environ["COMPILATION_CACHE_ENABLED"] = "false"  # Disable cache for tests
    
    # Create test directories
    test_dirs = [
        "tests/logs",
        "tests/artifacts", 
        "tests/tmp",
        "tests/outputs"
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    import shutil
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir, ignore_errors=True)


# ==============================================================================
# Hardware and Simulation Fixtures
# ==============================================================================

@pytest.fixture
def mock_loihi_hardware():
    """Mock Loihi 3 hardware for testing."""
    mock_hardware = Mock()
    mock_hardware.is_available = True
    mock_hardware.num_chips = 2
    mock_hardware.cores_per_chip = 128
    mock_hardware.deploy.return_value = True
    mock_hardware.run.return_value = np.random.random((1, 1000))  # Mock output
    return mock_hardware


@pytest.fixture
def hardware_simulation():
    """Enable hardware simulation mode."""
    original_value = os.environ.get("HARDWARE_SIMULATION_ENABLED")
    os.environ["HARDWARE_SIMULATION_ENABLED"] = "true"
    yield
    if original_value:
        os.environ["HARDWARE_SIMULATION_ENABLED"] = original_value
    else:
        os.environ.pop("HARDWARE_SIMULATION_ENABLED", None)


# ==============================================================================
# Compiler and Configuration Fixtures  
# ==============================================================================

@pytest.fixture
def basic_compiler_config():
    """Basic compiler configuration for testing."""
    return CompilerConfig(
        optimization_level=1,
        target="simulation",
        time_steps=4,
        spike_encoding="rate",
        debug_mode=True
    )


@pytest.fixture
def spike_compiler(basic_compiler_config):
    """Spike compiler instance with test configuration."""
    return SpikeCompiler(config=basic_compiler_config)


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
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


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


@pytest.fixture
def performance_monitor():
    """Performance monitoring context manager."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.peak_memory = 0
            
        def __enter__(self):
            import time
            try:
                import psutil
                self.start_time = time.time()
                self.process = psutil.Process()
                self.initial_memory = self.process.memory_info().rss
            except ImportError:
                self.start_time = time.time()
                self.initial_memory = 0
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            import time
            self.end_time = time.time()
            if hasattr(self, 'process'):
                self.peak_memory = max(self.peak_memory, 
                                     self.process.memory_info().rss - self.initial_memory)
            
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
            
    return PerformanceMonitor()


# ==============================================================================
# Pytest Hooks and Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    markers = [
        "unit: Unit tests - fast, isolated tests",
        "integration: Integration tests - test component interactions", 
        "performance: Performance tests - measure speed, memory, energy",
        "hardware: Tests requiring physical neuromorphic hardware",
        "simulation: Tests using hardware simulation",
        "slow: Slow running tests (>1 second)",
        "gpu: Tests requiring GPU hardware",
        "loihi: Tests specific to Intel Loihi hardware",
        "model: Tests involving neural network models",
        "compiler: Tests for compilation pipeline",
        "optimization: Tests for optimization passes",
        "energy: Energy profiling tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file location
        if "performance" in str(item.fspath) or "benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip hardware tests if hardware not available
    if item.get_closest_marker("hardware"):
        if not os.environ.get("TEST_HARDWARE_AVAILABLE", "false").lower() == "true":
            pytest.skip("Hardware tests require TEST_HARDWARE_AVAILABLE=true")
    
    # Skip GPU tests if no GPU available  
    if item.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU tests require CUDA-capable GPU")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)