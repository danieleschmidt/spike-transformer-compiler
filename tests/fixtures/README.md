# Test Fixtures

This directory contains test fixtures and sample data for the Spike-Transformer-Compiler test suite.

## Directory Structure

```
fixtures/
├── models/           # Sample neural network models for testing
├── data/             # Test input data (images, spike trains, etc.)
├── configs/          # Configuration files for testing
├── expected/         # Expected outputs for validation
├── hardware/         # Hardware simulation fixtures
└── benchmarks/       # Performance benchmark data
```

## Usage

Fixtures are automatically discovered by pytest and can be used in tests:

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_model():
    """Load a sample SpikeFormer model for testing."""
    fixture_path = Path(__file__).parent / "fixtures" / "models" / "sample_spikeformer.pth"
    return torch.load(fixture_path)

def test_compilation(sample_model):
    """Test compilation with sample model."""
    compiler = SpikeCompiler()
    compiled_model = compiler.compile(sample_model)
    assert compiled_model is not None
```

## Adding New Fixtures

1. Place fixture files in the appropriate subdirectory
2. Keep fixtures small (< 10MB) to avoid repository bloat
3. Use descriptive names that indicate the fixture purpose
4. Add documentation for complex fixtures
5. Consider using pytest fixtures for dynamic test data

## Fixture Guidelines

- **Small Size**: Keep fixtures under 10MB when possible
- **Representative**: Fixtures should represent realistic use cases
- **Documented**: Include README files for complex fixture sets
- **Versioned**: Use semantic versioning for fixture data when applicable
- **Deterministic**: Ensure fixtures produce consistent results

## Available Fixtures

### Models
- `sample_spikeformer.pth` - Minimal SpikeFormer model for basic testing
- `quantized_model.pth` - Pre-quantized model for optimization testing
- `large_model.pth` - Larger model for performance testing

### Data
- `mnist_samples.npz` - MNIST digit samples for classification testing
- `spike_trains.npz` - Synthetic spike train data
- `energy_profiles.json` - Sample energy profiling data

### Configurations
- `basic_config.yaml` - Basic compiler configuration
- `optimization_config.yaml` - Configuration with all optimizations enabled
- `hardware_config.yaml` - Hardware-specific configuration

## Performance Considerations

Large fixtures can slow down test execution. Consider:

1. Using smaller representative samples
2. Lazy loading of fixtures
3. Parametrized tests to reduce fixture duplication
4. Mocking when appropriate instead of real data