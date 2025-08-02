# End-to-End Tests

This directory contains end-to-end tests that validate the complete compilation pipeline from PyTorch models to Loihi 3 deployment.

## Test Categories

### Complete Pipeline Tests
- **test_full_compilation.py**: Complete compilation from PyTorch to Loihi binary
- **test_model_deployment.py**: Model deployment and execution on hardware/simulator
- **test_optimization_pipeline.py**: Full optimization pipeline with all passes

### Real-world Scenarios
- **test_image_classification.py**: Complete image classification workflow
- **test_real_time_inference.py**: Real-time inference scenarios
- **test_multi_chip_deployment.py**: Multi-chip scaling and deployment

### Performance Validation
- **test_energy_efficiency.py**: Energy consumption validation
- **test_latency_requirements.py**: Latency requirement validation
- **test_accuracy_preservation.py**: Model accuracy preservation tests

## Hardware Requirements

Some e2e tests require specific hardware:

- **Simulation Mode**: All tests can run in simulation mode
- **Loihi 3 Hardware**: Hardware tests marked with `@pytest.mark.hardware`
- **GPU**: Some baseline comparisons require GPU hardware

## Configuration

E2E tests use configuration files from `tests/fixtures/configs/`:

```python
@pytest.fixture
def e2e_config():
    return load_config("tests/fixtures/configs/e2e_config.yaml")
```

## Running E2E Tests

```bash
# Run all e2e tests
pytest tests/e2e/

# Run only simulation tests (no hardware required)
pytest tests/e2e/ -m "not hardware"

# Run with detailed output
pytest tests/e2e/ -v -s

# Run specific test category
pytest tests/e2e/test_full_compilation.py
```

## Test Data

E2E tests use larger test datasets stored in `tests/fixtures/data/e2e/`:

- Sample models of various sizes
- Representative input datasets
- Expected output references
- Performance baselines

## Timeout and Resource Limits

E2E tests have extended timeouts due to compilation complexity:

- Default timeout: 300 seconds
- Large model timeout: 600 seconds
- Hardware tests timeout: 900 seconds

## CI/CD Integration

E2E tests in CI:

- **PR Validation**: Basic e2e tests run on every PR
- **Nightly Builds**: Full e2e suite including hardware simulation
- **Release Validation**: Complete test suite with performance validation

## Debugging E2E Failures

1. **Check Logs**: E2E tests produce detailed logs in `tests/logs/`
2. **Intermediate Artifacts**: Check compilation artifacts in debug mode
3. **Resource Usage**: Monitor memory and CPU usage during tests
4. **Hardware Status**: Verify hardware/simulator status for hardware tests