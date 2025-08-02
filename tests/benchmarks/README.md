# Performance Benchmarks

This directory contains performance benchmarks for the Spike-Transformer-Compiler, measuring compilation speed, energy efficiency, and inference performance.

## Benchmark Categories

### Compilation Performance
- **test_compilation_speed.py**: Measure compilation time for various model sizes
- **test_memory_usage.py**: Monitor memory consumption during compilation
- **test_optimization_impact.py**: Measure impact of different optimization passes

### Runtime Performance  
- **test_inference_latency.py**: Measure inference latency on different hardware
- **test_energy_consumption.py**: Detailed energy profiling and measurement
- **test_throughput.py**: Measure inference throughput (images/second)

### Accuracy Benchmarks
- **test_accuracy_vs_optimization.py**: Trade-off between optimization and accuracy
- **test_quantization_impact.py**: Impact of quantization on model accuracy
- **test_hardware_vs_simulation.py**: Accuracy comparison between hardware and simulation

### Scaling Benchmarks
- **test_multi_chip_scaling.py**: Performance scaling across multiple chips
- **test_model_size_scaling.py**: Performance vs model size relationships
- **test_batch_size_impact.py**: Impact of batch size on performance

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ -m "performance"

# Run specific benchmark category
pytest tests/benchmarks/test_compilation_speed.py

# Run with detailed profiling
pytest tests/benchmarks/ --profile

# Run and save results
pytest tests/benchmarks/ --benchmark-json=results.json
```

## Benchmark Configuration

Benchmarks use specialized configuration in `pytest.ini`:

```ini
markers =
    performance: Performance benchmarks
    energy: Energy measurement benchmarks
    slow: Long-running benchmarks
    hardware: Hardware-dependent benchmarks
```

## Hardware Requirements

Different benchmarks have different requirements:

- **CPU Benchmarks**: Standard CPU performance tests
- **GPU Baseline**: Require NVIDIA GPU for baseline comparisons
- **Loihi Hardware**: Require Intel Loihi 3 hardware
- **Energy Measurement**: Require power measurement hardware

## Benchmark Data

Results are stored in structured format:

```json
{
  "test_name": "compilation_speed_spikeformer_base",
  "timestamp": "2025-08-02T10:30:00Z",
  "metrics": {
    "compilation_time_ms": 45000,
    "memory_peak_mb": 2048,
    "model_size_mb": 85
  },
  "environment": {
    "hardware": "Loihi3-2chip",
    "software_version": "1.0.0"
  }
}
```

## Performance Targets

Current performance targets:

| Metric | Target | Current |
|--------|--------|---------|
| Compilation Time (SpikeFormer-Base) | < 60s | 45s |
| Memory Usage (Peak) | < 4GB | 2GB |
| Energy per Inference | < 5mJ | 2.1mJ |
| Inference Latency | < 25ms | 15ms |

## Regression Testing

Benchmarks include regression detection:

- **Performance Regression**: >10% slowdown triggers failure
- **Energy Regression**: >5% increase in energy consumption
- **Accuracy Regression**: >1% accuracy loss

## CI/CD Integration

### Pull Request Validation
- Basic performance smoke tests
- Quick compilation benchmarks
- Memory usage validation

### Nightly Performance Tests
- Full benchmark suite
- Performance regression detection
- Energy efficiency validation

### Release Validation
- Comprehensive performance testing
- Hardware validation on all supported platforms
- Performance report generation

## Profiling Integration

Benchmarks integrate with profiling tools:

```python
@pytest.mark.performance
def test_compilation_with_profiling():
    with Profiler() as prof:
        result = compile_model(model)
    
    # Analyze hotspots
    assert prof.get_cpu_time() < MAX_CPU_TIME
    assert prof.get_memory_peak() < MAX_MEMORY
```

## Baseline Comparisons

Benchmarks compare against established baselines:

- **GPU (PyTorch)**: Standard deep learning baseline
- **CPU (ONNX)**: CPU inference baseline
- **Previous Versions**: Regression testing against previous releases

## Result Analysis

Benchmark results are analyzed for:

1. **Performance Trends**: Track performance over time
2. **Hardware Utilization**: Analyze resource usage patterns
3. **Bottleneck Identification**: Identify performance bottlenecks
4. **Optimization Opportunities**: Guide optimization efforts

## Adding New Benchmarks

1. **Create Test File**: Add to appropriate subdirectory
2. **Use Markers**: Mark with appropriate pytest markers
3. **Document Requirements**: List hardware/software requirements
4. **Set Timeouts**: Configure appropriate test timeouts
5. **Add Assertions**: Include performance assertions