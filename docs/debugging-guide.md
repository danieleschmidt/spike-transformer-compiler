# Debugging Guide

This guide covers debugging techniques and tools for Spike-Transformer-Compiler development.

## IDE Setup

### Visual Studio Code

The repository includes comprehensive VS Code configuration:

**Extensions**: Install recommended extensions from `.vscode/extensions.json`
**Settings**: Pre-configured for Python development with neuromorphic focus
**Tasks**: Integrated build, test, and quality tasks
**Launch Configurations**: Debug configurations for various scenarios

#### Debug Configurations

1. **Python: Current File** - Debug the currently open Python file
2. **Debug: spike-compile CLI** - Debug the command-line interface
3. **Debug: Compiler Unit Tests** - Debug specific unit tests
4. **Debug: Integration Tests** - Debug integration test suite
5. **Debug: Performance Tests** - Debug performance benchmarks
6. **Profile: Compilation Performance** - Profile compilation with cProfile
7. **Attach to Remote Docker** - Debug in Docker containers

### PyCharm/IntelliJ

For JetBrains IDEs, the repository includes `.idea/` configuration:

```bash
# Open project in PyCharm
pycharm-professional .

# Or IntelliJ with Python plugin
idea .
```

## Debugging Techniques

### Compiler Pipeline Debugging

```python
from spike_transformer_compiler import SpikeCompiler

# Enable debug mode
compiler = SpikeCompiler(debug=True, verbose=True)

# Enable IR dumping
compiler.set_debug_options(
    dump_ir=True,
    dump_passes=True
)

# Compile with debug information
try:
    result = compiler.compile(model, input_shape)
except Exception as e:
    # Debug information will be available in logs
    print(f"Compilation failed: {e}")
```

### Optimization Pass Debugging

```python
from spike_transformer_compiler.optimization import Optimizer

optimizer = Optimizer()

# Add passes with debug information
optimizer.add_pass(OptimizationPass.SPIKE_COMPRESSION, debug=True)
optimizer.add_pass(OptimizationPass.WEIGHT_QUANTIZATION, bits=4, debug=True)

# Run with detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
result = optimizer.run(ir)
```

### Hardware Simulation Debugging

```python
# Use simulation backend for debugging
compiler = SpikeCompiler(target="simulation", debug=True)

# Compile model
compiled_model = compiler.compile(model, input_shape)

# Show detailed execution trace
compiled_model.debug_trace()

# Analyze spike patterns
spike_data = compiled_model.get_spike_data()
print(f"Total spikes: {spike_data.total_spikes}")
print(f"Spike rate: {spike_data.average_rate} Hz")
```

## Profiling

### Performance Profiling

```bash
# Profile compilation performance
python -m cProfile -o compilation.prof -m spike_transformer_compiler.cli compile model.pth

# Analyze profile
python -m pstats compilation.prof
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler examples/compile_large_model.py

# Or use line-by-line profiling
@profile
def compile_model():
    # Your compilation code here
    pass
```

### Energy Profiling

```python
from spike_transformer_compiler.profiling import EnergyProfiler

profiler = EnergyProfiler(backend="simulation")

# Profile energy consumption
energy_report = profiler.profile(
    compiled_model,
    test_inputs=test_data,
    breakdown_by=["layer", "operation"]
)

# Visualize results
profiler.plot_energy_breakdown(energy_report)
```

## Common Debugging Scenarios

### Compilation Failures

```python
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    compiler.compile(model, input_shape)
except CompilationError as e:
    print(f"Compilation failed at stage: {e.stage}")
    print(f"Error details: {e.details}")
    
    # Inspect intermediate representation
    if hasattr(e, 'ir_state'):
        print(f"IR state: {e.ir_state}")
```

### Performance Issues

```python
import time
from spike_transformer_compiler.monitoring import monitor_compilation

@monitor_compilation(target="loihi3", model_type="spikeformer")
def debug_slow_compilation():
    start = time.time()
    
    # Your compilation code
    result = compiler.compile(model, input_shape)
    
    duration = time.time() - start
    print(f"Compilation took {duration:.2f} seconds")
    
    return result
```

### Hardware Issues

```python
# Check hardware connectivity
from spike_transformer_compiler.backend import Loihi3Backend

backend = Loihi3Backend()

# Test connection
if backend.test_connection():
    print("Hardware connection OK")
else:
    print("Hardware connection failed")
    backend.diagnose_connection_issues()
```

## Testing and Validation

### Unit Test Debugging

```bash
# Run specific test with debugging
pytest tests/test_compiler.py::TestSpikeCompiler::test_compile_basic -vvv --pdb

# Run with coverage and debugging
pytest --cov=spike_transformer_compiler --pdb-trace tests/
```

### Integration Test Debugging

```bash
# Run integration tests with detailed output
pytest tests/integration/ -v --tb=long

# Run hardware tests (requires hardware)
pytest tests/integration/ -m hardware --tb=short
```

### Performance Test Debugging

```bash
# Run performance tests with profiling
pytest tests/performance/ -v --profile

# Compare against baseline
pytest tests/performance/ --benchmark-compare
```

## Docker Debugging

### Development Container

```bash
# Build and run development container
docker-compose up spike-compiler-dev

# Attach debugger
docker-compose exec spike-compiler-dev python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m spike_transformer_compiler.cli
```

### Remote Debugging

```python
# Add to your Python code for remote debugging
import debugpy
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()  # Blocks execution until debugger attaches
```

## Logging Configuration

### Structured Logging

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        })

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)

# Use structured logging for neuromorphic operations
logger = logging.getLogger('spike_compiler')
logger.info('Compilation started', extra={
    'model_type': 'spikeformer',
    'target': 'loihi3',
    'optimization_level': 2
})
```

## Advanced Debugging

### IR Visualization

```python
# Visualize intermediate representation
from spike_transformer_compiler.ir import IRVisualizer

visualizer = IRVisualizer()
visualizer.plot_graph(ir, output_file="ir_graph.pdf")
visualizer.plot_optimization_passes(ir, passes, output_file="optimization_trace.pdf")
```

### Spike Pattern Analysis

```python
# Analyze spike patterns for debugging
from spike_transformer_compiler.analysis import SpikeAnalyzer

analyzer = SpikeAnalyzer()
spike_patterns = analyzer.analyze_patterns(compiled_model, test_inputs)

# Visualize spike raster plots
analyzer.plot_raster(spike_patterns, layers=['attention', 'feedforward'])

# Detect anomalous patterns
anomalies = analyzer.detect_anomalies(spike_patterns)
if anomalies:
    print(f"Found {len(anomalies)} anomalous spike patterns")
```

## Troubleshooting Common Issues

### Import Errors
- Ensure `PYTHONPATH` includes `src/` directory
- Verify virtual environment is activated
- Check package installation with `pip list`

### Test Failures
- Run tests individually to isolate issues
- Check test fixtures and sample data
- Verify hardware availability for hardware tests

### Performance Issues
- Profile code with cProfile or py-spy
- Check memory usage with memory_profiler
- Monitor system resources during compilation

### Hardware Connectivity
- Verify Loihi SDK installation
- Check hardware permissions and drivers
- Test with simulation backend first

For additional debugging support, see the [Development Guide](DEVELOPMENT.md) or contact the development team.