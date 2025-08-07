# Spike-Transformer-Compiler API Reference

## Core Classes

### SpikeCompiler

Main interface for neuromorphic model compilation.

```python
from spike_transformer_compiler import SpikeCompiler

compiler = SpikeCompiler(
    target="loihi3",           # Target hardware: "loihi3", "simulation"
    optimization_level=2,      # Optimization level: 0-3
    time_steps=4,             # Number of time steps
    debug=False,              # Enable debug mode
    verbose=False             # Enable verbose output
)
```

**Methods:**

#### `compile(model, input_shape, **kwargs)`

Compile PyTorch model to neuromorphic hardware.

**Parameters:**
- `model` (nn.Module): PyTorch model to compile
- `input_shape` (tuple): Input tensor shape
- `chip_config` (str, optional): Hardware configuration
- `optimizer` (optional): Custom optimization pipeline
- `resource_allocator` (optional): Resource allocation strategy
- `profile_energy` (bool): Enable energy profiling
- `secure_mode` (bool): Enable security validation

**Returns:**
- `CompiledModel`: Ready-to-deploy compiled model

**Example:**
```python
compiled_model = compiler.compile(
    model,
    input_shape=(1, 3, 224, 224),
    chip_config="ncl-2chip",
    profile_energy=True
)
```

### CompiledModel

Represents a compiled neuromorphic model.

**Methods:**

#### `run(input_data, time_steps=4, return_spike_trains=False)`

Execute inference on compiled model.

**Parameters:**
- `input_data`: Input tensor or array
- `time_steps` (int): Number of temporal steps
- `return_spike_trains` (bool): Return detailed spike data

**Returns:**
- Output tensor or spike data dictionary

**Example:**
```python
output = compiled_model.run(
    input_image,
    time_steps=4,
    return_spike_trains=True
)
```

## Kernels

### DSFormerAttention

CVPR 2025 DSFormer attention mechanism for neuromorphic hardware.

```python
from spike_transformer_compiler.kernels import DSFormerAttention

attention = DSFormerAttention(
    embed_dim=768,          # Embedding dimension
    num_heads=12,           # Number of attention heads
    spike_mode="binary",    # Spike mode: "binary", "graded"
    window_size=4,          # Temporal window size
    sparse_ratio=0.1        # Attention sparsity ratio
)
```

**Methods:**

#### `get_resource_requirements(sequence_length, batch_size=1)`

Calculate hardware resource requirements.

**Returns:**
- Dictionary with neurons, synapses, memory requirements

#### `compile_for_loihi3(sequence_length)`

Generate Loihi 3 specific configuration.

#### `estimate_energy(sequence_length, activity_rate=0.05)`

Estimate energy consumption in nJ.

### AdaptiveEncoder

Adaptive spike encoding with learned parameters.

```python
from spike_transformer_compiler.kernels import AdaptiveEncoder

encoder = AdaptiveEncoder(
    encoding_type="hybrid",    # "rate", "temporal", "phase", "hybrid"
    time_steps=4,             # Number of time steps
    adaptation_rate=0.01,     # Learning rate for adaptation
    target_sparsity=0.05      # Target spike sparsity
)
```

**Methods:**

#### `encode(data)`

Encode continuous data to spike trains.

#### `decode(spikes)`

Decode spike trains to continuous values.

#### `adapt(input_distribution, epochs=50)`

Adapt encoding parameters based on data distribution.

## Runtime System

### NeuromorphicExecutor

High-performance runtime executor.

```python
from spike_transformer_compiler.runtime import NeuromorphicExecutor

executor = NeuromorphicExecutor(
    num_cores=8,              # Number of processing cores
    enable_parallel=True,     # Enable parallel execution
    debug=False               # Debug mode
)
```

### MemoryManager

Memory management for neuromorphic execution.

```python
from spike_transformer_compiler.runtime import MemoryManager

memory_manager = MemoryManager(
    memory_limit_mb=1024     # Memory limit in MB
)
```

### MultiChipCommunicator

Multi-chip communication system.

```python
from spike_transformer_compiler.runtime import MultiChipCommunicator

communicator = MultiChipCommunicator(
    chip_ids=["chip_0", "chip_1"],
    interconnect="mesh",      # "mesh", "torus", "hierarchical"
    buffer_size=10000        # Message buffer size
)
```

## Applications

### RealtimeVision

Real-time vision processing system.

```python
from spike_transformer_compiler.applications import RealtimeVision

vision_system = RealtimeVision(
    compiled_model=loihi_model,
    camera_interface="opencv",
    buffer_size=10,
    num_worker_threads=4
)

# Process video stream
stats = vision_system.process_stream(
    fps=30,
    callback=lambda result: print(f"Detected: {result}"),
    energy_budget=100,  # mW
    duration=60         # seconds
)
```

### ResearchPlatform

Comprehensive research platform for experiments.

```python
from spike_transformer_compiler.applications import ResearchPlatform, ExperimentConfig

platform = ResearchPlatform(base_dir="research_workspace")

# Configure experiment
config = ExperimentConfig(
    experiment_name="encoding_comparison",
    model_type="spikeformer",
    dataset="imagenet",
    time_steps=4,
    encoding_method="hybrid"
)

# Run experiment
results = platform.experiment_manager.run_experiment(config)
```

## Optimization

### PerformanceOptimizer

Advanced performance optimization with parallelization.

```python
from spike_transformer_compiler.optimization import PerformanceOptimizer

optimizer = PerformanceOptimizer(
    max_workers=8,
    enable_caching=True,
    enable_profiling=True
)

# Optimize compilation pipeline
optimized_graph, results = optimizer.optimize_compilation_pipeline(
    graph, optimization_passes, backend_config
)
```

## Configuration

### Environment Variables

```bash
# Core settings
SPIKE_COMPILER_TARGET="loihi3"
SPIKE_COMPILER_OPTIMIZATION_LEVEL=2
SPIKE_COMPILER_TIME_STEPS=4

# Performance
SPIKE_COMPILER_WORKERS=8
NEUROMORPHIC_MEMORY_LIMIT="2GB"

# Security
SPIKE_COMPILER_SECURE_MODE=true
MAX_MODEL_SIZE="100MB"

# Logging
SPIKE_COMPILER_LOG_LEVEL="INFO"
ENABLE_PERFORMANCE_PROFILING=true
```

### Configuration File

```yaml
# config.yaml
compiler:
  target: "loihi3"
  optimization_level: 2
  time_steps: 4
  
hardware:
  loihi3:
    num_chips: 2
    cores_per_chip: 128
    
performance:
  enable_caching: true
  parallel_workers: 8
  
security:
  secure_mode: true
  timeout_seconds: 300
```

## CLI Interface

### Basic Commands

```bash
# Compile model
spike-compile model.py --target loihi3 --optimize 2

# Run server
spike-compile server --port 8080 --workers 4

# Benchmark performance
spike-compile benchmark --model spikeformer --dataset imagenet

# Run experiments
spike-compile experiment --config experiment.yaml
```

### Advanced Usage

```bash
# Profile compilation
spike-compile profile --model transformer.py --input-shape 1,3,224,224

# Hardware utilization analysis
spike-compile hardware-stats --target loihi3 --chips 4

# Energy analysis
spike-compile energy-profile --model efficient_net.py --time-steps 8
```

## Error Handling

### Exception Hierarchy

```python
from spike_transformer_compiler.exceptions import (
    CompilationError,        # Base compilation error
    ValidationError,         # Input validation error
    RuntimeError,           # Runtime execution error
    HardwareError,          # Hardware-specific error
    SecurityError           # Security validation error
)

try:
    compiled_model = compiler.compile(model, input_shape)
except CompilationError as e:
    print(f"Compilation failed: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Suggestions: {e.suggestions}")
```

## Monitoring and Metrics

### Performance Metrics

```python
# Get compilation statistics
stats = compiler.get_compilation_stats()

# Runtime performance
perf_stats = executor.get_performance_stats()

# Memory usage
memory_stats = memory_manager.get_stats()

# Energy consumption
energy_stats = compiled_model.get_energy_stats()
```

### Health Checks

```python
# Check system health
health_status = compiler.health_check()

# Hardware status
hw_status = backend.get_hardware_status()

# Memory status
mem_status = memory_manager.get_health_status()
```

## Examples

### Basic Usage

```python
import torch
import torch.nn as nn
from spike_transformer_compiler import SpikeCompiler

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Compile for neuromorphic hardware
compiler = SpikeCompiler(target="loihi3")
compiled_model = compiler.compile(model, input_shape=(1, 784))

# Run inference
input_data = torch.randn(1, 784)
output = compiled_model.run(input_data)
```

### Advanced Optimization

```python
from spike_transformer_compiler import SpikeCompiler
from spike_transformer_compiler.optimization import OptimizationPass

# Configure advanced optimization
compiler = SpikeCompiler(
    target="loihi3",
    optimization_level=3,
    time_steps=8
)

# Custom optimization pipeline
optimizer = compiler.create_optimizer()
optimizer.add_pass(OptimizationPass.SPIKE_COMPRESSION)
optimizer.add_pass(OptimizationPass.WEIGHT_QUANTIZATION, bits=4)
optimizer.add_pass(OptimizationPass.TEMPORAL_FUSION)

# Compile with custom optimization
compiled_model = compiler.compile(
    model,
    input_shape=(1, 3, 224, 224),
    optimizer=optimizer,
    profile_energy=True
)
```

### Research Experiment

```python
from spike_transformer_compiler.applications import ResearchPlatform, ExperimentConfig

# Set up research platform
platform = ResearchPlatform()

# Define experiment
config = ExperimentConfig(
    experiment_name="attention_comparison",
    model_type="spikeformer",
    batch_sizes=[1, 4, 8, 16],
    encoding_methods=["rate", "temporal", "hybrid"],
    num_trials=5
)

# Run experiment suite
results = platform.experiment_manager.run_experiment_suite(
    config, 
    parameter_sweeps={"optimization_level": [1, 2, 3]}
)

# Analyze results
analysis = platform.analyze_results(results)
```

This API reference provides comprehensive documentation for all major classes, methods, and usage patterns in the Spike-Transformer-Compiler.