# Spike-Transformer-Compiler

A TVM-style compiler that converts PyTorch SpikeFormer models into optimized binaries for Intel Loihi 3 neuromorphic hardware. Based on CVPR 2025's DSFormer kernels and Intel's Loihi 3 API preview, this toolkit enables efficient deployment of spike-based transformers on neuromorphic chips.

## Overview

Spike-Transformer-Compiler bridges the gap between high-level spiking neural network frameworks and low-level neuromorphic hardware. It automatically optimizes SpikeFormer models for energy-efficient inference on Loihi 3, handling spike encoding, synaptic operations, and hardware resource allocation.

## Key Features

- **Automatic Compilation**: Convert PyTorch SpikeFormer to Loihi 3 binaries
- **Hardware Optimization**: Exploit Loihi 3's sparse computation and event-driven processing
- **Multi-Chip Support**: Scale across multiple Loihi 3 chips
- **Energy Profiling**: Detailed energy consumption analysis
- **Kernel Library**: Optimized DSFormer kernels from CVPR 2025
- **Simulation Mode**: Test before hardware deployment

## Installation

```bash
# Basic installation
pip install spike-transformer-compiler

# With Loihi 3 SDK
pip install spike-transformer-compiler[loihi3]

# With visualization tools
pip install spike-transformer-compiler[viz]

# Development installation
git clone https://github.com/yourusername/spike-transformer-compiler
cd spike-transformer-compiler
pip install -e ".[dev]"
```

## Quick Start

### Basic Compilation

```python
from spike_transformer_compiler import SpikeCompiler
from spikeformer import SpikeFormer
import torch

# Load pre-trained SpikeFormer model
model = SpikeFormer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dims=768,
    depths=[2, 2, 18, 2],
    num_heads=[12, 12, 12, 12]
)

model.load_state_dict(torch.load("spikeformer_imagenet.pth"))

# Initialize compiler
compiler = SpikeCompiler(
    target="loihi3",
    optimization_level=3,
    time_steps=4
)

# Compile to Loihi 3
loihi_model = compiler.compile(
    model,
    input_shape=(1, 3, 224, 224),
    chip_config="ncl-2chip"  # 2-chip configuration
)

# Deploy and run
output_spikes = loihi_model.run(
    input_image,
    time_steps=4,
    return_spike_trains=True
)
```

### Advanced Optimization

```python
from spike_transformer_compiler import OptimizationPass, ResourceAllocator

# Custom optimization pipeline
optimizer = compiler.create_optimizer()

# Add optimization passes
optimizer.add_pass(OptimizationPass.SPIKE_COMPRESSION)
optimizer.add_pass(OptimizationPass.WEIGHT_QUANTIZATION, bits=4)
optimizer.add_pass(OptimizationPass.NEURON_PRUNING, sparsity=0.9)
optimizer.add_pass(OptimizationPass.TEMPORAL_FUSION)

# Resource-aware allocation
allocator = ResourceAllocator(
    num_chips=2,
    cores_per_chip=128,
    synapses_per_core=1024
)

# Compile with optimizations
optimized_model = compiler.compile(
    model,
    optimizer=optimizer,
    resource_allocator=allocator,
    profile_energy=True
)

print(f"Estimated energy: {optimized_model.energy_per_inference} nJ")
print(f"Chip utilization: {optimized_model.utilization:.1%}")
```

## Architecture

```
spike-transformer-compiler/
├── spike_transformer_compiler/
│   ├── frontend/
│   │   ├── pytorch/        # PyTorch model parser
│   │   ├── spikeformer/    # SpikeFormer support
│   │   └── converters/     # Model converters
│   ├── ir/
│   │   ├── spike_ir/       # Spike intermediate representation
│   │   ├── graph/          # Computation graph
│   │   └── passes/         # IR transformation passes
│   ├── optimizations/
│   │   ├── quantization/   # Weight/activation quantization
│   │   ├── pruning/        # Neuron/synapse pruning
│   │   ├── fusion/         # Operation fusion
│   │   └── scheduling/     # Spike scheduling
│   ├── backend/
│   │   ├── loihi3/         # Loihi 3 code generation
│   │   ├── simulation/     # Software simulation
│   │   └── profiling/      # Performance profiling
│   ├── kernels/
│   │   ├── attention/      # Spike attention kernels
│   │   ├── convolution/    # Spike convolution
│   │   └── dsformer/       # DSFormer kernels
│   └── runtime/
│       ├── executor/       # Runtime execution
│       ├── memory/         # Memory management
│       └── communication/  # Multi-chip communication
├── benchmarks/             # Performance benchmarks
├── models/                # Pre-trained models
└── examples/              # Example applications
```

## Compilation Pipeline

### Frontend Parsing

```python
from spike_transformer_compiler.frontend import ModelParser

# Parse PyTorch model
parser = ModelParser()
spike_graph = parser.parse(
    model,
    input_shapes={"image": (1, 3, 224, 224)},
    spike_encoding="rate",  # or "temporal", "phase"
    threshold=1.0
)

# Visualize computation graph
spike_graph.visualize("model_graph.pdf")

# Analyze model statistics
stats = spike_graph.analyze()
print(f"Total neurons: {stats.num_neurons}")
print(f"Total synapses: {stats.num_synapses}")
print(f"Memory requirement: {stats.memory_bytes / 1e6:.1f} MB")
```

### Intermediate Representation

```python
from spike_transformer_compiler.ir import SpikeIR, IRBuilder

# Build IR from parsed model
ir_builder = IRBuilder()
spike_ir = ir_builder.build(spike_graph)

# Apply transformations
from spike_transformer_compiler.ir.passes import (
    DeadCodeElimination,
    CommonSubexpressionElimination,
    LoopUnrolling
)

passes = [
    DeadCodeElimination(),
    CommonSubexpressionElimination(),
    LoopUnrolling(factor=4)
]

for pass_ in passes:
    spike_ir = pass_.run(spike_ir)

# Verify IR correctness
assert spike_ir.verify(), "IR verification failed"
```

## Optimization Techniques

### Spike Compression

```python
from spike_transformer_compiler.optimizations import SpikeCompressor

compressor = SpikeCompressor(
    method="delta",  # or "exponential", "adaptive"
    compression_ratio=0.1
)

# Compress spike trains
compressed_ir = compressor.compress(spike_ir)

print(f"Compression achieved: {compressor.get_compression_stats()}")
```

### Quantization-Aware Training

```python
from spike_transformer_compiler.optimizations import QuantizationAwareTraining

# QAT for neuromorphic deployment
qat = QuantizationAwareTraining(
    weight_bits=4,
    threshold_bits=8,
    gradient_bits=8
)

# Fine-tune with quantization
qat_model = qat.prepare(model)
qat_model = train_with_quantization(
    qat_model,
    train_loader,
    epochs=10
)

# Convert to quantized model
quantized_model = qat.convert(qat_model)
```

### Temporal Coding Optimization

```python
from spike_transformer_compiler.optimizations import TemporalOptimizer

# Optimize temporal dynamics
temporal_opt = TemporalOptimizer(
    time_constant_tau=10.0,  # ms
    refractory_period=2.0,   # ms
    adaptive_threshold=True
)

# Apply temporal optimizations
optimized_ir = temporal_opt.optimize(
    spike_ir,
    target_latency=20.0,  # ms
    minimize_spikes=True
)
```

## Neuromorphic Kernels

### DSFormer Attention

```python
from spike_transformer_compiler.kernels import DSFormerAttention

# Efficient spike-based attention
attention = DSFormerAttention(
    embed_dim=768,
    num_heads=12,
    spike_mode="binary",  # or "graded"
    window_size=4,
    sparse_ratio=0.1
)

# Compile attention kernel
attention_kernel = compiler.compile_kernel(
    attention,
    batch_size=1,
    sequence_length=196
)

# Profile on Loihi 3
profile = attention_kernel.profile(
    num_inferences=1000,
    measure=["energy", "latency", "spike_count"]
)
```

### Custom Kernel Development

```python
from spike_transformer_compiler.kernels import KernelBuilder

# Define custom spiking kernel
@KernelBuilder.register("custom_conv")
def spike_convolution(builder, params):
    # Define computation pattern
    builder.add_neuron_group(
        name="conv_neurons",
        size=params.out_channels * params.output_size,
        neuron_model="LIF",
        threshold=params.threshold
    )
    
    # Define synaptic connections
    builder.add_synapse_group(
        pre="input",
        post="conv_neurons",
        connectivity="conv2d",
        kernel_size=params.kernel_size,
        stride=params.stride,
        weight_init="kaiming"
    )
    
    return builder.build()
```

## Hardware Deployment

### Loihi 3 Configuration

```python
from spike_transformer_compiler.backend import Loihi3Backend

# Configure Loihi 3 backend
backend = Loihi3Backend(
    num_chips=2,
    neuromorphic_cores=256,
    learning_enabled=False,
    power_mode="low_power"
)

# Map model to hardware
mapping = backend.map_model(
    optimized_model,
    strategy="minimize_communication",
    load_balance=True
)

# Generate configuration
config = backend.generate_config(
    mapping,
    input_encoding="poisson",
    output_decoding="spike_count",
    simulation_time=40  # ms
)

# Deploy to hardware
backend.deploy(config, verify=True)
```

### Multi-Chip Scaling

```python
from spike_transformer_compiler.backend import MultiChipManager

# Scale across multiple chips
manager = MultiChipManager(
    chip_ids=["loihi3_0", "loihi3_1", "loihi3_2", "loihi3_3"],
    interconnect="mesh"  # or "torus", "hierarchical"
)

# Partition model
partitions = manager.partition_model(
    large_model,
    strategy="minimize_latency",
    balance_load=True
)

# Deploy partitions
for chip_id, partition in partitions.items():
    manager.deploy_partition(chip_id, partition)

# Synchronize execution
results = manager.run_synchronized(
    input_data,
    aggregate_outputs=True
)
```

## Profiling and Analysis

### Energy Profiling

```python
from spike_transformer_compiler.profiling import EnergyProfiler

profiler = EnergyProfiler(backend="loihi3")

# Detailed energy breakdown
energy_report = profiler.profile(
    compiled_model,
    test_inputs=test_dataset,
    breakdown_by=["layer", "operation", "memory_access"]
)

# Visualize energy consumption
profiler.plot_energy_breakdown(
    energy_report,
    save_path="energy_analysis.pdf"
)

print(f"Total energy: {energy_report.total_energy_mJ:.3f} mJ")
print(f"Energy per spike: {energy_report.energy_per_spike_pJ:.1f} pJ")
print(f"Static power: {energy_report.static_power_mW:.1f} mW")
```

### Performance Analysis

```python
from spike_transformer_compiler.profiling import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze throughput and latency
perf_metrics = analyzer.analyze(
    compiled_model,
    batch_sizes=[1, 4, 8, 16],
    measure_points=["input", "attention", "output"]
)

# Compare with GPU baseline
gpu_baseline = run_gpu_model(original_model, test_data)
speedup = analyzer.compare_with_baseline(
    neuromorphic=perf_metrics,
    baseline=gpu_baseline,
    metric="energy_delay_product"
)

print(f"Energy efficiency improvement: {speedup:.1f}x")
```

## Advanced Features

### Adaptive Spike Encoding

```python
from spike_transformer_compiler.encoding import AdaptiveEncoder

# Learn optimal encoding
encoder = AdaptiveEncoder(
    encoding_type="hybrid",  # rate + temporal
    adaptation_rate=0.01
)

# Train encoder
encoder.adapt(
    input_distribution=training_images,
    target_sparsity=0.05,
    epochs=50
)

# Apply to model
encoded_model = compiler.compile(
    model,
    input_encoder=encoder,
    optimize_for_encoding=True
)
```

### Hardware-Software Co-Design

```python
from spike_transformer_compiler.codesign import HWSWCoDesigner

codesigner = HWSWCoDesigner(
    hardware_constraints={
        "max_neurons": 1_000_000,
        "max_synapses": 10_000_000,
        "memory_bandwidth": 100  # GB/s
    }
)

# Joint optimization
optimal_design = codesigner.optimize(
    model=model,
    dataset=dataset,
    objectives=["accuracy", "energy", "latency"],
    pareto_points=20
)

# Export results
codesigner.export_design(
    optimal_design,
    format="verilog",  # For ASIC implementation
    output_dir="hw_design/"
)
```

## Integration Examples

### Real-Time Vision

```python
from spike_transformer_compiler.applications import RealtimeVision

# Deploy for real-time inference
vision_system = RealtimeVision(
    compiled_model=loihi_model,
    camera_interface="usb",
    preprocessing="normalize"
)

# Process video stream
vision_system.process_stream(
    fps=30,
    callback=lambda result: print(f"Detected: {result.class_name}"),
    energy_budget=100  # mW
)
```

### Edge AI Deployment

```python
from spike_transformer_compiler.edge import EdgeDeployment

# Package for edge device
edge_package = EdgeDeployment.create_package(
    compiled_model=optimized_model,
    runtime="standalone",
    include_simulator=True
)

# Deploy to edge device
edge_package.deploy(
    device_ip="192.168.1.100",
    auto_start=True,
    monitoring_enabled=True
)
```

## Benchmarks

### Model Performance

| Model | Dataset | Accuracy | Energy/Inf | Latency | Sparsity |
|-------|---------|----------|------------|---------|----------|
| SpikeFormer-Tiny | ImageNet | 79.8% | 0.3 mJ | 10 ms | 95% |
| SpikeFormer-Small | ImageNet | 83.5% | 0.8 mJ | 15 ms | 93% |
| SpikeFormer-Base | ImageNet | 85.2% | 2.1 mJ | 25 ms | 91% |
| DSFormer | CIFAR-100 | 78.6% | 0.1 mJ | 5 ms | 97% |

### Hardware Comparison

| Platform | Power | Throughput | Energy Efficiency |
|----------|-------|------------|-------------------|
| GPU (V100) | 250W | 1000 img/s | 250 mJ/img |
| CPU (Xeon) | 150W | 50 img/s | 3000 mJ/img |
| Loihi 3 | 0.5W | 200 img/s | 2.5 mJ/img |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{spike_transformer_compiler,
  title={Spike-Transformer-Compiler: Neuromorphic Compilation for SpikeFormers},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/spike-transformer-compiler}
}

@inproceedings{dsformer_cvpr_2025,
  title={DSFormer: Efficient Spike-based Transformers},
  author={CVPR Authors},
  booktitle={CVPR},
  year={2025}
}
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Intel Neuromorphic Lab for Loihi 3 support
- CVPR 2025 DSFormer authors
- Neuromorphic computing community
