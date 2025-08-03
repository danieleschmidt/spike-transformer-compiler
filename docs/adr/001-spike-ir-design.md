# ADR-001: Spike Intermediate Representation Design

**Status**: Proposed  
**Date**: 2025-08-03  
**Authors**: Daniel Schmidt  
**Reviewers**: TBD  

## Context

The Spike-Transformer-Compiler needs an intermediate representation (IR) that can efficiently represent spiking neural network computations while enabling hardware-specific optimizations. The IR must bridge the gap between high-level PyTorch models and low-level neuromorphic hardware instructions.

## Decision

We will implement a **graph-based Spike IR** with the following characteristics:

### Core Design Principles
1. **Temporal Awareness**: Explicit time dimension in all operations
2. **Spike Semantics**: Native support for spike trains and event-driven computation  
3. **Hardware Abstraction**: Target-agnostic representation with hardware-specific lowering
4. **Optimization Friendly**: Support for standard compiler optimizations

### IR Components

#### Node Types
```python
# Neuron Operations
SpikeNeuron(neuron_model, threshold, reset_mode)
LIF(tau_mem, tau_syn, v_threshold, v_reset)
AdaptiveLIF(tau_mem, tau_syn, tau_adapt, beta)

# Spike Operations  
SpikeConv2d(in_channels, out_channels, kernel_size, stride)
SpikeLinear(in_features, out_features)
SpikeAttention(embed_dim, num_heads, dropout)

# Temporal Operations
SpikeEncoding(method='rate|temporal|phase')
SpikeDecoding(method='count|rate|first_spike')
TemporalPooling(window_size, method='sum|max|avg')

# Control Flow
TimeLoop(num_steps)
ConditionalSpike(condition)
```

#### Data Types
```python
SpikeTensor(shape, dtype=spike_binary, temporal_dim=-1)
MembraneState(shape, dtype=float32)
SynapticWeights(shape, dtype=quantized, sparsity_mask)
```

#### Graph Structure
- **Directed Acyclic Graph (DAG)** representation
- **Temporal edges** for time-dependent data flow
- **Control dependencies** for spike timing constraints
- **Memory annotations** for buffer allocation

## Alternatives Considered

### 1. MLIR-based IR
**Pros**: Mature optimization infrastructure, industry adoption  
**Cons**: Limited neuromorphic primitives, complex integration  
**Decision**: Too heavyweight for initial implementation

### 2. PyTorch FX Graph
**Pros**: Native PyTorch integration, existing tooling  
**Cons**: Not designed for temporal/spiking semantics  
**Decision**: Insufficient spike-specific support

### 3. Custom AST-based IR
**Pros**: Maximum flexibility, simple implementation  
**Cons**: Limited optimization opportunities, reinventing wheels  
**Decision**: Graph-based approach provides better optimization potential

## Implementation Plan

### Phase 1: Core IR Infrastructure
```python
# IR Builder
class SpikeIRBuilder:
    def create_neuron_node(self, neuron_type, params)
    def create_spike_op(self, op_type, inputs, outputs)
    def add_temporal_edge(self, src, dst, delay=0)
    def build_graph(self) -> SpikeGraph

# Graph Representation
class SpikeGraph:
    def verify(self) -> bool
    def visualize(self, output_path: str)
    def analyze_resources(self) -> ResourceStats
    def to_dot(self) -> str
```

### Phase 2: Optimization Passes
```python
# Dead Code Elimination
class DeadSpikeElimination(Pass):
    def run(self, graph: SpikeGraph) -> SpikeGraph

# Spike Fusion
class SpikeFusion(Pass):
    def fuse_consecutive_ops(self, ops: List[SpikeOp])
    def merge_temporal_windows(self, windows: List[TimeWindow])

# Resource Allocation
class ResourceAllocator(Pass):
    def allocate_neurons(self, graph: SpikeGraph, hw_config: HardwareConfig)
    def schedule_spikes(self, graph: SpikeGraph, timing_constraints: Constraints)
```

### Phase 3: Backend Lowering
```python
# Target-specific lowering
class Loihi3Lowering(Pass):
    def lower_spike_ops(self, graph: SpikeGraph) -> Loihi3Program
    def map_resources(self, allocation: ResourceAllocation)
    def generate_code(self, program: Loihi3Program) -> str
```

## Consequences

### Positive
- **Optimizations**: Standard compiler techniques apply to spike-specific operations
- **Portability**: Single IR supports multiple neuromorphic targets
- **Debugging**: Graph visualization aids in understanding model behavior
- **Performance**: Efficient representation enables aggressive optimization

### Negative
- **Complexity**: Custom IR requires significant implementation effort
- **Learning Curve**: Developers need to understand spike-specific semantics
- **Maintenance**: IR evolution must be carefully managed for backwards compatibility

### Mitigation Strategies
- Start with minimal viable IR, expand features incrementally
- Provide comprehensive documentation and examples
- Implement strong validation and testing for IR correctness
- Design with extensibility in mind for future hardware targets

## Validation

### Success Criteria
1. **Completeness**: Can represent all SpikeFormer operations
2. **Efficiency**: Compilation time <10x slower than PyTorch JIT
3. **Correctness**: IR transformations preserve semantic equivalence
4. **Extensibility**: Easy addition of new spike operations and hardware targets

### Test Plan
- **Unit Tests**: Individual IR node creation and manipulation
- **Integration Tests**: End-to-end model conversion and optimization
- **Performance Tests**: Compilation speed benchmarks
- **Validation Tests**: Numerical equivalence with PyTorch simulation

## Future Considerations

### Potential Extensions
- **Dynamic Shapes**: Support for variable-length spike sequences
- **Probabilistic Spikes**: Stochastic neuron models and noise injection
- **Learning Rules**: Online learning and plasticity in compiled models
- **Multi-modal**: Integration with non-spiking components

### Integration Points
- **Frontend**: PyTorch → Spike IR conversion
- **Optimization**: MLIR integration for advanced passes
- **Backend**: Multiple neuromorphic hardware targets
- **Runtime**: Dynamic compilation and adaptive optimization

---

**References**
- Intel Loihi Programming Model
- TVM Tensor IR Design
- MLIR Neuromorphic Computing Dialect
- SpikeFormer Architecture Papers