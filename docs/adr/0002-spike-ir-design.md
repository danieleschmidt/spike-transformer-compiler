# ADR-0002: Spike Intermediate Representation Design

**Status**: Accepted  
**Date**: 2025-08-02  
**Decision makers**: Core Development Team, Neuromorphic Computing Experts  

## Context and Problem Statement

The Spike-Transformer-Compiler requires an intermediate representation (IR) that can effectively model spiking neural network operations, temporal dynamics, and neuromorphic hardware constraints. The IR must support optimization passes while maintaining semantic correctness for spike-based computations.

## Decision Drivers

- Need to represent temporal spike dynamics and timing constraints
- Support for various neuromorphic hardware architectures (Loihi 3, future targets)
- Enable aggressive optimization passes specific to spiking computations
- Maintain compatibility with traditional deep learning frameworks
- Support for multi-timestep operations and spike encoding schemes

## Considered Options

1. **Extend existing IR (MLIR/XLA)**: Adapt existing compiler IR with spiking-specific operations
2. **Custom Spike IR**: Design purpose-built IR for neuromorphic operations
3. **Graph-based representation**: Use computation graphs with spike-specific metadata
4. **Hybrid approach**: Combine graph representation with spike-specific IR constructs

## Decision Outcome

**Chosen option**: Custom Spike IR with graph-based foundation

The Spike IR will be built as a custom intermediate representation that combines:
- Graph-based computation representation for structural operations
- Spike-specific temporal constructs for timing and dynamics
- Hardware abstraction layer for target-specific optimizations

### Core Components

```python
# Spike IR Node Types
class SpikeNode:
    - neuron_groups: Collections of spiking neurons
    - synapse_groups: Synaptic connections with weights
    - temporal_operations: Time-dependent spike operations
    - memory_operations: Spike buffer and state management

# Temporal Semantics
class TemporalSemantics:
    - time_steps: Discrete time progression
    - spike_trains: Temporal spike sequences
    - refractory_periods: Neuron recovery times
    - synaptic_delays: Connection timing constraints
```

### Positive Consequences

- Native support for spiking neural network semantics
- Optimized for neuromorphic hardware compilation
- Enables spike-specific optimization passes (temporal fusion, spike compression)
- Clear separation between computation and temporal concerns
- Extensible to future neuromorphic architectures

### Negative Consequences

- Additional complexity compared to reusing existing IR
- Need to implement custom optimization passes
- Potential learning curve for developers familiar with traditional IRs
- Maintenance overhead for custom IR implementation

## Implementation

### Phase 1: Core IR Infrastructure
- Basic node types and graph representation
- Temporal semantics and spike train modeling
- IR construction and validation

### Phase 2: Optimization Framework
- Pass manager for spike-specific optimizations
- Dead code elimination for sparse spike patterns
- Temporal operation fusion and scheduling

### Phase 3: Hardware Backend Interface
- Target-specific lowering from Spike IR
- Resource allocation and mapping
- Hardware constraint validation

## Links

- [MLIR Documentation](https://mlir.llvm.org/)
- [Neuromorphic Computing Principles](https://www.nature.com/articles/s41586-021-03748-0)
- Related: ADR-0003 (Temporal Optimization Strategy)