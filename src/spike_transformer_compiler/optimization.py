"""Optimization passes for spike-based neural networks."""

from enum import Enum
from typing import Any


class OptimizationPass(Enum):
    """Available optimization passes for neuromorphic compilation."""
    
    SPIKE_COMPRESSION = "spike_compression"
    WEIGHT_QUANTIZATION = "weight_quantization" 
    NEURON_PRUNING = "neuron_pruning"
    TEMPORAL_FUSION = "temporal_fusion"


class Optimizer:
    """Optimization pipeline for neuromorphic models."""
    
    def __init__(self):
        self.passes = []
        
    def add_pass(self, pass_type: OptimizationPass, **kwargs: Any) -> None:
        """Add an optimization pass to the pipeline."""
        self.passes.append((pass_type, kwargs))
        
    def run(self, ir: Any) -> Any:
        """Run all optimization passes on the IR."""
        optimized_ir = ir
        for pass_type, kwargs in self.passes:
            optimized_ir = self._run_pass(pass_type, optimized_ir, **kwargs)
        return optimized_ir
        
    def _run_pass(self, pass_type: OptimizationPass, ir: Any, **kwargs: Any) -> Any:
        """Run a single optimization pass."""
        from .ir.spike_graph import SpikeGraph
        from .ir.types import SynapticWeights
        import numpy as np
        
        if not isinstance(ir, SpikeGraph):
            return ir
            
        if pass_type == OptimizationPass.SPIKE_COMPRESSION:
            return self._compress_spikes(ir, **kwargs)
        elif pass_type == OptimizationPass.WEIGHT_QUANTIZATION:
            return self._quantize_weights(ir, **kwargs)
        elif pass_type == OptimizationPass.NEURON_PRUNING:
            return self._prune_neurons(ir, **kwargs)
        elif pass_type == OptimizationPass.TEMPORAL_FUSION:
            return self._fuse_temporal_ops(ir, **kwargs)
        else:
            return ir
            
    def _compress_spikes(self, graph: SpikeGraph, compression_ratio: float = 0.1) -> SpikeGraph:
        """Apply spike compression to reduce data movement."""
        # Update metadata to indicate spike compression
        graph.metadata["spike_compression"] = compression_ratio
        
        # Mark nodes for spike compression
        for node in graph.nodes.values():
            if "spike" in node.operation.lower():
                node.metadata["compressed"] = True
                node.metadata["compression_ratio"] = compression_ratio
                
        return graph
        
    def _quantize_weights(self, graph: SpikeGraph, bits: int = 8) -> SpikeGraph:
        """Apply weight quantization to reduce memory usage."""
        # Update synaptic weights in edges
        for edge in graph.edges:
            if isinstance(edge.data_type, SynapticWeights):
                edge.data_type.quantized = True
                edge.data_type.dtype = f"int{bits}"
                
        graph.metadata["weight_quantization"] = f"{bits}bit"
        return graph
        
    def _prune_neurons(self, graph: SpikeGraph, sparsity: float = 0.9) -> SpikeGraph:
        """Apply neuron pruning to reduce computational complexity."""
        # Mark neurons for pruning
        for node in graph.nodes.values():
            if "neuron" in node.operation.lower():
                node.metadata["pruned"] = True
                node.metadata["sparsity"] = sparsity
                
                # Reduce neuron count
                if "num_neurons" in node.parameters:
                    original_count = node.parameters["num_neurons"]
                    node.parameters["num_neurons"] = int(original_count * (1 - sparsity))
                    
        graph.metadata["neuron_pruning"] = sparsity
        return graph
        
    def _fuse_temporal_ops(self, graph: SpikeGraph, **kwargs) -> SpikeGraph:
        """Fuse temporal operations for efficiency."""
        # Mark temporal operations for fusion
        temporal_nodes = []
        for node in graph.nodes.values():
            if "temporal" in node.operation.lower():
                temporal_nodes.append(node.id)
                
        if len(temporal_nodes) > 1:
            graph.metadata["temporal_fusion"] = temporal_nodes
            
        return graph