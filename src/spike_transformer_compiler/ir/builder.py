"""Builder for constructing Spike IR graphs."""

from typing import Dict, List, Any, Optional, Tuple
import uuid
from .spike_graph import SpikeGraph, SpikeNode, SpikeEdge, NodeType
from .types import SpikeTensor, MembraneState, SynapticWeights, SpikeType


class SpikeIRBuilder:
    """Builder for constructing Spike IR graphs."""
    
    def __init__(self, graph_name: str = "spike_model"):
        self.graph = SpikeGraph(graph_name)
        self._node_counter = 0
        
    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate unique node ID."""
        node_id = f"{prefix}_{self._node_counter}"
        self._node_counter += 1
        return node_id
        
    def add_input(
        self,
        name: str,
        shape: Tuple[int, ...],
        spike_type: SpikeType = SpikeType.BINARY,
        node_id: Optional[str] = None
    ) -> str:
        """Add input node to the graph."""
        if node_id is None:
            node_id = self._generate_node_id("input")
            
        spike_tensor = SpikeTensor(shape, spike_type)
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.INPUT,
            operation=f"input_{name}",
            inputs=[],
            outputs=[node_id],
            parameters={"name": name, "shape": shape},
            metadata={"shape": shape, "data_type": spike_tensor}
        )
        
        self.graph.add_node(node)
        return node_id
        
    def add_output(
        self,
        input_node: str,
        name: str,
        node_id: Optional[str] = None
    ) -> str:
        """Add output node to the graph."""
        if node_id is None:
            node_id = self._generate_node_id("output")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.OUTPUT,
            operation=f"output_{name}",
            inputs=[input_node],
            outputs=[],
            parameters={"name": name},
            metadata={"shape": input_shape}
        )
        
        self.graph.add_node(node)
        
        # Add edge from input to output
        edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=SpikeTensor(input_shape)
        )
        self.graph.add_edge(edge)
        
        return node_id
        
    def add_spike_neuron(
        self,
        input_node: str,
        neuron_model: str = "LIF",
        threshold: float = 1.0,
        reset_mode: str = "zero",
        tau_mem: float = 10.0,
        tau_syn: float = 5.0,
        node_id: Optional[str] = None
    ) -> str:
        """Add spiking neuron layer."""
        if node_id is None:
            node_id = self._generate_node_id("neuron")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.SPIKE_NEURON,
            operation=f"spike_{neuron_model.lower()}",
            inputs=[input_node],
            outputs=[node_id],
            parameters={
                "neuron_model": neuron_model,
                "threshold": threshold,
                "reset_mode": reset_mode,
                "tau_mem": tau_mem,
                "tau_syn": tau_syn,
                "num_neurons": input_shape[-1] if input_shape else 1
            },
            metadata={"shape": input_shape}
        )
        
        self.graph.add_node(node)
        
        # Add edges
        spike_edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=SpikeTensor(input_shape, SpikeType.BINARY)
        )
        self.graph.add_edge(spike_edge)
        
        return node_id
        
    def add_spike_conv2d(
        self,
        input_node: str,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        node_id: Optional[str] = None
    ) -> str:
        """Add spike convolution layer."""
        if node_id is None:
            node_id = self._generate_node_id("conv")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        in_channels = input_shape[1] if len(input_shape) >= 3 else 1
        
        # Calculate output shape
        if len(input_shape) >= 3:
            out_h = (input_shape[2] + 2 * padding - kernel_size) // stride + 1
            out_w = (input_shape[3] + 2 * padding - kernel_size) // stride + 1
            output_shape = (input_shape[0], out_channels, out_h, out_w)
        else:
            output_shape = (input_shape[0], out_channels)
            
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.SPIKE_CONV,
            operation="spike_conv2d",
            inputs=[input_node],
            outputs=[node_id],
            parameters={
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "bias": bias,
                "output_height": output_shape[2] if len(output_shape) >= 3 else 1,
                "output_width": output_shape[3] if len(output_shape) >= 4 else 1
            },
            metadata={"shape": output_shape}
        )
        
        self.graph.add_node(node)
        
        # Add data edge
        data_edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=SpikeTensor(input_shape, SpikeType.BINARY)
        )
        self.graph.add_edge(data_edge)
        
        return node_id
        
    def add_spike_linear(
        self,
        input_node: str,
        out_features: int,
        bias: bool = True,
        node_id: Optional[str] = None
    ) -> str:
        """Add spike linear/dense layer."""
        if node_id is None:
            node_id = self._generate_node_id("linear")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        in_features = input_shape[-1] if input_shape else 1
        
        # Output shape: batch_size x out_features
        output_shape = input_shape[:-1] + (out_features,)
        
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.SPIKE_LINEAR,
            operation="spike_linear",
            inputs=[input_node],
            outputs=[node_id],
            parameters={
                "in_features": in_features,
                "out_features": out_features,
                "bias": bias
            },
            metadata={"shape": output_shape}
        )
        
        self.graph.add_node(node)
        
        # Add data edge with synaptic weights
        weights = SynapticWeights(
            shape=(in_features, out_features),
            dtype="float32",
            sparsity=0.0
        )
        
        data_edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=weights
        )
        self.graph.add_edge(data_edge)
        
        return node_id
        
    def add_spike_attention(
        self,
        input_node: str,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        node_id: Optional[str] = None
    ) -> str:
        """Add spike attention layer."""
        if node_id is None:
            node_id = self._generate_node_id("attention")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.SPIKE_ATTENTION,
            operation="spike_attention",
            inputs=[input_node],
            outputs=[node_id],
            parameters={
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "dropout": dropout,
                "sequence_length": input_shape[1] if len(input_shape) > 1 else 1
            },
            metadata={"shape": input_shape}
        )
        
        self.graph.add_node(node)
        
        # Add data edge
        data_edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=SpikeTensor(input_shape, SpikeType.BINARY)
        )
        self.graph.add_edge(data_edge)
        
        return node_id
        
    def add_spike_encoding(
        self,
        input_node: str,
        encoding_method: str = "rate",
        time_steps: int = 4,
        node_id: Optional[str] = None
    ) -> str:
        """Add spike encoding layer."""
        if node_id is None:
            node_id = self._generate_node_id("encoding")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        
        # Add temporal dimension
        output_shape = (time_steps,) + input_shape
        
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.SPIKE_ENCODING,
            operation=f"spike_encoding_{encoding_method}",
            inputs=[input_node],
            outputs=[node_id],
            parameters={
                "method": encoding_method,
                "time_steps": time_steps
            },
            metadata={"shape": output_shape}
        )
        
        self.graph.add_node(node)
        
        # Update graph metadata
        self.graph.metadata["time_steps"] = time_steps
        self.graph.metadata["spike_encoding"] = encoding_method
        
        # Add data edge
        spike_type = {
            "rate": SpikeType.BINARY,
            "temporal": SpikeType.TEMPORAL,
            "phase": SpikeType.PHASE
        }.get(encoding_method, SpikeType.BINARY)
        
        data_edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=SpikeTensor(output_shape, spike_type, temporal_dim=0)
        )
        self.graph.add_edge(data_edge)
        
        return node_id
        
    def add_temporal_pooling(
        self,
        input_node: str,
        window_size: int = 4,
        method: str = "sum",
        node_id: Optional[str] = None
    ) -> str:
        """Add temporal pooling operation."""
        if node_id is None:
            node_id = self._generate_node_id("temporal_pool")
            
        input_shape = self.graph.get_node(input_node).metadata["shape"]
        
        # Reduce temporal dimension
        if len(input_shape) > 0 and input_shape[0] > 1:
            output_shape = (max(1, input_shape[0] // window_size),) + input_shape[1:]
        else:
            output_shape = input_shape
            
        node = SpikeNode(
            id=node_id,
            node_type=NodeType.TEMPORAL_POOL,
            operation=f"temporal_{method}_pool",
            inputs=[input_node],
            outputs=[node_id],
            parameters={
                "window_size": window_size,
                "method": method
            },
            metadata={"shape": output_shape}
        )
        
        self.graph.add_node(node)
        
        # Add data edge
        data_edge = SpikeEdge(
            source=input_node,
            target=node_id,
            data_type=SpikeTensor(output_shape, SpikeType.BINARY)
        )
        self.graph.add_edge(data_edge)
        
        return node_id
        
    def set_graph_metadata(self, **kwargs) -> None:
        """Set graph metadata."""
        self.graph.metadata.update(kwargs)
        
    def build(self) -> SpikeGraph:
        """Build and return the completed graph."""
        if not self.graph.verify():
            raise ValueError("Graph verification failed. Check node connections and cycles.")
            
        return self.graph
        
    def reset(self) -> None:
        """Reset builder for new graph construction."""
        self.graph = SpikeGraph(self.graph.name)
        self._node_counter = 0