"""Spike computation graph representation."""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from .types import SpikeTensor, MembraneState, SynapticWeights


class NodeType(Enum):
    """Types of nodes in the spike graph."""
    INPUT = "input"
    OUTPUT = "output"
    SPIKE_NEURON = "spike_neuron"
    SPIKE_CONV = "spike_conv"
    SPIKE_LINEAR = "spike_linear"
    SPIKE_ATTENTION = "spike_attention"
    SPIKE_ENCODING = "spike_encoding"
    SPIKE_DECODING = "spike_decoding"
    TEMPORAL_POOL = "temporal_pool"
    TIME_LOOP = "time_loop"


@dataclass
class SpikeNode:
    """Node in the spike computation graph."""
    
    id: str
    node_type: NodeType
    operation: str
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if "shape" not in self.metadata:
            self.metadata["shape"] = None
        if "memory_estimate" not in self.metadata:
            self.metadata["memory_estimate"] = 0
            
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)
        
    def set_parameter(self, key: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[key] = value
        
    def estimate_computation(self) -> float:
        """Estimate computational cost (in operations)."""
        if self.node_type == NodeType.SPIKE_CONV:
            # Convolution: output_size * kernel_size * channels
            out_h = self.get_parameter("output_height", 1)
            out_w = self.get_parameter("output_width", 1)
            kernel_size = self.get_parameter("kernel_size", 3)
            in_channels = self.get_parameter("in_channels", 1)
            out_channels = self.get_parameter("out_channels", 1)
            return out_h * out_w * kernel_size * kernel_size * in_channels * out_channels
            
        elif self.node_type == NodeType.SPIKE_LINEAR:
            # Linear: input_size * output_size
            in_features = self.get_parameter("in_features", 1)
            out_features = self.get_parameter("out_features", 1)
            return in_features * out_features
            
        elif self.node_type == NodeType.SPIKE_ATTENTION:
            # Attention: sequence_length^2 * embed_dim
            seq_len = self.get_parameter("sequence_length", 1)
            embed_dim = self.get_parameter("embed_dim", 1)
            return seq_len * seq_len * embed_dim
            
        else:
            return 1.0  # Minimal cost for other operations


@dataclass
class SpikeEdge:
    """Edge in the spike computation graph."""
    
    source: str
    target: str
    data_type: Any  # SpikeTensor, MembraneState, etc.
    delay: int = 0  # Temporal delay in time steps
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def get_bandwidth_requirement(self) -> float:
        """Estimate bandwidth requirement in MB/s."""
        if isinstance(self.data_type, SpikeTensor):
            memory_per_timestep = self.data_type.estimate_memory()
            # Assume 1000 Hz spike rate
            return memory_per_timestep * 1000 / (1024 * 1024)
        return 0.0


class SpikeGraph:
    """Spike computation graph."""
    
    def __init__(self, name: str = "spike_graph"):
        self.name = name
        self.nodes: Dict[str, SpikeNode] = {}
        self.edges: List[SpikeEdge] = []
        self.metadata: Dict[str, Any] = {
            "time_steps": 1,
            "spike_encoding": "rate",
            "hardware_target": "simulation"
        }
        
    def add_node(self, node: SpikeNode) -> None:
        """Add a node to the graph."""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists in graph")
        self.nodes[node.id] = node
        
    def add_edge(self, edge: SpikeEdge) -> None:
        """Add an edge to the graph."""
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} not found")
        self.edges.append(edge)
        
    def get_node(self, node_id: str) -> Optional[SpikeNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
        
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor nodes."""
        return [edge.source for edge in self.edges if edge.target == node_id]
        
    def get_successors(self, node_id: str) -> List[str]:
        """Get successor nodes."""
        return [edge.target for edge in self.edges if edge.source == node_id]
        
    def topological_sort(self) -> List[str]:
        """Get topologically sorted node order."""
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1
            
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for successor in self.get_successors(node_id):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
                    
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")
            
        return result
        
    def verify(self) -> bool:
        """Verify graph correctness."""
        try:
            # Check for cycles
            self.topological_sort()
            
            # Check edge validity
            for edge in self.edges:
                if edge.source not in self.nodes:
                    return False
                if edge.target not in self.nodes:
                    return False
                    
            # Check node connectivity
            for node_id, node in self.nodes.items():
                # Verify input connections
                for input_name in node.inputs:
                    found = False
                    for edge in self.edges:
                        if edge.target == node_id and edge.source in self.nodes:
                            found = True
                            break
                    if not found and node.node_type != NodeType.INPUT:
                        return False
                        
            return True
            
        except Exception:
            return False
            
    def analyze_resources(self) -> Dict[str, Any]:
        """Analyze resource requirements."""
        total_memory = 0
        total_computation = 0
        neuron_count = 0
        synapse_count = 0
        
        for node in self.nodes.values():
            total_computation += node.estimate_computation()
            total_memory += node.metadata.get("memory_estimate", 0)
            
            if node.node_type in [NodeType.SPIKE_NEURON, NodeType.SPIKE_CONV, 
                                  NodeType.SPIKE_LINEAR, NodeType.SPIKE_ATTENTION]:
                neuron_count += node.get_parameter("num_neurons", 1)
                
        for edge in self.edges:
            if isinstance(edge.data_type, SynapticWeights):
                synapse_count += edge.data_type.shape[0] * edge.data_type.shape[1]
                
        return {
            "total_memory_bytes": total_memory,
            "total_computation_ops": total_computation,
            "neuron_count": neuron_count,
            "synapse_count": synapse_count,
            "sparsity_ratio": self._calculate_sparsity(),
            "critical_path_length": self._calculate_critical_path()
        }
        
    def _calculate_sparsity(self) -> float:
        """Calculate average sparsity across the graph."""
        total_weights = 0
        sparse_weights = 0
        
        for edge in self.edges:
            if isinstance(edge.data_type, SynapticWeights):
                weight_count = edge.data_type.shape[0] * edge.data_type.shape[1]
                total_weights += weight_count
                sparse_weights += weight_count * edge.data_type.sparsity
                
        return sparse_weights / total_weights if total_weights > 0 else 0.0
        
    def _calculate_critical_path(self) -> int:
        """Calculate critical path length through the graph."""
        # Simple implementation: longest path from input to output
        sorted_nodes = self.topological_sort()
        distances = {node_id: 0 for node_id in self.nodes}
        
        for node_id in sorted_nodes:
            for successor in self.get_successors(node_id):
                distances[successor] = max(distances[successor], distances[node_id] + 1)
                
        return max(distances.values()) if distances else 0
        
    def visualize(self, output_path: str) -> None:
        """Generate visualization of the graph."""
        try:
            import graphviz
            
            dot = graphviz.Digraph(comment=self.name)
            dot.attr(rankdir='TB')
            
            # Add nodes
            for node_id, node in self.nodes.items():
                label = f"{node.operation}\\n{node_id}"
                if node.node_type == NodeType.INPUT:
                    dot.node(node_id, label, shape='box', style='filled', fillcolor='lightgreen')
                elif node.node_type == NodeType.OUTPUT:
                    dot.node(node_id, label, shape='box', style='filled', fillcolor='lightcoral')
                else:
                    dot.node(node_id, label, shape='ellipse')
                    
            # Add edges
            for edge in self.edges:
                label = ""
                if edge.delay > 0:
                    label = f"delay={edge.delay}"
                dot.edge(edge.source, edge.target, label=label)
                
            dot.render(output_path, format='pdf', cleanup=True)
            
        except ImportError:
            # Generate simple text representation
            with open(f"{output_path}.txt", "w") as f:
                f.write(f"Graph: {self.name}\n")
                f.write("Nodes:\n")
                for node_id, node in self.nodes.items():
                    f.write(f"  {node_id}: {node.operation} ({node.node_type.value})\n")
                f.write("Edges:\n")
                for edge in self.edges:
                    f.write(f"  {edge.source} -> {edge.target}")
                    if edge.delay > 0:
                        f.write(f" (delay={edge.delay})")
                    f.write("\n")
                    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "nodes": {
                node_id: {
                    "node_type": node.node_type.value,
                    "operation": node.operation,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "parameters": node.parameters,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "delay": edge.delay,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ]
        }
        
    def save(self, file_path: str) -> None:
        """Save graph to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)