"""Simulation backend for testing compiled models."""

from typing import Any, Dict, List, Optional
import numpy as np
import time
from ..ir.spike_graph import SpikeGraph, NodeType
from ..compiler import CompiledModel


class SimulationExecutor:
    """Executes compiled models in simulation."""
    
    def __init__(self, graph: SpikeGraph, debug: bool = False):
        self.graph = graph
        self.debug = debug
        self.debug_trace = []
        self.energy_consumption = 0.0
        
    def run(self, input_data: Any, time_steps: int = 4, return_spike_trains: bool = False) -> Any:
        """Run simulation."""
        self.debug_trace.clear()
        self.energy_consumption = 0.0
        
        # Initialize state
        state = self._initialize_state(input_data, time_steps)
        
        # Execute timesteps
        for t in range(time_steps):
            start_time = time.time()
            self._simulate_timestep(state, t)
            duration = (time.time() - start_time) * 1000  # ms
            
            if self.debug:
                self.debug_trace.append({
                    "step": t,
                    "operation": "timestep_simulation",
                    "duration": duration,
                    "spikes": self._count_spikes(state),
                    "memory": self._estimate_memory_usage(state)
                })
                
        # Extract output
        output = self._extract_output(state, return_spike_trains)
        
        # Estimate energy consumption
        self.energy_consumption = self._estimate_energy(state, time_steps)
        
        return output
        
    def _initialize_state(self, input_data: Any, time_steps: int) -> Dict[str, Any]:
        """Initialize simulation state."""
        state = {
            "membrane_potentials": {},
            "spike_trains": {},
            "time_steps": time_steps,
            "current_step": 0
        }
        
        # Initialize spike trains for all nodes
        for node_id, node in self.graph.nodes.items():
            shape = node.metadata.get("shape", (1,))
            if isinstance(shape, tuple) and len(shape) > 0:
                state["spike_trains"][node_id] = np.zeros((time_steps,) + shape)
                state["membrane_potentials"][node_id] = np.zeros(shape)
            else:
                state["spike_trains"][node_id] = np.zeros((time_steps, 1))
                state["membrane_potentials"][node_id] = np.zeros((1,))
                
        # Set input data
        input_nodes = [node.id for node in self.graph.nodes.values() 
                      if node.node_type == NodeType.INPUT]
        if input_nodes:
            input_id = input_nodes[0]
            # Convert input to spike train
            if isinstance(input_data, np.ndarray):
                spike_train = self._encode_to_spikes(input_data, time_steps)
                state["spike_trains"][input_id] = spike_train
                
        return state
        
    def _simulate_timestep(self, state: Dict[str, Any], timestep: int) -> None:
        """Simulate one timestep."""
        # Execute nodes in topological order
        execution_order = self.graph.topological_sort()
        
        for node_id in execution_order:
            node = self.graph.get_node(node_id)
            if node is None:
                continue
                
            # Skip input nodes (already initialized)
            if node.node_type == NodeType.INPUT:
                continue
                
            # Get input spikes
            input_spikes = self._get_input_spikes(state, node_id, timestep)
            
            # Simulate node
            output_spikes, membrane_potential = self._simulate_node(
                node, input_spikes, state["membrane_potentials"][node_id]
            )
            
            # Update state
            state["spike_trains"][node_id][timestep] = output_spikes
            state["membrane_potentials"][node_id] = membrane_potential
            
    def _get_input_spikes(self, state: Dict[str, Any], node_id: str, timestep: int) -> np.ndarray:
        """Get input spikes for a node."""
        predecessors = self.graph.get_predecessors(node_id)
        
        if not predecessors:
            return np.array([0.0])
            
        # Sum spikes from all predecessors
        total_input = None
        for pred_id in predecessors:
            pred_spikes = state["spike_trains"][pred_id][timestep]
            if total_input is None:
                total_input = pred_spikes.copy()
            else:
                total_input += pred_spikes
                
        return total_input if total_input is not None else np.array([0.0])
        
    def _simulate_node(self, node, input_spikes: np.ndarray, membrane_potential: np.ndarray) -> tuple:
        """Simulate individual node."""
        if node.node_type == NodeType.SPIKE_NEURON:
            return self._simulate_neuron(node, input_spikes, membrane_potential)
        elif node.node_type == NodeType.SPIKE_LINEAR:
            return self._simulate_linear(node, input_spikes, membrane_potential)
        elif node.node_type == NodeType.SPIKE_CONV:
            return self._simulate_convolution(node, input_spikes, membrane_potential)
        elif node.node_type == NodeType.TEMPORAL_POOL:
            return self._simulate_temporal_pooling(node, input_spikes, membrane_potential)
        else:
            # Pass through for unsupported operations
            return input_spikes, membrane_potential
            
    def _simulate_neuron(self, node, input_spikes: np.ndarray, membrane_potential: np.ndarray) -> tuple:
        """Simulate spiking neuron."""
        # LIF neuron model parameters
        tau_mem = node.get_parameter("tau_mem", 10.0)
        tau_syn = node.get_parameter("tau_syn", 5.0)
        threshold = node.get_parameter("threshold", 1.0)
        reset_mode = node.get_parameter("reset_mode", "zero")
        
        dt = 1.0  # timestep in ms
        
        # Membrane potential dynamics
        membrane_potential *= np.exp(-dt / tau_mem)
        membrane_potential += input_spikes * dt / tau_syn
        
        # Spike generation
        output_spikes = (membrane_potential >= threshold).astype(float)
        
        # Reset membrane potential
        if reset_mode == "zero":
            membrane_potential[output_spikes > 0] = 0.0
        elif reset_mode == "subtract":
            membrane_potential[output_spikes > 0] -= threshold
            
        return output_spikes, membrane_potential
        
    def _simulate_linear(self, node, input_spikes: np.ndarray, membrane_potential: np.ndarray) -> tuple:
        """Simulate linear layer with random weights."""
        in_features = node.get_parameter("in_features", input_spikes.size)
        out_features = node.get_parameter("out_features", input_spikes.size)
        
        # Use random weights for simulation
        np.random.seed(42)  # Reproducible
        weights = np.random.randn(in_features, out_features) * 0.1
        
        # Linear transformation
        input_flat = input_spikes.flatten()[:in_features]
        output = np.dot(input_flat, weights)
        
        # Reshape to expected output shape
        if len(membrane_potential.shape) > 1:
            output = output.reshape(membrane_potential.shape)
        else:
            output = output[:membrane_potential.size]
            
        return output, membrane_potential
        
    def _simulate_convolution(self, node, input_spikes: np.ndarray, membrane_potential: np.ndarray) -> tuple:
        """Simulate convolution (simplified)."""
        kernel_size = node.get_parameter("kernel_size", 3)
        stride = node.get_parameter("stride", 1)
        
        # Simplified convolution - apply average pooling
        if len(input_spikes.shape) >= 2:
            h, w = input_spikes.shape[-2:]
            out_h = (h - kernel_size) // stride + 1
            out_w = (w - kernel_size) // stride + 1
            
            output = np.zeros((out_h, out_w))
            for i in range(out_h):
                for j in range(out_w):
                    window = input_spikes[i*stride:i*stride+kernel_size, 
                                        j*stride:j*stride+kernel_size]
                    output[i, j] = np.mean(window)
        else:
            output = input_spikes * 0.5  # Simple scaling
            
        return output, membrane_potential
        
    def _simulate_temporal_pooling(self, node, input_spikes: np.ndarray, membrane_potential: np.ndarray) -> tuple:
        """Simulate temporal pooling."""
        method = node.get_parameter("method", "sum")
        
        if method == "sum":
            output = np.sum(input_spikes)
        elif method == "max":
            output = np.max(input_spikes)
        elif method == "avg":
            output = np.mean(input_spikes)
        else:
            output = input_spikes
            
        # Ensure output has same shape as membrane potential
        if np.isscalar(output):
            output = np.full_like(membrane_potential, output)
        
        return output, membrane_potential
        
    def _encode_to_spikes(self, data: np.ndarray, time_steps: int) -> np.ndarray:
        """Encode continuous data to spike trains using rate coding."""
        # Normalize data to [0, 1]
        data_normalized = np.clip(data, 0, 1)
        
        # Create spike trains
        spike_train = np.zeros((time_steps,) + data.shape)
        
        for t in range(time_steps):
            # Poisson spike generation
            random_vals = np.random.random(data.shape)
            spike_train[t] = (random_vals < data_normalized).astype(float)
            
        return spike_train
        
    def _extract_output(self, state: Dict[str, Any], return_spike_trains: bool) -> Any:
        """Extract output from simulation state."""
        output_nodes = [node.id for node in self.graph.nodes.values() 
                       if node.node_type == NodeType.OUTPUT]
        
        if not output_nodes:
            return None
            
        output_id = output_nodes[0]
        
        if return_spike_trains:
            return state["spike_trains"][output_id]
        else:
            # Return spike count over time
            return np.sum(state["spike_trains"][output_id], axis=0)
            
    def _count_spikes(self, state: Dict[str, Any]) -> int:
        """Count total spikes in current state."""
        total_spikes = 0
        for node_id, spike_train in state["spike_trains"].items():
            if len(spike_train.shape) > 0:
                total_spikes += np.sum(spike_train)
        return int(total_spikes)
        
    def _estimate_memory_usage(self, state: Dict[str, Any]) -> int:
        """Estimate memory usage in bytes."""
        memory = 0
        for node_id, potential in state["membrane_potentials"].items():
            memory += potential.nbytes
        for node_id, spikes in state["spike_trains"].items():
            memory += spikes.nbytes
        return memory
        
    def _estimate_energy(self, state: Dict[str, Any], time_steps: int) -> float:
        """Estimate energy consumption in nJ."""
        total_spikes = self._count_spikes(state)
        total_neurons = len(state["membrane_potentials"])
        
        # Simple energy model: 1 pJ per spike, 0.1 pJ per neuron per timestep
        spike_energy = total_spikes * 1.0  # pJ
        static_energy = total_neurons * time_steps * 0.1  # pJ
        
        return (spike_energy + static_energy) / 1000.0  # Convert to nJ
        
    def get_energy_consumption(self) -> float:
        """Get total energy consumption."""
        return self.energy_consumption
        
    def get_debug_trace(self) -> List[Dict[str, Any]]:
        """Get debug execution trace."""
        return self.debug_trace.copy()


class SimulationBackend:
    """Simulation backend for testing compiled models."""
    
    def __init__(self, resource_allocator=None):
        self.resource_allocator = resource_allocator
        
    def compile_graph(self, graph: SpikeGraph, profile_energy: bool = False, debug: bool = False) -> CompiledModel:
        """Compile graph for simulation."""
        # Create executor
        executor = SimulationExecutor(graph, debug=debug)
        
        # Create compiled model
        compiled_model = CompiledModel()
        compiled_model.executor = executor
        
        # Estimate utilization (simulation always has 100% availability)
        compiled_model.utilization = 1.0
        
        return compiled_model