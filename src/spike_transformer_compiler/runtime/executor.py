"""High-performance runtime executor for neuromorphic models."""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from ..ir.spike_graph import SpikeGraph
from ..exceptions import RuntimeError as SpikeRuntimeError, ErrorContext
from ..logging_config import runtime_logger
from .memory import MemoryManager, SpikeBufferManager
from .communication import MultiChipCommunicator


class ExecutionEngine:
    """Core execution engine for neuromorphic computations."""
    
    def __init__(
        self,
        num_cores: int = 1,
        memory_manager: Optional[MemoryManager] = None,
        enable_parallel: bool = True,
        debug: bool = False
    ):
        self.num_cores = num_cores
        self.memory_manager = memory_manager or MemoryManager()
        self.enable_parallel = enable_parallel
        self.debug = debug
        
        # Runtime state
        self.execution_state = {}
        self.performance_stats = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=num_cores) if enable_parallel else None
        
        # Debugging and monitoring
        self.execution_trace = []
        self.error_count = 0
        self.recovery_attempts = 0
        
        runtime_logger.info(f"ExecutionEngine initialized with {num_cores} cores")
        
    def execute_graph(
        self,
        graph: SpikeGraph,
        input_data: Dict[str, np.ndarray],
        time_steps: int = 4,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """Execute spike graph with comprehensive error handling."""
        execution_id = f"exec_{int(time.time() * 1000)}"
        
        with ErrorContext("graph_execution", execution_id=execution_id):
            try:
                # Initialize execution
                self._initialize_execution(graph, input_data, execution_id)
                
                # Execute with monitoring
                results = self._execute_with_monitoring(
                    graph, input_data, time_steps, return_intermediate, execution_id
                )
                
                # Post-process results
                processed_results = self._post_process_results(results, execution_id)
                
                runtime_logger.info(f"Execution {execution_id} completed successfully")
                return processed_results
                
            except Exception as e:
                self.error_count += 1
                runtime_logger.error(f"Execution {execution_id} failed: {str(e)}")
                
                # Attempt recovery
                if self.recovery_attempts < 3:
                    runtime_logger.info(f"Attempting recovery for execution {execution_id}")
                    return self._attempt_recovery(graph, input_data, time_steps, e)
                else:
                    raise SpikeRuntimeError(
                        f"Execution failed after {self.recovery_attempts} recovery attempts: {str(e)}",
                        error_code="EXECUTION_FAILED",
                        details={"execution_id": execution_id, "error_count": self.error_count}
                    ) from e
                    
    def _initialize_execution(
        self,
        graph: SpikeGraph,
        input_data: Dict[str, np.ndarray],
        execution_id: str
    ) -> None:
        """Initialize execution state with validation."""
        # Validate inputs
        self._validate_execution_inputs(graph, input_data)
        
        # Initialize memory for graph
        self.memory_manager.allocate_graph_memory(graph)
        
        # Set up execution state
        self.execution_state[execution_id] = {
            "graph": graph,
            "input_data": input_data,
            "node_states": {},
            "spike_buffers": {},
            "start_time": time.time(),
            "current_timestep": 0
        }
        
        # Initialize node states
        for node in graph.nodes:
            self.execution_state[execution_id]["node_states"][node.id] = {
                "membrane_potential": np.zeros(self._get_node_size(node)),
                "spike_history": [],
                "last_spike_time": -1,
                "activation_count": 0
            }
            
        runtime_logger.debug(f"Execution {execution_id} initialized with {len(graph.nodes)} nodes")
        
    def _execute_with_monitoring(
        self,
        graph: SpikeGraph,
        input_data: Dict[str, np.ndarray],
        time_steps: int,
        return_intermediate: bool,
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute graph with comprehensive monitoring."""
        results = {"outputs": {}, "performance": {}, "intermediate": {} if return_intermediate else None}
        
        start_time = time.time()
        
        for t in range(time_steps):
            timestep_start = time.time()
            
            if self.debug:
                runtime_logger.debug(f"Executing timestep {t} for {execution_id}")
                
            # Execute timestep
            timestep_results = self._execute_timestep(graph, t, execution_id)
            
            # Collect intermediate results if requested
            if return_intermediate:
                results["intermediate"][t] = timestep_results.copy()
                
            # Update performance metrics
            timestep_duration = time.time() - timestep_start
            results["performance"][f"timestep_{t}_duration"] = timestep_duration
            
            # Memory and health checks
            self._perform_health_checks(execution_id, t)
            
            # Update execution trace
            if self.debug:
                self.execution_trace.append({
                    "execution_id": execution_id,
                    "timestep": t,
                    "duration": timestep_duration,
                    "active_nodes": len([n for n in graph.nodes if self._is_node_active(n.id, execution_id)]),
                    "memory_usage": self.memory_manager.get_current_usage()
                })
                
        # Collect final outputs
        results["outputs"] = self._collect_outputs(graph, execution_id)
        results["performance"]["total_duration"] = time.time() - start_time
        results["performance"]["avg_timestep_duration"] = results["performance"]["total_duration"] / time_steps
        
        return results
        
    def _execute_timestep(
        self,
        graph: SpikeGraph,
        timestep: int,
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute a single timestep with parallel processing."""
        timestep_results = {"spikes": {}, "potentials": {}}
        
        # Topological execution order
        execution_order = self._get_execution_order(graph)
        
        if self.enable_parallel and self.thread_pool:
            # Parallel execution for independent nodes
            futures = {}
            for node_id in execution_order:
                if self._can_execute_parallel(node_id, graph):
                    future = self.thread_pool.submit(
                        self._execute_node, node_id, timestep, execution_id
                    )
                    futures[node_id] = future
                else:
                    # Execute sequentially for dependent nodes
                    node_result = self._execute_node(node_id, timestep, execution_id)
                    timestep_results["spikes"][node_id] = node_result["spikes"]
                    timestep_results["potentials"][node_id] = node_result["potential"]
                    
            # Collect parallel results
            for node_id, future in futures.items():
                try:
                    node_result = future.result(timeout=5.0)  # 5 second timeout
                    timestep_results["spikes"][node_id] = node_result["spikes"]
                    timestep_results["potentials"][node_id] = node_result["potential"]
                except Exception as e:
                    runtime_logger.error(f"Node {node_id} execution failed: {str(e)}")
                    # Provide default result to continue execution
                    timestep_results["spikes"][node_id] = np.zeros(self._get_node_size_by_id(node_id, graph))
                    timestep_results["potentials"][node_id] = np.zeros(self._get_node_size_by_id(node_id, graph))
                    
        else:
            # Sequential execution
            for node_id in execution_order:
                try:
                    node_result = self._execute_node(node_id, timestep, execution_id)
                    timestep_results["spikes"][node_id] = node_result["spikes"]
                    timestep_results["potentials"][node_id] = node_result["potential"]
                except Exception as e:
                    runtime_logger.error(f"Sequential execution failed for node {node_id}: {str(e)}")
                    raise
                    
        return timestep_results
        
    def _execute_node(
        self,
        node_id: str,
        timestep: int,
        execution_id: str
    ) -> Dict[str, np.ndarray]:
        """Execute a single node with error handling."""
        try:
            # Get node and its current state
            graph = self.execution_state[execution_id]["graph"]
            node = graph.get_node(node_id)
            node_state = self.execution_state[execution_id]["node_states"][node_id]
            
            # Collect inputs from predecessor nodes
            input_spikes = self._collect_node_inputs(node, timestep, execution_id)
            
            # Execute node computation
            if node.node_type.name == "INPUT":
                output_spikes, membrane_potential = self._execute_input_node(node, timestep, execution_id)
            elif node.node_type.name == "SPIKE_NEURON":
                output_spikes, membrane_potential = self._execute_spike_neuron(
                    node, input_spikes, node_state, timestep
                )
            elif node.node_type.name == "SPIKE_CONV":
                output_spikes, membrane_potential = self._execute_spike_conv(
                    node, input_spikes, node_state, timestep
                )
            elif node.node_type.name == "SPIKE_ATTENTION":
                output_spikes, membrane_potential = self._execute_spike_attention(
                    node, input_spikes, node_state, timestep
                )
            else:
                # Generic node execution
                output_spikes, membrane_potential = self._execute_generic_node(
                    node, input_spikes, node_state, timestep
                )
                
            # Update node state
            node_state["membrane_potential"] = membrane_potential
            node_state["spike_history"].append(output_spikes.copy())
            node_state["activation_count"] += np.sum(output_spikes > 0)
            
            # Limit spike history size
            if len(node_state["spike_history"]) > 10:
                node_state["spike_history"].pop(0)
                
            return {
                "spikes": output_spikes,
                "potential": membrane_potential
            }
            
        except Exception as e:
            runtime_logger.error(f"Node execution failed for {node_id}: {str(e)}")
            # Return safe default values
            node_size = self._get_node_size_by_id(node_id, self.execution_state[execution_id]["graph"])
            return {
                "spikes": np.zeros(node_size),
                "potential": np.zeros(node_size)
            }
            
    def _execute_spike_neuron(
        self,
        node: Any,
        input_spikes: np.ndarray,
        node_state: Dict[str, Any],
        timestep: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute spiking neuron with LIF dynamics."""
        # Get neuron parameters
        threshold = node.parameters.get("threshold", 1.0)
        tau_mem = node.parameters.get("tau_mem", 10.0)
        tau_syn = node.parameters.get("tau_syn", 5.0)
        reset_mode = node.parameters.get("reset_mode", "zero")
        
        # Current membrane potential
        membrane_v = node_state["membrane_potential"]
        
        # Synaptic current integration
        dt = 1.0  # ms
        decay_mem = np.exp(-dt / tau_mem)
        decay_syn = np.exp(-dt / tau_syn)
        
        # Update membrane potential with LIF dynamics
        membrane_v = membrane_v * decay_mem + np.sum(input_spikes) * (1 - decay_syn)
        
        # Generate spikes
        spike_mask = membrane_v >= threshold
        output_spikes = spike_mask.astype(np.float32)
        
        # Reset membrane potential
        if reset_mode == "zero":
            membrane_v[spike_mask] = 0.0
        elif reset_mode == "subtract":
            membrane_v[spike_mask] -= threshold
            
        return output_spikes, membrane_v
        
    def _execute_input_node(
        self,
        node: Any,
        timestep: int,
        execution_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute input node by providing external data."""
        input_data = self.execution_state[execution_id]["input_data"]
        node_name = node.parameters.get("name", "default")
        
        if node_name in input_data:
            data = input_data[node_name]
            # Convert to spikes if needed
            if data.ndim > 1:
                # Rate-based encoding
                spike_rates = np.clip(data, 0, 1)
                spikes = (np.random.random(data.shape) < spike_rates).astype(np.float32)
            else:
                spikes = data.astype(np.float32)
        else:
            # Default zero input
            spikes = np.zeros(self._get_node_size(node))
            
        return spikes, spikes.copy()  # Membrane potential = spike output for input nodes
        
    def _execute_spike_conv(
        self,
        node: Any,
        input_spikes: np.ndarray,
        node_state: Dict[str, Any],
        timestep: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute spike convolution operation."""
        # Simplified spike convolution
        # In practice, this would use optimized kernels
        kernel_size = node.parameters.get("kernel_size", 3)
        threshold = node.parameters.get("threshold", 1.0)
        
        # Simple pooling-based approximation
        if input_spikes.size > 0:
            pooled = np.mean(input_spikes) * kernel_size
            output_spikes = np.array([1.0 if pooled > threshold else 0.0])
        else:
            output_spikes = np.array([0.0])
            
        return output_spikes, output_spikes.copy()
        
    def _execute_spike_attention(
        self,
        node: Any,
        input_spikes: np.ndarray,
        node_state: Dict[str, Any],
        timestep: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute spike-based attention."""
        # Simplified spike attention
        embed_dim = node.parameters.get("embed_dim", 128)
        sparse_ratio = node.parameters.get("sparse_ratio", 0.1)
        
        if input_spikes.size > 0:
            # Sparse attention approximation
            attention_weights = np.random.random(input_spikes.shape) < sparse_ratio
            attended_spikes = input_spikes * attention_weights
            output = np.mean(attended_spikes) * np.ones(min(embed_dim, 64))
        else:
            output = np.zeros(min(embed_dim, 64))
            
        return output, output.copy()
        
    def _execute_generic_node(
        self,
        node: Any,
        input_spikes: np.ndarray,
        node_state: Dict[str, Any],
        timestep: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute generic node operation."""
        # Default pass-through behavior
        if input_spikes.size > 0:
            return input_spikes.copy(), input_spikes.copy()
        else:
            default_size = self._get_node_size(node)
            zeros = np.zeros(default_size)
            return zeros, zeros
            
    # Helper methods
    def _validate_execution_inputs(self, graph: SpikeGraph, input_data: Dict[str, np.ndarray]) -> None:
        """Validate execution inputs."""
        if not graph.nodes:
            raise SpikeRuntimeError("Cannot execute empty graph", error_code="EMPTY_GRAPH")
            
        # Check for required inputs
        input_nodes = [node for node in graph.nodes if node.node_type.name == "INPUT"]
        for input_node in input_nodes:
            input_name = input_node.parameters.get("name", "default")
            if input_name not in input_data:
                runtime_logger.warning(f"Missing input data for {input_name}, using zeros")
                
    def _get_execution_order(self, graph: SpikeGraph) -> List[str]:
        """Get topologically sorted execution order."""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            # Visit dependencies first
            node = graph.get_node(node_id)
            for input_id in node.inputs:
                if input_id in graph.nodes:
                    visit(input_id)
                    
            order.append(node_id)
            
        for node in graph.nodes:
            visit(node.id)
            
        return order
        
    def _can_execute_parallel(self, node_id: str, graph: SpikeGraph) -> bool:
        """Check if node can be executed in parallel."""
        # For simplicity, assume nodes with no dependencies can be parallel
        node = graph.get_node(node_id)
        return len(node.inputs) == 0
        
    def _collect_node_inputs(
        self,
        node: Any,
        timestep: int,
        execution_id: str
    ) -> np.ndarray:
        """Collect input spikes for a node."""
        inputs = []
        
        for input_id in node.inputs:
            if input_id in self.execution_state[execution_id]["node_states"]:
                input_state = self.execution_state[execution_id]["node_states"][input_id]
                if input_state["spike_history"]:
                    inputs.append(input_state["spike_history"][-1])
                    
        return np.concatenate(inputs) if inputs else np.array([])
        
    def _get_node_size(self, node: Any) -> int:
        """Get the output size of a node."""
        if hasattr(node, 'metadata') and 'shape' in node.metadata:
            return int(np.prod(node.metadata['shape']))
        return node.parameters.get("num_neurons", 1)
        
    def _get_node_size_by_id(self, node_id: str, graph: SpikeGraph) -> int:
        """Get node size by ID."""
        node = graph.get_node(node_id)
        return self._get_node_size(node)
        
    def _is_node_active(self, node_id: str, execution_id: str) -> bool:
        """Check if a node is currently active."""
        node_state = self.execution_state[execution_id]["node_states"].get(node_id)
        if node_state:
            return node_state["activation_count"] > 0
        return False
        
    def _perform_health_checks(self, execution_id: str, timestep: int) -> None:
        """Perform runtime health checks."""
        # Memory usage check
        current_memory = self.memory_manager.get_current_usage()
        if current_memory > self.memory_manager.get_memory_limit() * 0.9:
            runtime_logger.warning(f"High memory usage detected: {current_memory} MB")
            
        # Execution time check
        execution_time = time.time() - self.execution_state[execution_id]["start_time"]
        if execution_time > 30.0:  # 30 second timeout
            runtime_logger.warning(f"Long execution time detected: {execution_time:.1f}s")
            
    def _collect_outputs(self, graph: SpikeGraph, execution_id: str) -> Dict[str, np.ndarray]:
        """Collect final outputs from the graph."""
        outputs = {}
        
        output_nodes = [node for node in graph.nodes if node.node_type.name == "OUTPUT"]
        for output_node in output_nodes:
            output_name = output_node.parameters.get("name", output_node.id)
            node_state = self.execution_state[execution_id]["node_states"][output_node.id]
            
            if node_state["spike_history"]:
                outputs[output_name] = node_state["spike_history"][-1]
            else:
                outputs[output_name] = np.zeros(self._get_node_size(output_node))
                
        return outputs
        
    def _post_process_results(self, results: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Post-process execution results."""
        # Add metadata
        results["metadata"] = {
            "execution_id": execution_id,
            "execution_time": time.time() - self.execution_state[execution_id]["start_time"],
            "memory_peak": self.memory_manager.get_peak_usage(),
            "error_count": self.error_count,
            "recovery_attempts": self.recovery_attempts
        }
        
        # Cleanup execution state
        if execution_id in self.execution_state:
            del self.execution_state[execution_id]
            
        return results
        
    def _attempt_recovery(
        self,
        graph: SpikeGraph,
        input_data: Dict[str, np.ndarray],
        time_steps: int,
        error: Exception
    ) -> Dict[str, Any]:
        """Attempt to recover from execution failure."""
        self.recovery_attempts += 1
        runtime_logger.info(f"Recovery attempt {self.recovery_attempts}")
        
        # Simple recovery: reduce time steps and try again
        reduced_time_steps = max(1, time_steps // 2)
        runtime_logger.info(f"Reducing time steps from {time_steps} to {reduced_time_steps}")
        
        try:
            return self.execute_graph(graph, input_data, reduced_time_steps, False)
        except Exception as recovery_error:
            runtime_logger.error(f"Recovery failed: {str(recovery_error)}")
            raise error  # Re-raise original error
            
    def cleanup(self) -> None:
        """Cleanup runtime resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            
        self.memory_manager.cleanup()
        self.execution_state.clear()
        self.execution_trace.clear()
        
        runtime_logger.info("ExecutionEngine cleanup completed")
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_executions": len(self.performance_stats),
            "error_count": self.error_count,
            "recovery_attempts": self.recovery_attempts,
            "average_execution_time": np.mean([s.get("total_duration", 0) for s in self.performance_stats.values()]) if self.performance_stats else 0,
            "memory_stats": self.memory_manager.get_stats()
        }


class NeuromorphicExecutor(ExecutionEngine):
    """High-level executor interface for neuromorphic models."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spike_buffer_manager = SpikeBufferManager()
        self.multi_chip_communicator = MultiChipCommunicator()
        
    def run_inference(
        self,
        compiled_model: Any,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]],
        time_steps: int = 4,
        return_spike_trains: bool = False
    ) -> Any:
        """Run inference on a compiled neuromorphic model."""
        # Convert input format
        if isinstance(input_data, np.ndarray):
            input_dict = {"model_input": input_data}
        else:
            input_dict = input_data
            
        # Execute using the compiled model's graph
        results = self.execute_graph(
            compiled_model.graph,
            input_dict,
            time_steps,
            return_intermediate=return_spike_trains
        )
        
        if return_spike_trains:
            return {
                "output": results["outputs"],
                "spike_trains": results["intermediate"]
            }
        else:
            # Return primary output
            outputs = results["outputs"]
            if len(outputs) == 1:
                return list(outputs.values())[0]
            return outputs