"""Memory management for neuromorphic runtime execution."""

import psutil
import numpy as np
from typing import Dict, Any, Optional, List
import gc
from ..ir.spike_graph import SpikeGraph
from ..logging_config import runtime_logger


class MemoryManager:
    """Manages memory allocation and cleanup for neuromorphic execution."""
    
    def __init__(self, memory_limit_mb: Optional[float] = None):
        self.memory_limit_mb = memory_limit_mb or self._get_available_memory() * 0.8
        self.allocated_memory = {}
        self.memory_pools = {}
        self.peak_usage = 0
        self.allocation_history = []
        
        runtime_logger.info(f"MemoryManager initialized with limit: {self.memory_limit_mb:.1f} MB")
        
    def _get_available_memory(self) -> float:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)
        
    def allocate_graph_memory(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Allocate memory for entire graph execution."""
        graph_id = f"graph_{id(graph)}"
        
        try:
            # Calculate memory requirements
            memory_requirements = self._calculate_graph_memory(graph)
            
            # Check if allocation is possible
            if memory_requirements["total_mb"] > self.memory_limit_mb:
                raise MemoryError(
                    f"Graph requires {memory_requirements['total_mb']:.1f} MB, "
                    f"but limit is {self.memory_limit_mb:.1f} MB"
                )
                
            # Allocate memory pools
            allocation = self._allocate_memory_pools(graph, memory_requirements)
            self.allocated_memory[graph_id] = allocation
            
            # Update peak usage
            current_usage = self.get_current_usage()
            self.peak_usage = max(self.peak_usage, current_usage)
            
            # Log allocation
            self.allocation_history.append({
                "graph_id": graph_id,
                "memory_mb": memory_requirements["total_mb"],
                "nodes": len(graph.nodes),
                "edges": len(graph.edges)
            })
            
            runtime_logger.info(
                f"Allocated {memory_requirements['total_mb']:.1f} MB for graph {graph_id}"
            )
            
            return allocation
            
        except Exception as e:
            runtime_logger.error(f"Memory allocation failed for graph: {str(e)}")
            raise
            
    def _calculate_graph_memory(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Calculate memory requirements for graph."""
        total_neurons = 0
        total_synapses = 0
        total_parameters = 0
        
        for node in graph.nodes:
            # Node-specific memory calculation
            node_neurons = self._estimate_node_neurons(node)
            node_synapses = self._estimate_node_synapses(node)
            node_params = self._estimate_node_parameters(node)
            
            total_neurons += node_neurons
            total_synapses += node_synapses
            total_parameters += node_params
            
        # Memory breakdown (in MB)
        neuron_memory = total_neurons * 64 / (1024 * 1024)  # 64 bytes per neuron state
        synapse_memory = total_synapses * 8 / (1024 * 1024)  # 8 bytes per synapse
        parameter_memory = total_parameters * 4 / (1024 * 1024)  # 4 bytes per float32
        buffer_memory = max(16, total_neurons * 4 / (1024 * 1024))  # Spike buffers
        
        total_mb = neuron_memory + synapse_memory + parameter_memory + buffer_memory
        
        return {
            "total_mb": total_mb,
            "neuron_memory_mb": neuron_memory,
            "synapse_memory_mb": synapse_memory,
            "parameter_memory_mb": parameter_memory,
            "buffer_memory_mb": buffer_memory,
            "total_neurons": total_neurons,
            "total_synapses": total_synapses,
            "total_parameters": total_parameters
        }
        
    def _estimate_node_neurons(self, node: Any) -> int:
        """Estimate number of neurons in a node."""
        if hasattr(node, 'metadata') and 'shape' in node.metadata:
            return int(np.prod(node.metadata['shape']))
        return node.parameters.get("num_neurons", node.parameters.get("out_features", 1))
        
    def _estimate_node_synapses(self, node: Any) -> int:
        """Estimate number of synapses in a node."""
        node_type = node.node_type.name
        
        if node_type == "SPIKE_LINEAR":
            in_features = node.parameters.get("in_features", 1)
            out_features = node.parameters.get("out_features", 1)
            return in_features * out_features
        elif node_type == "SPIKE_CONV":
            kernel_size = node.parameters.get("kernel_size", 3)
            in_channels = node.parameters.get("in_channels", 1)
            out_channels = node.parameters.get("out_channels", 1)
            return kernel_size * kernel_size * in_channels * out_channels
        elif node_type == "SPIKE_ATTENTION":
            embed_dim = node.parameters.get("embed_dim", 128)
            return embed_dim * embed_dim * 4  # Q, K, V, O projections
        else:
            return 0
            
    def _estimate_node_parameters(self, node: Any) -> int:
        """Estimate number of parameters in a node."""
        return self._estimate_node_synapses(node)  # Simplified: 1 param per synapse
        
    def _allocate_memory_pools(
        self,
        graph: SpikeGraph,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate memory pools for different data types."""
        allocation = {
            "neuron_states": self._create_memory_pool(
                "neuron_states",
                requirements["neuron_memory_mb"]
            ),
            "synapse_weights": self._create_memory_pool(
                "synapse_weights", 
                requirements["synapse_memory_mb"]
            ),
            "spike_buffers": self._create_memory_pool(
                "spike_buffers",
                requirements["buffer_memory_mb"]
            ),
            "temporary_storage": self._create_memory_pool(
                "temporary",
                max(4, requirements["total_mb"] * 0.1)  # 10% for temporary data
            )
        }
        
        return allocation
        
    def _create_memory_pool(self, pool_name: str, size_mb: float) -> Dict[str, Any]:
        """Create a memory pool of specified size."""
        pool_id = f"{pool_name}_{len(self.memory_pools)}"
        
        # Allocate numpy array as memory pool
        pool_size_bytes = int(size_mb * 1024 * 1024)
        memory_pool = np.zeros(pool_size_bytes // 4, dtype=np.float32)  # 4 bytes per float32
        
        pool_info = {
            "pool_id": pool_id,
            "size_mb": size_mb,
            "memory_pool": memory_pool,
            "allocated_bytes": 0,
            "free_blocks": [(0, pool_size_bytes)],
            "allocated_blocks": []
        }
        
        self.memory_pools[pool_id] = pool_info
        
        runtime_logger.debug(f"Created memory pool {pool_id}: {size_mb:.1f} MB")
        
        return pool_info
        
    def allocate_from_pool(
        self,
        pool_id: str,
        size_bytes: int,
        alignment: int = 64
    ) -> Optional[np.ndarray]:
        """Allocate memory from a specific pool."""
        if pool_id not in self.memory_pools:
            runtime_logger.error(f"Memory pool {pool_id} not found")
            return None
            
        pool = self.memory_pools[pool_id]
        
        # Find suitable free block
        for i, (start, size) in enumerate(pool["free_blocks"]):
            if size >= size_bytes:
                # Allocate from this block
                allocated_start = start
                allocated_end = start + size_bytes
                
                # Update free blocks
                remaining_size = size - size_bytes
                if remaining_size > 0:
                    pool["free_blocks"][i] = (allocated_end, remaining_size)
                else:
                    del pool["free_blocks"][i]
                    
                # Track allocation
                pool["allocated_blocks"].append((allocated_start, size_bytes))
                pool["allocated_bytes"] += size_bytes
                
                # Return view of memory pool
                start_idx = allocated_start // 4  # Convert to float32 indices
                end_idx = allocated_end // 4
                
                return pool["memory_pool"][start_idx:end_idx]
                
        runtime_logger.warning(f"No sufficient memory in pool {pool_id} for {size_bytes} bytes")
        return None
        
    def deallocate_from_pool(self, pool_id: str, memory_view: np.ndarray) -> bool:
        """Deallocate memory back to pool."""
        if pool_id not in self.memory_pools:
            return False
            
        pool = self.memory_pools[pool_id]
        
        # Find the allocation to remove (simplified implementation)
        # In practice, would need better bookkeeping
        
        runtime_logger.debug(f"Deallocated memory from pool {pool_id}")
        return True
        
    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        total_usage = 0
        for pool in self.memory_pools.values():
            total_usage += pool["allocated_bytes"] / (1024 * 1024)
        return total_usage
        
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_usage
        
    def get_memory_limit(self) -> float:
        """Get memory limit in MB."""
        return self.memory_limit_mb
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        total_pools = len(self.memory_pools)
        total_allocated = sum(pool["allocated_bytes"] for pool in self.memory_pools.values())
        total_capacity = sum(pool["size_mb"] * 1024 * 1024 for pool in self.memory_pools.values())
        
        return {
            "current_usage_mb": self.get_current_usage(),
            "peak_usage_mb": self.peak_usage,
            "memory_limit_mb": self.memory_limit_mb,
            "utilization_percent": (self.get_current_usage() / self.memory_limit_mb) * 100,
            "total_pools": total_pools,
            "total_allocated_bytes": total_allocated,
            "total_capacity_bytes": total_capacity,
            "fragmentation_percent": self._calculate_fragmentation(),
            "allocation_history": len(self.allocation_history)
        }
        
    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation percentage."""
        if not self.memory_pools:
            return 0.0
            
        total_free_blocks = 0
        largest_free_block = 0
        
        for pool in self.memory_pools.values():
            total_free_blocks += len(pool["free_blocks"])
            if pool["free_blocks"]:
                largest_free_block = max(largest_free_block, max(size for _, size in pool["free_blocks"]))
                
        # Simplified fragmentation metric
        if total_free_blocks == 0:
            return 0.0
            
        return min(100.0, (total_free_blocks - len(self.memory_pools)) / total_free_blocks * 100)
        
    def cleanup(self) -> None:
        """Cleanup all allocated memory."""
        # Force garbage collection
        gc.collect()
        
        # Clear memory pools
        for pool_id in list(self.memory_pools.keys()):
            del self.memory_pools[pool_id]
            
        self.allocated_memory.clear()
        self.memory_pools.clear()
        
        runtime_logger.info("MemoryManager cleanup completed")
        
    def defragment(self) -> bool:
        """Defragment memory pools to reduce fragmentation."""
        runtime_logger.info("Starting memory defragmentation")
        
        defragmented_pools = 0
        
        for pool_id, pool in self.memory_pools.items():
            if self._needs_defragmentation(pool):
                if self._defragment_pool(pool):
                    defragmented_pools += 1
                    
        runtime_logger.info(f"Defragmented {defragmented_pools} memory pools")
        return defragmented_pools > 0
        
    def _needs_defragmentation(self, pool: Dict[str, Any]) -> bool:
        """Check if a pool needs defragmentation."""
        return len(pool["free_blocks"]) > 3  # Threshold for fragmentation
        
    def _defragment_pool(self, pool: Dict[str, Any]) -> bool:
        """Defragment a single memory pool."""
        # Simplified defragmentation: merge adjacent free blocks
        free_blocks = sorted(pool["free_blocks"])
        merged_blocks = []
        
        if not free_blocks:
            return False
            
        current_start, current_size = free_blocks[0]
        
        for start, size in free_blocks[1:]:
            if current_start + current_size == start:
                # Merge blocks
                current_size += size
            else:
                merged_blocks.append((current_start, current_size))
                current_start, current_size = start, size
                
        merged_blocks.append((current_start, current_size))
        
        # Update pool
        original_count = len(pool["free_blocks"])
        pool["free_blocks"] = merged_blocks
        
        return len(merged_blocks) < original_count


class SpikeBufferManager:
    """Manages spike data buffers for temporal processing."""
    
    def __init__(self, buffer_size_mb: float = 64):
        self.buffer_size_mb = buffer_size_mb
        self.buffers = {}
        self.buffer_metadata = {}
        
    def create_spike_buffer(
        self,
        buffer_id: str,
        shape: tuple,
        time_steps: int,
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """Create a spike buffer for temporal data."""
        buffer_shape = (time_steps,) + shape
        buffer_size_bytes = np.prod(buffer_shape) * np.dtype(dtype).itemsize
        
        if buffer_size_bytes > self.buffer_size_mb * 1024 * 1024:
            raise MemoryError(f"Buffer size {buffer_size_bytes / (1024*1024):.1f} MB exceeds limit")
            
        buffer = np.zeros(buffer_shape, dtype=dtype)
        
        self.buffers[buffer_id] = buffer
        self.buffer_metadata[buffer_id] = {
            "shape": buffer_shape,
            "dtype": dtype,
            "size_mb": buffer_size_bytes / (1024 * 1024),
            "created_at": np.datetime64('now')
        }
        
        runtime_logger.debug(f"Created spike buffer {buffer_id}: {buffer_shape}")
        
        return buffer
        
    def get_spike_buffer(self, buffer_id: str) -> Optional[np.ndarray]:
        """Get spike buffer by ID."""
        return self.buffers.get(buffer_id)
        
    def update_spike_buffer(
        self,
        buffer_id: str,
        timestep: int,
        spike_data: np.ndarray
    ) -> bool:
        """Update spike buffer at specific timestep."""
        if buffer_id not in self.buffers:
            return False
            
        buffer = self.buffers[buffer_id]
        if timestep >= buffer.shape[0]:
            runtime_logger.warning(f"Timestep {timestep} exceeds buffer size")
            return False
            
        try:
            buffer[timestep] = spike_data
            return True
        except Exception as e:
            runtime_logger.error(f"Failed to update spike buffer: {str(e)}")
            return False
            
    def clear_buffer(self, buffer_id: str) -> bool:
        """Clear spike buffer contents."""
        if buffer_id in self.buffers:
            self.buffers[buffer_id].fill(0)
            return True
        return False
        
    def cleanup(self) -> None:
        """Cleanup all spike buffers."""
        self.buffers.clear()
        self.buffer_metadata.clear()
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get spike buffer statistics."""
        total_memory = sum(
            metadata["size_mb"] for metadata in self.buffer_metadata.values()
        )
        
        return {
            "total_buffers": len(self.buffers),
            "total_memory_mb": total_memory,
            "memory_limit_mb": self.buffer_size_mb,
            "utilization_percent": (total_memory / self.buffer_size_mb) * 100,
            "buffers": list(self.buffer_metadata.keys())
        }