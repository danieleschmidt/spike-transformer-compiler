"""Advanced memory optimization with graph partitioning for large-scale neuromorphic compilation."""

import math
import time
import threading
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import weakref

import numpy as np

from ..ir.spike_graph import SpikeGraph, SpikeNode, SpikeEdge, NodeType
from ..logging_config import compiler_logger
from ..performance import PerformanceProfiler


@dataclass
class MemoryBlock:
    """Represents a memory block allocation."""
    block_id: str
    size_bytes: int
    alignment: int = 64
    lifetime_start: int = 0
    lifetime_end: int = float('inf')
    shared_count: int = 0
    memory_type: str = "general"  # general, weight, activation, intermediate


@dataclass 
class GraphPartition:
    """Represents a partition of the spike graph."""
    partition_id: str
    nodes: Set[str]
    edges: List[SpikeEdge]
    memory_requirement: int
    computation_cost: float
    communication_cost: float
    dependencies: Set[str]


class AdvancedMemoryManager:
    """Advanced memory manager with intelligent allocation strategies."""
    
    def __init__(self, total_memory_mb: int = 8192):
        self.total_memory_bytes = total_memory_mb * 1024 * 1024
        self.allocated_memory = 0
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        
        # Memory pools for different types
        self.memory_pools = {
            'weights': MemoryPool('weights', self.total_memory_bytes // 4),
            'activations': MemoryPool('activations', self.total_memory_bytes // 4), 
            'intermediate': MemoryPool('intermediate', self.total_memory_bytes // 4),
            'general': MemoryPool('general', self.total_memory_bytes // 4)
        }
        
        # Advanced features
        self.memory_compactor = MemoryCompactor()
        self.allocation_optimizer = AllocationOptimizer()
        self.fragmentation_analyzer = FragmentationAnalyzer()
        
        # Tracking
        self.allocation_history = deque(maxlen=1000)
        self.peak_usage = 0
        self.fragmentation_ratio = 0.0
        
        compiler_logger.logger.info(f"Advanced memory manager initialized with {total_memory_mb}MB")
    
    def allocate_block(
        self,
        size_bytes: int,
        memory_type: str = "general",
        alignment: int = 64,
        lifetime_hint: Optional[Tuple[int, int]] = None
    ) -> Optional[MemoryBlock]:
        """Allocate a memory block with advanced optimization."""
        
        # Choose optimal memory pool
        pool = self.memory_pools.get(memory_type, self.memory_pools['general'])
        
        # Try allocation with optimization
        optimized_size = self.allocation_optimizer.optimize_allocation_size(
            size_bytes, memory_type, self._get_usage_pattern()
        )
        
        block = pool.allocate(optimized_size, alignment)
        if not block:
            # Try compaction and retry
            freed_bytes = self.memory_compactor.compact_pool(pool)
            if freed_bytes >= optimized_size:
                block = pool.allocate(optimized_size, alignment)
        
        if not block:
            # Try cross-pool reallocation
            block = self._try_cross_pool_allocation(optimized_size, alignment, memory_type)
        
        if block:
            # Set lifetime information
            if lifetime_hint:
                block.lifetime_start, block.lifetime_end = lifetime_hint
                
            block.memory_type = memory_type
            self.memory_blocks[block.block_id] = block
            self.allocated_memory += block.size_bytes
            
            # Track peak usage
            self.peak_usage = max(self.peak_usage, self.allocated_memory)
            
            # Record allocation
            self.allocation_history.append({
                'timestamp': time.time(),
                'operation': 'allocate',
                'block_id': block.block_id,
                'size_bytes': block.size_bytes,
                'memory_type': memory_type
            })
        
        return block
    
    def deallocate_block(self, block_id: str) -> bool:
        """Deallocate a memory block."""
        if block_id not in self.memory_blocks:
            return False
        
        block = self.memory_blocks[block_id]
        pool = self.memory_pools[block.memory_type]
        
        success = pool.deallocate(block_id)
        if success:
            self.allocated_memory -= block.size_bytes
            del self.memory_blocks[block_id]
            
            # Record deallocation
            self.allocation_history.append({
                'timestamp': time.time(),
                'operation': 'deallocate', 
                'block_id': block_id,
                'size_bytes': block.size_bytes,
                'memory_type': block.memory_type
            })
        
        return success
    
    def _try_cross_pool_allocation(
        self, 
        size_bytes: int, 
        alignment: int, 
        preferred_type: str
    ) -> Optional[MemoryBlock]:
        """Try allocation across different memory pools."""
        # Priority order for cross-pool allocation
        pool_priority = {
            'weights': ['general', 'intermediate', 'activations'],
            'activations': ['intermediate', 'general', 'weights'],
            'intermediate': ['general', 'activations', 'weights'],
            'general': ['intermediate', 'activations', 'weights']
        }
        
        fallback_pools = pool_priority.get(preferred_type, ['general'])
        
        for pool_type in fallback_pools:
            pool = self.memory_pools[pool_type]
            block = pool.allocate(size_bytes, alignment)
            if block:
                compiler_logger.logger.debug(f"Cross-pool allocation: {preferred_type} -> {pool_type}")
                return block
        
        return None
    
    def _get_usage_pattern(self) -> Dict[str, Any]:
        """Get current memory usage pattern."""
        if len(self.allocation_history) < 10:
            return {'pattern': 'unknown'}
        
        recent_allocations = list(self.allocation_history)[-50:]
        
        # Analyze patterns
        alloc_sizes = [a['size_bytes'] for a in recent_allocations if a['operation'] == 'allocate']
        avg_alloc_size = np.mean(alloc_sizes) if alloc_sizes else 0
        
        # Allocation frequency by type
        type_counts = defaultdict(int)
        for alloc in recent_allocations:
            if alloc['operation'] == 'allocate':
                type_counts[alloc['memory_type']] += 1
        
        return {
            'pattern': 'analyzed',
            'avg_allocation_size': avg_alloc_size,
            'dominant_type': max(type_counts, key=type_counts.get) if type_counts else 'general',
            'allocation_frequency': len(alloc_sizes) / max(1, len(recent_allocations))
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        pool_stats = {}
        for name, pool in self.memory_pools.items():
            pool_stats[name] = pool.get_stats()
        
        fragmentation = self.fragmentation_analyzer.analyze_fragmentation(self.memory_pools)
        
        return {
            'total_memory_bytes': self.total_memory_bytes,
            'allocated_bytes': self.allocated_memory,
            'utilization': self.allocated_memory / self.total_memory_bytes,
            'peak_usage_bytes': self.peak_usage,
            'active_blocks': len(self.memory_blocks),
            'fragmentation_ratio': fragmentation,
            'pool_stats': pool_stats,
            'allocation_efficiency': self._calculate_allocation_efficiency()
        }
    
    def _calculate_allocation_efficiency(self) -> float:
        """Calculate allocation efficiency score."""
        if not self.allocation_history:
            return 1.0
        
        # Count successful vs failed allocations
        recent = list(self.allocation_history)[-100:]
        allocations = [a for a in recent if a['operation'] == 'allocate']
        
        # Simple efficiency metric based on fragmentation and utilization
        efficiency = (1.0 - self.fragmentation_ratio) * 0.5
        efficiency += min(1.0, self.allocated_memory / self.total_memory_bytes) * 0.5
        
        return efficiency


class MemoryPool:
    """Memory pool for specific allocation types."""
    
    def __init__(self, pool_type: str, max_size_bytes: int):
        self.pool_type = pool_type
        self.max_size_bytes = max_size_bytes
        self.allocated_bytes = 0
        self.blocks: Dict[str, MemoryBlock] = {}
        self.free_blocks: List[Tuple[int, int]] = [(0, max_size_bytes)]  # (start, size)
        self.allocation_counter = 0
        
    def allocate(self, size_bytes: int, alignment: int = 64) -> Optional[MemoryBlock]:
        """Allocate a block from the pool."""
        # Align size
        aligned_size = self._align_size(size_bytes, alignment)
        
        # Find suitable free block
        for i, (start, free_size) in enumerate(self.free_blocks):
            if free_size >= aligned_size:
                # Create block
                block_id = f"{self.pool_type}_block_{self.allocation_counter}"
                self.allocation_counter += 1
                
                block = MemoryBlock(
                    block_id=block_id,
                    size_bytes=aligned_size,
                    alignment=alignment
                )
                
                # Update free blocks
                remaining_size = free_size - aligned_size
                if remaining_size > 0:
                    self.free_blocks[i] = (start + aligned_size, remaining_size)
                else:
                    self.free_blocks.pop(i)
                
                self.blocks[block_id] = block
                self.allocated_bytes += aligned_size
                
                return block
        
        return None
    
    def deallocate(self, block_id: str) -> bool:
        """Deallocate a block and return it to the free list."""
        if block_id not in self.blocks:
            return False
        
        block = self.blocks.pop(block_id)
        self.allocated_bytes -= block.size_bytes
        
        # Add to free list and merge adjacent blocks
        self._add_to_free_list(0, block.size_bytes)  # Simplified - would need actual address
        
        return True
    
    def _add_to_free_list(self, start: int, size: int) -> None:
        """Add block to free list and merge adjacent blocks."""
        # Insert in sorted order
        inserted = False
        for i, (free_start, free_size) in enumerate(self.free_blocks):
            if start < free_start:
                self.free_blocks.insert(i, (start, size))
                inserted = True
                break
        
        if not inserted:
            self.free_blocks.append((start, size))
        
        # Merge adjacent blocks (simplified implementation)
        merged = []
        for start_addr, block_size in sorted(self.free_blocks):
            if merged and merged[-1][0] + merged[-1][1] == start_addr:
                # Merge with previous
                merged[-1] = (merged[-1][0], merged[-1][1] + block_size)
            else:
                merged.append((start_addr, block_size))
        
        self.free_blocks = merged
    
    def _align_size(self, size: int, alignment: int) -> int:
        """Align size to boundary."""
        return ((size + alignment - 1) // alignment) * alignment
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_type': self.pool_type,
            'max_size_bytes': self.max_size_bytes,
            'allocated_bytes': self.allocated_bytes,
            'utilization': self.allocated_bytes / self.max_size_bytes,
            'active_blocks': len(self.blocks),
            'free_blocks': len(self.free_blocks),
            'largest_free_block': max((size for _, size in self.free_blocks), default=0)
        }


class MemoryCompactor:
    """Compacts memory pools to reduce fragmentation."""
    
    def compact_pool(self, pool: MemoryPool) -> int:
        """Compact a memory pool and return bytes freed."""
        if len(pool.free_blocks) <= 1:
            return 0  # No fragmentation
        
        # Simulate compaction (in real implementation would move data)
        total_free = sum(size for _, size in pool.free_blocks)
        
        # Merge all free blocks into one
        if total_free > 0:
            pool.free_blocks = [(0, total_free)]  # Simplified
            
            compiler_logger.logger.debug(f"Compacted {pool.pool_type} pool, recovered {total_free} bytes")
            return total_free
        
        return 0


class AllocationOptimizer:
    """Optimizes allocation sizes and strategies."""
    
    def optimize_allocation_size(
        self, 
        requested_size: int, 
        memory_type: str, 
        usage_pattern: Dict[str, Any]
    ) -> int:
        """Optimize allocation size based on usage patterns."""
        
        # Pattern-based optimization
        if usage_pattern.get('pattern') == 'analyzed':
            avg_size = usage_pattern.get('avg_allocation_size', requested_size)
            
            # If request is much smaller than average, round up to reduce fragmentation
            if requested_size < avg_size * 0.5:
                optimized_size = min(requested_size * 2, avg_size)
            else:
                optimized_size = requested_size
        else:
            optimized_size = requested_size
        
        # Ensure power-of-2 sizes for certain types
        if memory_type in ['weights', 'activations']:
            optimized_size = self._next_power_of_2(optimized_size)
        
        return optimized_size
    
    def _next_power_of_2(self, n: int) -> int:
        """Find next power of 2."""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()


class FragmentationAnalyzer:
    """Analyzes memory fragmentation."""
    
    def analyze_fragmentation(self, memory_pools: Dict[str, MemoryPool]) -> float:
        """Analyze fragmentation across all pools."""
        total_fragmentation = 0.0
        pool_count = 0
        
        for pool in memory_pools.values():
            if pool.allocated_bytes > 0:
                pool_fragmentation = self._calculate_pool_fragmentation(pool)
                total_fragmentation += pool_fragmentation
                pool_count += 1
        
        return total_fragmentation / max(1, pool_count)
    
    def _calculate_pool_fragmentation(self, pool: MemoryPool) -> float:
        """Calculate fragmentation for a single pool."""
        if not pool.free_blocks:
            return 0.0
        
        total_free = sum(size for _, size in pool.free_blocks)
        largest_free = max(size for _, size in pool.free_blocks)
        
        if total_free == 0:
            return 0.0
        
        # Fragmentation ratio: 1 - (largest_free / total_free)
        fragmentation = 1.0 - (largest_free / total_free)
        return fragmentation


class GraphPartitioner:
    """Partitions spike graphs for distributed memory optimization."""
    
    def __init__(self):
        self.partitioning_strategies = {
            'min_cut': self._min_cut_partition,
            'load_balance': self._load_balanced_partition,
            'memory_aware': self._memory_aware_partition
        }
        
    def partition_graph(
        self, 
        graph: SpikeGraph, 
        num_partitions: int,
        strategy: str = 'memory_aware',
        memory_constraints: Optional[Dict[str, int]] = None
    ) -> List[GraphPartition]:
        """Partition graph using specified strategy."""
        
        if strategy not in self.partitioning_strategies:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")
        
        partitioner = self.partitioning_strategies[strategy]
        partitions = partitioner(graph, num_partitions, memory_constraints or {})
        
        # Validate partitions
        self._validate_partitions(graph, partitions)
        
        compiler_logger.logger.info(
            f"Graph partitioned into {len(partitions)} partitions using {strategy} strategy"
        )
        
        return partitions
    
    def _memory_aware_partition(
        self, 
        graph: SpikeGraph, 
        num_partitions: int, 
        memory_constraints: Dict[str, int]
    ) -> List[GraphPartition]:
        """Memory-aware graph partitioning."""
        
        partitions = []
        unassigned_nodes = set(graph.nodes.keys())
        
        for i in range(num_partitions):
            partition_id = f"partition_{i}"
            max_memory = memory_constraints.get(partition_id, float('inf'))
            
            partition_nodes = set()
            partition_memory = 0
            
            # Greedy assignment based on memory requirements
            nodes_by_memory = sorted(
                unassigned_nodes,
                key=lambda nid: self._estimate_node_memory(graph.nodes[nid]),
                reverse=True
            )
            
            for node_id in nodes_by_memory:
                node_memory = self._estimate_node_memory(graph.nodes[node_id])
                
                if partition_memory + node_memory <= max_memory:
                    partition_nodes.add(node_id)
                    partition_memory += node_memory
                    unassigned_nodes.remove(node_id)
                    
                    if len(partition_nodes) >= len(graph.nodes) // num_partitions:
                        break
            
            # Create partition
            if partition_nodes:
                partition = self._create_partition(graph, partition_id, partition_nodes)
                partitions.append(partition)
        
        # Assign remaining nodes to partitions with available capacity
        for node_id in unassigned_nodes:
            best_partition = min(
                partitions, 
                key=lambda p: p.memory_requirement
            )
            best_partition.nodes.add(node_id)
            best_partition.memory_requirement += self._estimate_node_memory(graph.nodes[node_id])
        
        return partitions
    
    def _min_cut_partition(
        self, 
        graph: SpikeGraph, 
        num_partitions: int, 
        memory_constraints: Dict[str, int]
    ) -> List[GraphPartition]:
        """Min-cut graph partitioning to minimize communication."""
        # Simplified min-cut implementation
        return self._balanced_partition(graph, num_partitions)
    
    def _load_balanced_partition(
        self, 
        graph: SpikeGraph, 
        num_partitions: int, 
        memory_constraints: Dict[str, int]
    ) -> List[GraphPartition]:
        """Load-balanced partitioning."""
        return self._balanced_partition(graph, num_partitions)
    
    def _balanced_partition(self, graph: SpikeGraph, num_partitions: int) -> List[GraphPartition]:
        """Simple balanced partitioning."""
        partitions = []
        nodes_per_partition = len(graph.nodes) // num_partitions
        
        node_list = list(graph.nodes.keys())
        
        for i in range(num_partitions):
            start_idx = i * nodes_per_partition
            end_idx = start_idx + nodes_per_partition
            
            if i == num_partitions - 1:  # Last partition gets remaining nodes
                partition_nodes = set(node_list[start_idx:])
            else:
                partition_nodes = set(node_list[start_idx:end_idx])
            
            if partition_nodes:
                partition_id = f"partition_{i}"
                partition = self._create_partition(graph, partition_id, partition_nodes)
                partitions.append(partition)
        
        return partitions
    
    def _create_partition(
        self, 
        graph: SpikeGraph, 
        partition_id: str, 
        nodes: Set[str]
    ) -> GraphPartition:
        """Create partition from node set."""
        
        # Find edges within partition and crossing partition boundaries
        internal_edges = []
        external_dependencies = set()
        
        for edge in graph.edges:
            if edge.source in nodes and edge.target in nodes:
                internal_edges.append(edge)
            elif edge.source in nodes or edge.target in nodes:
                # This partition depends on another partition
                if edge.source not in nodes:
                    external_dependencies.add(self._find_node_partition(edge.source, nodes))
                if edge.target not in nodes:
                    external_dependencies.add(self._find_node_partition(edge.target, nodes))
        
        # Calculate resource requirements
        memory_req = sum(
            self._estimate_node_memory(graph.nodes[nid]) for nid in nodes
        )
        
        computation_cost = sum(
            graph.nodes[nid].estimate_computation() for nid in nodes
        )
        
        communication_cost = len(graph.edges) - len(internal_edges)  # Simplified
        
        return GraphPartition(
            partition_id=partition_id,
            nodes=nodes,
            edges=internal_edges,
            memory_requirement=memory_req,
            computation_cost=computation_cost,
            communication_cost=communication_cost,
            dependencies=external_dependencies
        )
    
    def _find_node_partition(self, node_id: str, partition_nodes: Set[str]) -> str:
        """Find which partition a node belongs to (simplified)."""
        return "external"  # Simplified implementation
    
    def _estimate_node_memory(self, node: SpikeNode) -> int:
        """Estimate memory requirement for a node."""
        base_memory = 1024  # 1KB base
        
        if node.node_type == NodeType.SPIKE_CONV:
            channels = node.get_parameter("out_channels", 1)
            kernel_size = node.get_parameter("kernel_size", 3)
            return base_memory * channels * kernel_size * kernel_size
            
        elif node.node_type == NodeType.SPIKE_LINEAR:
            out_features = node.get_parameter("out_features", 1)
            return base_memory * out_features
            
        elif node.node_type == NodeType.SPIKE_ATTENTION:
            embed_dim = node.get_parameter("embed_dim", 512)
            seq_len = node.get_parameter("sequence_length", 100)
            return base_memory * embed_dim * seq_len
        
        return base_memory
    
    def _validate_partitions(self, graph: SpikeGraph, partitions: List[GraphPartition]) -> None:
        """Validate that partitions cover all nodes exactly once."""
        all_partitioned_nodes = set()
        
        for partition in partitions:
            # Check for node overlap
            overlap = all_partitioned_nodes & partition.nodes
            if overlap:
                raise ValueError(f"Node overlap detected in partitions: {overlap}")
            
            all_partitioned_nodes.update(partition.nodes)
        
        # Check all nodes are covered
        missing_nodes = set(graph.nodes.keys()) - all_partitioned_nodes
        if missing_nodes:
            raise ValueError(f"Nodes not assigned to any partition: {missing_nodes}")
        
        compiler_logger.logger.debug("Graph partitioning validation passed")
    
    def optimize_partition_assignment(
        self, 
        partitions: List[GraphPartition],
        hardware_constraints: Dict[str, Any]
    ) -> Dict[str, str]:
        """Optimize assignment of partitions to hardware resources."""
        
        assignment = {}
        available_resources = list(hardware_constraints.keys())
        
        # Sort partitions by resource requirements (largest first)
        sorted_partitions = sorted(
            partitions, 
            key=lambda p: p.memory_requirement, 
            reverse=True
        )
        
        for partition in sorted_partitions:
            # Find best resource for this partition
            best_resource = None
            min_cost = float('inf')
            
            for resource_id in available_resources:
                constraints = hardware_constraints[resource_id]
                
                # Check if partition fits
                if partition.memory_requirement <= constraints.get('memory_mb', 0) * 1024 * 1024:
                    # Calculate assignment cost (simplified)
                    cost = partition.communication_cost + partition.computation_cost
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_resource = resource_id
            
            if best_resource:
                assignment[partition.partition_id] = best_resource
            else:
                compiler_logger.logger.warning(
                    f"Could not assign partition {partition.partition_id} to any resource"
                )
        
        return assignment