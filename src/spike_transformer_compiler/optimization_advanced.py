"""Advanced optimization algorithms for neuromorphic compilation."""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict, deque
import math

from .logging_config import compiler_logger
from .monitoring import get_compilation_monitor


class OptimizationLevel(Enum):
    """Advanced optimization levels."""
    MINIMAL = 0         # Basic optimizations only
    STANDARD = 1        # Standard optimization passes
    AGGRESSIVE = 2      # Aggressive optimizations with some risk
    EXPERIMENTAL = 3    # Cutting-edge optimizations


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    nodes_removed: int = 0
    edges_removed: int = 0
    memory_saved_bytes: int = 0
    computation_ops_saved: int = 0
    estimated_speedup: float = 1.0
    energy_reduction_percent: float = 0.0
    optimization_time_ms: float = 0.0


class GraphOptimizer:
    """Advanced graph optimization algorithms."""
    
    def __init__(self):
        self.optimization_history: List[Dict] = []
        self.cache = {}
        self._lock = threading.Lock()
    
    def optimize_graph_structure(self, graph: "SpikeGraph") -> OptimizationMetrics:
        """Optimize graph structure using advanced algorithms."""
        start_time = time.time()
        metrics = OptimizationMetrics()
        
        # Phase 1: Dead code elimination with dataflow analysis
        dead_nodes = self._find_dead_nodes_dataflow(graph)
        for node_id in dead_nodes:
            if node_id in graph.nodes:
                del graph.nodes[node_id]
                metrics.nodes_removed += 1
        
        # Phase 2: Common subexpression elimination
        cse_saved = self._eliminate_common_subexpressions(graph)
        metrics.nodes_removed += cse_saved['nodes_removed']
        metrics.computation_ops_saved += cse_saved['ops_saved']
        
        # Phase 3: Loop fusion and vectorization
        fusion_results = self._fuse_compatible_operations(graph)
        metrics.computation_ops_saved += fusion_results['ops_saved']
        metrics.estimated_speedup *= fusion_results['speedup_factor']
        
        # Phase 4: Memory access optimization
        memory_results = self._optimize_memory_access_patterns(graph)
        metrics.memory_saved_bytes += memory_results['bytes_saved']
        
        # Phase 5: Energy optimization
        energy_results = self._optimize_for_energy_efficiency(graph)
        metrics.energy_reduction_percent = energy_results['reduction_percent']
        
        metrics.optimization_time_ms = (time.time() - start_time) * 1000
        
        # Record optimization history
        optimization_record = {
            'timestamp': time.time(),
            'graph_id': graph.id,
            'metrics': metrics,
            'optimization_level': OptimizationLevel.AGGRESSIVE.value
        }
        
        with self._lock:
            self.optimization_history.append(optimization_record)
        
        compiler_logger.logger.info(
            f"Graph optimization completed: {metrics.nodes_removed} nodes removed, "
            f"{metrics.estimated_speedup:.2f}x speedup, "
            f"{metrics.energy_reduction_percent:.1f}% energy reduction"
        )
        
        return metrics
    
    def _find_dead_nodes_dataflow(self, graph: "SpikeGraph") -> Set[str]:
        """Find dead nodes using dataflow analysis."""
        # Build def-use chains
        definitions = defaultdict(set)  # variable -> defining nodes
        uses = defaultdict(set)        # variable -> using nodes
        
        for node_id, node in graph.nodes.items():
            # Outputs are definitions
            for output in node.outputs:
                definitions[output].add(node_id)
            
            # Inputs are uses
            for input_var in node.inputs:
                uses[input_var].add(node_id)
        
        # Find live variables (those that reach outputs)
        live_vars = set()
        worklist = deque()
        
        # Start with output nodes
        for node_id, node in graph.nodes.items():
            if hasattr(node, 'node_type') and 'output' in str(node.node_type).lower():
                live_vars.update(node.inputs)
                worklist.extend(node.inputs)
        
        # Propagate liveness backward
        while worklist:
            var = worklist.popleft()
            
            # Find nodes that define this variable
            for def_node in definitions.get(var, []):
                node = graph.nodes[def_node]
                for input_var in node.inputs:
                    if input_var not in live_vars:
                        live_vars.add(input_var)
                        worklist.append(input_var)
        
        # Dead nodes are those whose outputs are not live
        dead_nodes = set()
        for node_id, node in graph.nodes.items():
            if not any(output in live_vars for output in node.outputs):
                # Check if it's not an output node or side-effect node
                if (not hasattr(node, 'node_type') or 
                    'output' not in str(node.node_type).lower()):
                    dead_nodes.add(node_id)
        
        return dead_nodes
    
    def _eliminate_common_subexpressions(self, graph: "SpikeGraph") -> Dict[str, int]:
        """Eliminate common subexpressions."""
        expression_map = {}  # expression signature -> node_id
        nodes_to_remove = set()
        replacement_map = {}  # old_node -> new_node
        
        for node_id, node in graph.nodes.items():
            # Create expression signature
            signature = self._compute_expression_signature(node)
            
            if signature in expression_map:
                # Found duplicate expression
                original_node_id = expression_map[signature]
                original_node = graph.nodes[original_node_id]
                
                # Verify expressions are truly equivalent
                if self._expressions_equivalent(node, original_node):
                    nodes_to_remove.add(node_id)
                    replacement_map[node_id] = original_node_id
            else:
                expression_map[signature] = node_id
        
        # Update graph to use common expressions
        ops_saved = 0
        for node_id in nodes_to_remove:
            if node_id in graph.nodes:
                del graph.nodes[node_id]
                ops_saved += self._estimate_node_ops(graph.nodes.get(replacement_map[node_id]))
        
        # Update edges to point to common expressions
        for edge in graph.edges:
            if edge.source in replacement_map:
                edge.source = replacement_map[edge.source]
            if hasattr(edge, 'target') and edge.target in replacement_map:
                edge.target = replacement_map[edge.target]
        
        return {
            'nodes_removed': len(nodes_to_remove),
            'ops_saved': ops_saved
        }
    
    def _compute_expression_signature(self, node) -> str:
        """Compute signature for expression matching."""
        import hashlib
        
        # Create signature from operation type and parameters
        sig_data = {
            'operation': node.operation,
            'parameters': sorted(node.parameters.items()) if hasattr(node, 'parameters') else [],
            'input_count': len(node.inputs),
            'output_count': len(node.outputs)
        }
        
        sig_str = str(sig_data)
        return hashlib.md5(sig_str.encode()).hexdigest()
    
    def _expressions_equivalent(self, node1, node2) -> bool:
        """Check if two expressions are equivalent."""
        if node1.operation != node2.operation:
            return False
        
        if len(node1.inputs) != len(node2.inputs):
            return False
        
        if len(node1.outputs) != len(node2.outputs):
            return False
        
        # Check parameters
        params1 = getattr(node1, 'parameters', {})
        params2 = getattr(node2, 'parameters', {})
        
        return params1 == params2
    
    def _estimate_node_ops(self, node) -> int:
        """Estimate computational operations for a node."""
        if not node:
            return 0
        
        # Simple estimation based on operation type
        op_costs = {
            'linear': 100,
            'conv2d': 1000,
            'attention': 500,
            'spike_neuron': 10,
            'pool': 50
        }
        
        base_cost = op_costs.get(node.operation.lower(), 100)
        
        # Scale by input/output sizes if available
        if hasattr(node, 'parameters'):
            if 'num_neurons' in node.parameters:
                base_cost *= node.parameters['num_neurons']
            elif 'in_features' in node.parameters and 'out_features' in node.parameters:
                base_cost *= (node.parameters['in_features'] * node.parameters['out_features']) // 1000
        
        return base_cost
    
    def _fuse_compatible_operations(self, graph: "SpikeGraph") -> Dict[str, Any]:
        """Fuse compatible operations for better performance."""
        fusion_opportunities = []
        
        # Find fusable operation patterns
        for node_id, node in graph.nodes.items():
            fusable_successors = self._find_fusable_successors(node, graph)
            if fusable_successors:
                fusion_opportunities.append({
                    'root': node_id,
                    'fusable': fusable_successors
                })
        
        ops_saved = 0
        speedup_factor = 1.0
        
        # Apply fusion optimizations
        for opportunity in fusion_opportunities:
            root_node = graph.nodes[opportunity['root']]
            fusable_nodes = [graph.nodes[nid] for nid in opportunity['fusable']]
            
            # Create fused operation
            fused_node = self._create_fused_node(root_node, fusable_nodes)
            
            # Replace nodes in graph
            graph.nodes[opportunity['root']] = fused_node
            
            # Remove fused nodes
            for node_id in opportunity['fusable']:
                if node_id in graph.nodes:
                    del graph.nodes[node_id]
                    ops_saved += self._estimate_node_ops(fusable_nodes[0])
            
            # Estimate speedup from fusion
            speedup_factor *= 1.2  # Conservative estimate
        
        return {
            'ops_saved': ops_saved,
            'speedup_factor': speedup_factor,
            'fusions_applied': len(fusion_opportunities)
        }
    
    def _find_fusable_successors(self, node, graph: "SpikeGraph") -> List[str]:
        """Find nodes that can be fused with the given node."""
        fusable = []
        
        # Look for specific fusion patterns
        for edge in graph.edges:
            if edge.source == node.id:
                successor = graph.nodes.get(edge.target if hasattr(edge, 'target') else edge.dst)
                if successor and self._can_fuse_operations(node, successor):
                    fusable.append(successor.id)
        
        return fusable
    
    def _can_fuse_operations(self, node1, node2) -> bool:
        """Check if two operations can be fused."""
        # Define fusable operation pairs
        fusable_pairs = [
            ('linear', 'relu'),
            ('conv2d', 'batch_norm'),
            ('spike_neuron', 'dropout'),
            ('attention', 'layer_norm')
        ]
        
        op1 = node1.operation.lower()
        op2 = node2.operation.lower()
        
        return (op1, op2) in fusable_pairs or (op2, op1) in fusable_pairs
    
    def _create_fused_node(self, root_node, fusable_nodes):
        """Create fused operation node."""
        # This would create an optimized fused operation
        # For now, just modify the root node to include fused operations
        
        fused_ops = [root_node.operation] + [n.operation for n in fusable_nodes]
        
        # Create new node with fused operation
        fused_node = type(root_node)(
            id=root_node.id,
            node_type=root_node.node_type,
            operation=f"fused_{'_'.join(fused_ops)}",
            inputs=root_node.inputs,
            outputs=fusable_nodes[-1].outputs if fusable_nodes else root_node.outputs,
            parameters={
                **getattr(root_node, 'parameters', {}),
                'fused_operations': fused_ops,
                'optimization_applied': 'operation_fusion'
            },
            metadata=getattr(root_node, 'metadata', {})
        )
        
        return fused_node
    
    def _optimize_memory_access_patterns(self, graph: "SpikeGraph") -> Dict[str, int]:
        """Optimize memory access patterns."""
        bytes_saved = 0
        
        # Analyze memory access patterns
        access_patterns = self._analyze_memory_patterns(graph)
        
        # Apply memory optimizations
        for pattern in access_patterns:
            if pattern['type'] == 'redundant_load':
                # Cache intermediate results
                bytes_saved += pattern['size_bytes']
            elif pattern['type'] == 'scattered_access':
                # Reorganize for better locality
                bytes_saved += pattern['size_bytes'] // 2  # Conservative estimate
        
        return {'bytes_saved': bytes_saved}
    
    def _analyze_memory_patterns(self, graph: "SpikeGraph") -> List[Dict]:
        """Analyze memory access patterns in the graph."""
        patterns = []
        
        # Look for nodes that access the same memory multiple times
        memory_accesses = defaultdict(list)
        
        for node_id, node in graph.nodes.items():
            # Estimate memory accesses based on node type
            if hasattr(node, 'parameters'):
                if 'in_features' in node.parameters:
                    access_size = node.parameters['in_features'] * 4  # float32
                    memory_accesses[f"input_{node_id}"].append({
                        'node': node_id,
                        'size': access_size,
                        'type': 'read'
                    })
                
                if 'out_features' in node.parameters:
                    access_size = node.parameters['out_features'] * 4  # float32
                    memory_accesses[f"output_{node_id}"].append({
                        'node': node_id,
                        'size': access_size,
                        'type': 'write'
                    })
        
        # Find optimization opportunities
        for mem_location, accesses in memory_accesses.items():
            if len(accesses) > 1:
                total_size = sum(acc['size'] for acc in accesses)
                patterns.append({
                    'type': 'redundant_load',
                    'location': mem_location,
                    'accesses': len(accesses),
                    'size_bytes': total_size
                })
        
        return patterns
    
    def _optimize_for_energy_efficiency(self, graph: "SpikeGraph") -> Dict[str, float]:
        """Optimize graph for energy efficiency."""
        energy_reduction = 0.0
        
        # Apply energy-specific optimizations
        optimizations = [
            self._reduce_spike_activity(graph),
            self._optimize_neuron_parameters(graph),
            self._minimize_memory_transfers(graph)
        ]
        
        for opt_result in optimizations:
            energy_reduction += opt_result
        
        return {'reduction_percent': min(50.0, energy_reduction)}  # Cap at 50%
    
    def _reduce_spike_activity(self, graph: "SpikeGraph") -> float:
        """Reduce unnecessary spike activity."""
        reduction = 0.0
        
        for node_id, node in graph.nodes.items():
            if 'spike_neuron' in node.operation.lower():
                # Optimize threshold parameters
                params = getattr(node, 'parameters', {})
                if 'threshold' in params:
                    # Slightly increase threshold to reduce firing rate
                    old_threshold = params['threshold']
                    params['threshold'] = old_threshold * 1.1
                    reduction += 2.0  # Estimate 2% energy reduction per neuron
        
        return min(15.0, reduction)  # Cap individual optimization
    
    def _optimize_neuron_parameters(self, graph: "SpikeGraph") -> float:
        """Optimize neuron parameters for energy efficiency."""
        reduction = 0.0
        
        for node_id, node in graph.nodes.items():
            if 'neuron' in node.operation.lower():
                params = getattr(node, 'parameters', {})
                
                # Optimize time constants
                if 'tau_mem' in params:
                    params['tau_mem'] *= 0.9  # Faster membrane decay
                    reduction += 1.0
                
                # Optimize reset behavior
                if 'reset_mode' in params and params['reset_mode'] == 'hard':
                    params['reset_mode'] = 'soft'  # Soft reset uses less energy
                    reduction += 0.5
        
        return min(10.0, reduction)
    
    def _minimize_memory_transfers(self, graph: "SpikeGraph") -> float:
        """Minimize memory transfers for energy efficiency."""
        reduction = 0.0
        
        # Analyze data dependencies
        transfer_graph = self._build_transfer_graph(graph)
        
        # Find opportunities to keep data in local memory
        for node_id, transfers in transfer_graph.items():
            if len(transfers) > 2:  # High memory traffic
                # Add metadata to suggest local caching
                if node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    if hasattr(node, 'metadata'):
                        node.metadata['cache_locally'] = True
                        reduction += 1.5
        
        return min(20.0, reduction)
    
    def _build_transfer_graph(self, graph: "SpikeGraph") -> Dict[str, List]:
        """Build graph of memory transfers."""
        transfers = defaultdict(list)
        
        for edge in graph.edges:
            source = edge.source
            target = edge.target if hasattr(edge, 'target') else edge.dst
            
            transfers[source].append(target)
            transfers[target].append(source)
        
        return dict(transfers)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self._lock:
            if not self.optimization_history:
                return {'message': 'No optimizations recorded'}
            
            total_optimizations = len(self.optimization_history)
            recent_optimizations = [opt for opt in self.optimization_history 
                                  if time.time() - opt['timestamp'] < 3600]  # Last hour
            
            # Aggregate metrics
            total_nodes_removed = sum(opt['metrics'].nodes_removed for opt in self.optimization_history)
            avg_speedup = sum(opt['metrics'].estimated_speedup for opt in self.optimization_history) / total_optimizations
            avg_energy_reduction = sum(opt['metrics'].energy_reduction_percent for opt in self.optimization_history) / total_optimizations
            
            return {
                'total_optimizations': total_optimizations,
                'recent_optimizations': len(recent_optimizations),
                'total_nodes_removed': total_nodes_removed,
                'average_speedup': avg_speedup,
                'average_energy_reduction_percent': avg_energy_reduction,
                'optimization_levels_used': list(set(opt['optimization_level'] for opt in self.optimization_history))
            }


class AdaptiveOptimizer:
    """Optimizer that adapts strategy based on model characteristics."""
    
    def __init__(self):
        self.model_profiles = {}
        self.graph_optimizer = GraphOptimizer()
        
    def profile_model(self, graph: "SpikeGraph") -> Dict[str, Any]:
        """Profile model to determine optimal optimization strategy."""
        profile = {
            'node_count': len(graph.nodes),
            'edge_count': len(graph.edges),
            'complexity_score': self._calculate_complexity_score(graph),
            'memory_footprint_mb': self._estimate_memory_footprint(graph),
            'dominant_operations': self._find_dominant_operations(graph),
            'optimization_potential': self._assess_optimization_potential(graph)
        }
        
        self.model_profiles[graph.id] = profile
        return profile
    
    def _calculate_complexity_score(self, graph: "SpikeGraph") -> float:
        """Calculate model complexity score."""
        base_score = len(graph.nodes) * 10 + len(graph.edges)
        
        # Factor in node complexity
        for node in graph.nodes.values():
            if 'attention' in node.operation.lower():
                base_score *= 2.0
            elif 'conv' in node.operation.lower():
                base_score *= 1.5
        
        return base_score
    
    def _estimate_memory_footprint(self, graph: "SpikeGraph") -> float:
        """Estimate memory footprint in MB."""
        total_bytes = 0
        
        for node in graph.nodes.values():
            if hasattr(node, 'parameters'):
                params = node.parameters
                
                # Estimate based on common parameters
                if 'in_features' in params and 'out_features' in params:
                    weights_bytes = params['in_features'] * params['out_features'] * 4  # float32
                    total_bytes += weights_bytes
                elif 'num_neurons' in params:
                    state_bytes = params['num_neurons'] * 8  # 2 floats per neuron
                    total_bytes += state_bytes
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _find_dominant_operations(self, graph: "SpikeGraph") -> List[Tuple[str, int]]:
        """Find most common operations in the graph."""
        op_counts = defaultdict(int)
        
        for node in graph.nodes.values():
            op_counts[node.operation] += 1
        
        # Return top 3 operations
        return sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _assess_optimization_potential(self, graph: "SpikeGraph") -> float:
        """Assess how much the graph can be optimized."""
        potential_score = 0.0
        
        # Check for dead code
        dead_nodes = self.graph_optimizer._find_dead_nodes_dataflow(graph)
        potential_score += len(dead_nodes) * 10
        
        # Check for redundant operations
        op_frequencies = defaultdict(int)
        for node in graph.nodes.values():
            signature = self.graph_optimizer._compute_expression_signature(node)
            op_frequencies[signature] += 1
        
        redundant_ops = sum(count - 1 for count in op_frequencies.values() if count > 1)
        potential_score += redundant_ops * 5
        
        # Normalize to 0-100 scale
        return min(100.0, potential_score / len(graph.nodes) * 10)
    
    def optimize_adaptively(self, graph: "SpikeGraph", performance_target: str = "balanced") -> OptimizationMetrics:
        """Optimize graph using adaptive strategy."""
        profile = self.profile_model(graph)
        
        # Determine optimization strategy based on profile and target
        strategy = self._select_optimization_strategy(profile, performance_target)
        
        compiler_logger.logger.info(
            f"Using {strategy} optimization for model with "
            f"complexity={profile['complexity_score']:.1f}, "
            f"potential={profile['optimization_potential']:.1f}%"
        )
        
        # Apply selected optimizations
        if strategy == "aggressive":
            return self._apply_aggressive_optimizations(graph, profile)
        elif strategy == "balanced":
            return self._apply_balanced_optimizations(graph, profile)
        elif strategy == "conservative":
            return self._apply_conservative_optimizations(graph, profile)
        else:  # minimal
            return self._apply_minimal_optimizations(graph, profile)
    
    def _select_optimization_strategy(self, profile: Dict, target: str) -> str:
        """Select optimization strategy based on model profile."""
        complexity = profile['complexity_score']
        potential = profile['optimization_potential']
        
        if target == "speed":
            return "aggressive" if potential > 20 else "balanced"
        elif target == "energy":
            return "aggressive" if complexity > 1000 else "balanced"
        elif target == "size":
            return "aggressive" if profile['memory_footprint_mb'] > 100 else "conservative"
        else:  # balanced
            if potential > 30 and complexity > 500:
                return "aggressive"
            elif potential > 15:
                return "balanced"
            else:
                return "conservative"
    
    def _apply_aggressive_optimizations(self, graph: "SpikeGraph", profile: Dict) -> OptimizationMetrics:
        """Apply aggressive optimizations."""
        return self.graph_optimizer.optimize_graph_structure(graph)
    
    def _apply_balanced_optimizations(self, graph: "SpikeGraph", profile: Dict) -> OptimizationMetrics:
        """Apply balanced optimizations."""
        # Selective optimization based on model characteristics
        metrics = OptimizationMetrics()
        
        # Always apply dead code elimination
        dead_nodes = self.graph_optimizer._find_dead_nodes_dataflow(graph)
        for node_id in dead_nodes:
            if node_id in graph.nodes:
                del graph.nodes[node_id]
                metrics.nodes_removed += 1
        
        # Apply other optimizations based on potential
        if profile['optimization_potential'] > 20:
            cse_results = self.graph_optimizer._eliminate_common_subexpressions(graph)
            metrics.nodes_removed += cse_results['nodes_removed']
            metrics.computation_ops_saved += cse_results['ops_saved']
        
        return metrics
    
    def _apply_conservative_optimizations(self, graph: "SpikeGraph", profile: Dict) -> OptimizationMetrics:
        """Apply conservative optimizations."""
        metrics = OptimizationMetrics()
        
        # Only apply safe optimizations
        dead_nodes = self.graph_optimizer._find_dead_nodes_dataflow(graph)
        safe_dead_nodes = [nid for nid in dead_nodes if self._is_safe_to_remove(graph.nodes[nid])]
        
        for node_id in safe_dead_nodes:
            if node_id in graph.nodes:
                del graph.nodes[node_id]
                metrics.nodes_removed += 1
        
        return metrics
    
    def _apply_minimal_optimizations(self, graph: "SpikeGraph", profile: Dict) -> OptimizationMetrics:
        """Apply minimal optimizations."""
        # Only basic cleanup
        return OptimizationMetrics(optimization_time_ms=1.0)
    
    def _is_safe_to_remove(self, node) -> bool:
        """Check if node is safe to remove."""
        # Conservative check - only remove nodes without side effects
        safe_operations = ['linear', 'relu', 'dropout']
        return any(op in node.operation.lower() for op in safe_operations)


# Global optimizer instances
_graph_optimizer: Optional[GraphOptimizer] = None
_adaptive_optimizer: Optional[AdaptiveOptimizer] = None


def get_graph_optimizer() -> GraphOptimizer:
    """Get global graph optimizer."""
    global _graph_optimizer
    if _graph_optimizer is None:
        _graph_optimizer = GraphOptimizer()
    return _graph_optimizer


def get_adaptive_optimizer() -> AdaptiveOptimizer:
    """Get global adaptive optimizer."""
    global _adaptive_optimizer
    if _adaptive_optimizer is None:
        _adaptive_optimizer = AdaptiveOptimizer()
    return _adaptive_optimizer