"""Optimization passes for Spike IR."""

from typing import Dict, List, Set, Any
from abc import ABC, abstractmethod
from .spike_graph import SpikeGraph, SpikeNode, NodeType


class IRPass(ABC):
    """Base class for IR transformation passes."""
    
    def __init__(self, name: str):
        self.name = name
        self.statistics = {}
        
    @abstractmethod
    def run(self, graph: SpikeGraph) -> SpikeGraph:
        """Run the pass on the graph."""
        pass
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get pass execution statistics."""
        return self.statistics.copy()


class DeadCodeElimination(IRPass):
    """Remove unused nodes and edges from the graph."""
    
    def __init__(self):
        super().__init__("DeadCodeElimination")
        
    def run(self, graph: SpikeGraph) -> SpikeGraph:
        """Remove dead nodes that don't contribute to outputs."""
        self.statistics = {"nodes_removed": 0, "edges_removed": 0}
        
        # Find all reachable nodes from outputs
        reachable = set()
        output_nodes = [node.id for node in graph.nodes.values() 
                       if node.node_type == NodeType.OUTPUT]
        
        def mark_reachable(node_id: str):
            if node_id in reachable:
                return
            reachable.add(node_id)
            for pred in graph.get_predecessors(node_id):
                mark_reachable(pred)
                
        for output_id in output_nodes:
            mark_reachable(output_id)
            
        # Remove unreachable nodes
        nodes_to_remove = []
        for node_id in graph.nodes:
            if node_id not in reachable:
                nodes_to_remove.append(node_id)
                
        for node_id in nodes_to_remove:
            del graph.nodes[node_id]
            self.statistics["nodes_removed"] += 1
            
        # Remove edges connected to removed nodes
        edges_to_remove = []
        for i, edge in enumerate(graph.edges):
            if edge.source not in graph.nodes or edge.target not in graph.nodes:
                edges_to_remove.append(i)
                
        for i in reversed(edges_to_remove):
            graph.edges.pop(i)
            self.statistics["edges_removed"] += 1
            
        return graph


class SpikeFusion(IRPass):
    """Fuse consecutive spike operations for efficiency."""
    
    def __init__(self):
        super().__init__("SpikeFusion")
        
    def run(self, graph: SpikeGraph) -> SpikeGraph:
        """Fuse compatible spike operations."""
        self.statistics = {"fusions_performed": 0}
        
        # Find fusible operation pairs
        fusible_pairs = self._find_fusible_pairs(graph)
        
        for source_id, target_id in fusible_pairs:
            if source_id in graph.nodes and target_id in graph.nodes:
                self._fuse_nodes(graph, source_id, target_id)
                self.statistics["fusions_performed"] += 1
                
        return graph
        
    def _find_fusible_pairs(self, graph: SpikeGraph) -> List[tuple]:
        """Find pairs of nodes that can be fused."""
        fusible = []
        
        for node_id, node in graph.nodes.items():
            successors = graph.get_successors(node_id)
            
            # Only fuse if single successor
            if len(successors) == 1:
                successor_id = successors[0]
                successor = graph.get_node(successor_id)
                
                # Check if operations are fusible
                if self._can_fuse(node, successor):
                    fusible.append((node_id, successor_id))
                    
        return fusible
        
    def _can_fuse(self, node1: SpikeNode, node2: SpikeNode) -> bool:
        """Check if two nodes can be fused."""
        # Fuse consecutive linear operations
        if (node1.node_type == NodeType.SPIKE_LINEAR and 
            node2.node_type == NodeType.SPIKE_LINEAR):
            return True
            
        # Fuse neuron + activation
        if (node1.node_type == NodeType.SPIKE_NEURON and
            node2.node_type == NodeType.SPIKE_NEURON and
            node1.get_parameter("neuron_model") == node2.get_parameter("neuron_model")):
            return True
            
        return False
        
    def _fuse_nodes(self, graph: SpikeGraph, source_id: str, target_id: str) -> None:
        """Fuse two nodes into one."""
        source = graph.get_node(source_id)
        target = graph.get_node(target_id)
        
        # Create fused node
        fused_id = f"{source_id}_fused_{target_id}"
        fused_node = SpikeNode(
            id=fused_id,
            node_type=source.node_type,
            operation=f"fused_{source.operation}_{target.operation}",
            inputs=source.inputs.copy(),
            outputs=target.outputs.copy(),
            parameters={**source.parameters, **target.parameters},
            metadata={**source.metadata, **target.metadata}
        )
        
        # Add fused node
        graph.add_node(fused_node)
        
        # Redirect edges
        for edge in graph.edges[:]:  # Copy list to avoid modification during iteration
            if edge.target == source_id:
                edge.target = fused_id
            elif edge.source == target_id:
                edge.source = fused_id
            elif edge.source == source_id and edge.target == target_id:
                # Remove internal edge
                graph.edges.remove(edge)
                
        # Remove original nodes
        del graph.nodes[source_id]
        del graph.nodes[target_id]


class CommonSubexpressionElimination(IRPass):
    """Eliminate common subexpressions."""
    
    def __init__(self):
        super().__init__("CommonSubexpressionElimination")
        
    def run(self, graph: SpikeGraph) -> SpikeGraph:
        """Remove duplicate computations."""
        self.statistics = {"eliminations": 0}
        
        # Group nodes by operation signature
        signatures = {}
        for node_id, node in graph.nodes.items():
            sig = self._get_signature(node)
            if sig not in signatures:
                signatures[sig] = []
            signatures[sig].append(node_id)
            
        # Find duplicates
        for sig, node_list in signatures.items():
            if len(node_list) > 1:
                # Keep first node, redirect others
                canonical = node_list[0]
                for duplicate in node_list[1:]:
                    self._redirect_node(graph, duplicate, canonical)
                    self.statistics["eliminations"] += 1
                    
        return graph
        
    def _get_signature(self, node: SpikeNode) -> str:
        """Get operation signature for comparison."""
        # Create signature from operation type and key parameters
        params = node.parameters.copy()
        # Remove non-semantic parameters
        params.pop("name", None)
        
        return f"{node.operation}_{sorted(params.items())}"
        
    def _redirect_node(self, graph: SpikeGraph, old_id: str, new_id: str) -> None:
        """Redirect all references from old node to new node."""
        # Redirect edges
        for edge in graph.edges:
            if edge.source == old_id:
                edge.source = new_id
            if edge.target == old_id:
                edge.target = new_id
                
        # Remove old node
        if old_id in graph.nodes:
            del graph.nodes[old_id]


class MemoryOptimization(IRPass):
    """Optimize memory usage through buffer reuse."""
    
    def __init__(self):
        super().__init__("MemoryOptimization")
        
    def run(self, graph: SpikeGraph) -> SpikeGraph:
        """Optimize memory allocation."""
        self.statistics = {"memory_saved": 0}
        
        # Analyze buffer lifetimes
        lifetimes = self._analyze_lifetimes(graph)
        
        # Find reusable buffers
        reuse_plan = self._plan_buffer_reuse(lifetimes)
        
        # Apply buffer sharing
        self._apply_buffer_sharing(graph, reuse_plan)
        
        return graph
        
    def _analyze_lifetimes(self, graph: SpikeGraph) -> Dict[str, tuple]:
        """Analyze buffer lifetimes."""
        # Simple implementation: assume linear execution order
        execution_order = graph.topological_sort()
        lifetimes = {}
        
        for i, node_id in enumerate(execution_order):
            node = graph.get_node(node_id)
            # Buffer is live from creation to last use
            last_use = i
            for j in range(i + 1, len(execution_order)):
                if execution_order[j] in graph.get_successors(node_id):
                    last_use = j
            lifetimes[node_id] = (i, last_use)
            
        return lifetimes
        
    def _plan_buffer_reuse(self, lifetimes: Dict[str, tuple]) -> Dict[str, str]:
        """Plan which buffers can be reused."""
        reuse_plan = {}
        
        # Simple greedy algorithm
        for node_id, (start, end) in lifetimes.items():
            for other_id, (other_start, other_end) in lifetimes.items():
                if node_id != other_id and end < other_start:
                    # Non-overlapping lifetimes, can reuse
                    reuse_plan[other_id] = node_id
                    break
                    
        return reuse_plan
        
    def _apply_buffer_sharing(self, graph: SpikeGraph, reuse_plan: Dict[str, str]) -> None:
        """Apply buffer sharing plan."""
        for node_id, reused_buffer in reuse_plan.items():
            node = graph.get_node(node_id)
            if node:
                node.metadata["shared_buffer"] = reused_buffer
                # Estimate memory savings
                memory_estimate = node.metadata.get("memory_estimate", 0)
                self.statistics["memory_saved"] += memory_estimate


class PassManager:
    """Manages execution of optimization passes."""
    
    def __init__(self):
        self.passes = []
        
    def add_pass(self, pass_instance: IRPass) -> None:
        """Add a pass to the pipeline."""
        self.passes.append(pass_instance)
        
    def run_all(self, graph: SpikeGraph) -> SpikeGraph:
        """Run all passes in sequence."""
        current_graph = graph
        
        for pass_instance in self.passes:
            current_graph = pass_instance.run(current_graph)
            
            # Verify graph after each pass
            if not current_graph.verify():
                raise RuntimeError(f"Graph verification failed after {pass_instance.name}")
                
        return current_graph
        
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all passes."""
        return {pass_inst.name: pass_inst.get_statistics() for pass_inst in self.passes}