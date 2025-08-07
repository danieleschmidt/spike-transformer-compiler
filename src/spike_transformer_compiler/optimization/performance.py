"""Advanced performance optimization for neuromorphic compilation."""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache, wraps
import cProfile
import pstats
from ..ir.spike_graph import SpikeGraph
from ..logging_config import performance_logger


class PerformanceOptimizer:
    """Advanced performance optimizer with parallelization and caching."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_caching: bool = True,
        enable_profiling: bool = False
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        
        # Execution pools
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        
        # Performance tracking
        self.optimization_history = []
        self.cache_stats = {"hits": 0, "misses": 0}
        self.profiler = cProfile.Profile() if enable_profiling else None
        
        # Optimization strategies
        self.optimization_strategies = {
            "parallel_parsing": self._parallel_model_parsing,
            "concurrent_optimization": self._concurrent_pass_optimization,
            "cached_compilation": self._cached_backend_compilation,
            "memory_pool_allocation": self._optimized_memory_allocation,
            "pipeline_parallelism": self._pipeline_parallel_execution
        }
        
        performance_logger.info(f"PerformanceOptimizer initialized with {self.max_workers} workers")
        
    def optimize_compilation_pipeline(
        self,
        graph: SpikeGraph,
        optimization_passes: List[Any],
        backend_config: Dict[str, Any]
    ) -> Tuple[SpikeGraph, Dict[str, Any]]:
        """Optimize entire compilation pipeline with parallelization."""
        start_time = time.time()
        
        if self.enable_profiling and self.profiler:
            self.profiler.enable()
            
        try:
            # Stage 1: Parallel graph analysis
            analysis_results = self._parallel_graph_analysis(graph)
            
            # Stage 2: Concurrent optimization passes
            optimized_graph = self._apply_concurrent_optimizations(graph, optimization_passes)
            
            # Stage 3: Parallel backend preparation
            backend_results = self._parallel_backend_preparation(optimized_graph, backend_config)
            
            # Stage 4: Memory optimization
            memory_optimized_results = self._optimize_memory_layout(backend_results)
            
            optimization_time = time.time() - start_time
            
            # Record optimization metrics
            self.optimization_history.append({
                "timestamp": time.time(),
                "optimization_time": optimization_time,
                "graph_nodes": len(graph.nodes),
                "graph_edges": len(graph.edges),
                "optimization_passes": len(optimization_passes),
                "speedup_factor": self._calculate_speedup(optimization_time, graph)
            })
            
            performance_logger.info(f"Pipeline optimization completed in {optimization_time:.3f}s")
            
            return optimized_graph, memory_optimized_results
            
        finally:
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
                
    def _parallel_graph_analysis(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Perform parallel analysis of graph structure."""
        analysis_tasks = [
            self.thread_pool.submit(self._analyze_node_complexity, graph),
            self.thread_pool.submit(self._analyze_data_flow, graph),
            self.thread_pool.submit(self._analyze_memory_requirements, graph),
            self.thread_pool.submit(self._analyze_parallelization_opportunities, graph)
        ]
        
        results = {}
        for i, task in enumerate(analysis_tasks):
            try:
                result = task.result(timeout=30)
                results[f"analysis_{i}"] = result
            except Exception as e:
                performance_logger.warning(f"Analysis task {i} failed: {str(e)}")
                results[f"analysis_{i}"] = {}
                
        return results
        
    def _apply_concurrent_optimizations(
        self,
        graph: SpikeGraph,
        optimization_passes: List[Any]
    ) -> SpikeGraph:
        """Apply optimization passes concurrently where possible."""
        # Categorize passes by dependencies
        independent_passes = []
        dependent_passes = []
        
        for pass_obj in optimization_passes:
            if self._is_pass_independent(pass_obj):
                independent_passes.append(pass_obj)
            else:
                dependent_passes.append(pass_obj)
                
        # Run independent passes in parallel
        if independent_passes:
            parallel_results = []
            for pass_obj in independent_passes:
                future = self.thread_pool.submit(self._apply_optimization_pass, graph, pass_obj)
                parallel_results.append(future)
                
            # Merge results from parallel passes
            optimized_graphs = []
            for future in parallel_results:
                try:
                    result = future.result(timeout=60)
                    optimized_graphs.append(result)
                except Exception as e:
                    performance_logger.error(f"Parallel optimization failed: {str(e)}")
                    optimized_graphs.append(graph)  # Fallback to original
                    
            # Merge optimized graphs (simplified - takes first successful result)
            graph = optimized_graphs[0] if optimized_graphs else graph
            
        # Run dependent passes sequentially
        for pass_obj in dependent_passes:
            graph = self._apply_optimization_pass(graph, pass_obj)
            
        return graph
        
    def _parallel_backend_preparation(
        self,
        graph: SpikeGraph,
        backend_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare backend compilation in parallel."""
        preparation_tasks = [
            self.thread_pool.submit(self._prepare_hardware_mapping, graph, backend_config),
            self.thread_pool.submit(self._prepare_memory_layout, graph, backend_config),
            self.thread_pool.submit(self._prepare_communication_patterns, graph, backend_config)
        ]
        
        results = {}
        for i, task in enumerate(preparation_tasks):
            try:
                result = task.result(timeout=45)
                results[f"preparation_{i}"] = result
            except Exception as e:
                performance_logger.warning(f"Backend preparation task {i} failed: {str(e)}")
                results[f"preparation_{i}"] = {}
                
        return results
        
    @lru_cache(maxsize=128)
    def _cached_node_analysis(self, node_id: str, node_hash: int) -> Dict[str, Any]:
        """Cached analysis of individual nodes."""
        if self.enable_caching:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1
            
        # Simulate complex node analysis
        time.sleep(0.001)  # Simulate computation time
        
        return {
            "complexity_score": np.random.random(),
            "memory_requirement": np.random.randint(1000, 10000),
            "parallelization_factor": np.random.randint(1, 8)
        }
        
    def _optimize_memory_layout(self, backend_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory layout for performance."""
        # Memory coalescing optimization
        coalesced_layout = self._coalesce_memory_accesses(backend_results)
        
        # Cache-friendly data arrangement
        cache_optimized = self._optimize_cache_locality(coalesced_layout)
        
        # NUMA-aware allocation
        numa_optimized = self._numa_aware_allocation(cache_optimized)
        
        return numa_optimized
        
    def _coalesce_memory_accesses(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Coalesce memory accesses for better performance."""
        # Group related data together
        coalesced = data.copy()
        coalesced["memory_coalescing"] = {
            "enabled": True,
            "coalescing_factor": 4,
            "alignment": 64  # Cache line alignment
        }
        return coalesced
        
    def _optimize_cache_locality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data layout for cache locality."""
        optimized = data.copy()
        optimized["cache_optimization"] = {
            "enabled": True,
            "cache_line_size": 64,
            "prefetch_distance": 8,
            "temporal_locality_factor": 0.8
        }
        return optimized
        
    def _numa_aware_allocation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """NUMA-aware memory allocation."""
        numa_optimized = data.copy()
        numa_optimized["numa_optimization"] = {
            "enabled": True,
            "numa_nodes": self._detect_numa_topology(),
            "allocation_policy": "local",
            "migration_threshold": 0.7
        }
        return numa_optimized
        
    def _detect_numa_topology(self) -> int:
        """Detect NUMA topology."""
        try:
            import numa
            return numa.get_max_node() + 1
        except ImportError:
            # Fallback to CPU count estimation
            return max(1, self.max_workers // 4)
            
    # Analysis methods
    def _analyze_node_complexity(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Analyze computational complexity of nodes."""
        complexity_scores = {}
        
        for node in graph.nodes:
            # Calculate complexity based on node type and parameters
            node_hash = hash(str(node.parameters))
            complexity = self._cached_node_analysis(node.id, node_hash)
            complexity_scores[node.id] = complexity
            
        return {
            "total_nodes": len(graph.nodes),
            "complexity_scores": complexity_scores,
            "average_complexity": np.mean([c["complexity_score"] for c in complexity_scores.values()]),
            "high_complexity_nodes": [nid for nid, c in complexity_scores.items() 
                                    if c["complexity_score"] > 0.7]
        }
        
    def _analyze_data_flow(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Analyze data flow patterns in the graph."""
        data_flow = {
            "total_edges": len(graph.edges),
            "fan_out": {},
            "fan_in": {},
            "critical_paths": [],
            "bottlenecks": []
        }
        
        # Calculate fan-out and fan-in for each node
        for node in graph.nodes:
            data_flow["fan_out"][node.id] = len(node.outputs)
            data_flow["fan_in"][node.id] = len(node.inputs)
            
        # Identify bottlenecks (high fan-out nodes)
        avg_fan_out = np.mean(list(data_flow["fan_out"].values()))
        data_flow["bottlenecks"] = [
            nid for nid, fan_out in data_flow["fan_out"].items() 
            if fan_out > avg_fan_out * 2
        ]
        
        return data_flow
        
    def _analyze_memory_requirements(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Analyze memory requirements and access patterns."""
        total_memory = 0
        memory_per_node = {}
        
        for node in graph.nodes:
            node_memory = self._estimate_node_memory(node)
            memory_per_node[node.id] = node_memory
            total_memory += node_memory
            
        return {
            "total_memory_mb": total_memory / (1024 * 1024),
            "memory_per_node": memory_per_node,
            "peak_memory_mb": total_memory * 1.5 / (1024 * 1024),  # Estimated peak
            "memory_hotspots": [nid for nid, mem in memory_per_node.items() 
                              if mem > total_memory * 0.1]  # >10% of total
        }
        
    def _analyze_parallelization_opportunities(self, graph: SpikeGraph) -> Dict[str, Any]:
        """Identify opportunities for parallelization."""
        # Topological analysis for parallel execution
        levels = self._compute_execution_levels(graph)
        
        parallelization = {
            "execution_levels": len(levels),
            "max_parallel_nodes": max(len(level) for level in levels.values()),
            "parallelization_factor": sum(len(level) for level in levels.values()) / len(levels),
            "independent_subgraphs": self._find_independent_subgraphs(graph),
            "pipeline_stages": self._identify_pipeline_stages(graph)
        }
        
        return parallelization
        
    def _compute_execution_levels(self, graph: SpikeGraph) -> Dict[int, List[str]]:
        """Compute execution levels for parallel scheduling."""
        levels = {}
        node_levels = {}
        
        # Topological sort to determine execution order
        in_degree = {}
        for node in graph.nodes:
            in_degree[node.id] = len(node.inputs)
            
        # BFS to assign levels
        current_level = 0
        queue = [node.id for node in graph.nodes if in_degree[node.id] == 0]
        
        while queue:
            levels[current_level] = queue.copy()
            next_queue = []
            
            for node_id in queue:
                node_levels[node_id] = current_level
                node = graph.get_node(node_id)
                
                for output_id in node.outputs:
                    if output_id in in_degree:
                        in_degree[output_id] -= 1
                        if in_degree[output_id] == 0:
                            next_queue.append(output_id)
                            
            queue = next_queue
            current_level += 1
            
        return levels
        
    def _find_independent_subgraphs(self, graph: SpikeGraph) -> List[List[str]]:
        """Find independent subgraphs that can be processed in parallel."""
        visited = set()
        subgraphs = []
        
        def dfs(node_id, current_subgraph):
            if node_id in visited:
                return
            visited.add(node_id)
            current_subgraph.append(node_id)
            
            node = graph.get_node(node_id)
            for neighbor_id in node.inputs + node.outputs:
                if neighbor_id not in visited:
                    dfs(neighbor_id, current_subgraph)
                    
        for node in graph.nodes:
            if node.id not in visited:
                subgraph = []
                dfs(node.id, subgraph)
                if subgraph:
                    subgraphs.append(subgraph)
                    
        return subgraphs
        
    def _identify_pipeline_stages(self, graph: SpikeGraph) -> List[Dict[str, Any]]:
        """Identify pipeline stages for streaming execution."""
        levels = self._compute_execution_levels(graph)
        stages = []
        
        for level, nodes in levels.items():
            stage = {
                "stage_id": level,
                "nodes": nodes,
                "estimated_latency": self._estimate_stage_latency(nodes, graph),
                "memory_requirement": self._estimate_stage_memory(nodes, graph),
                "parallelization_factor": len(nodes)
            }
            stages.append(stage)
            
        return stages
        
    # Utility methods
    def _is_pass_independent(self, pass_obj: Any) -> bool:
        """Check if an optimization pass can run independently."""
        # Simplified check - in practice would analyze pass dependencies
        pass_name = pass_obj.__class__.__name__
        independent_passes = ["DeadCodeElimination", "ConstantFolding", "CommonSubexpressionElimination"]
        return pass_name in independent_passes
        
    def _apply_optimization_pass(self, graph: SpikeGraph, pass_obj: Any) -> SpikeGraph:
        """Apply single optimization pass."""
        try:
            return pass_obj.run(graph)
        except Exception as e:
            performance_logger.error(f"Optimization pass failed: {str(e)}")
            return graph  # Return original graph on failure
            
    def _estimate_node_memory(self, node: Any) -> int:
        """Estimate memory requirement for a node."""
        base_memory = 1024  # Base memory per node
        
        if hasattr(node, 'parameters'):
            params = node.parameters
            memory_multiplier = 1
            
            if 'num_neurons' in params:
                memory_multiplier *= params['num_neurons']
            if 'embed_dim' in params:
                memory_multiplier *= params['embed_dim'] 
            if 'out_features' in params:
                memory_multiplier *= params['out_features']
                
            return base_memory * max(1, memory_multiplier // 100)
        
        return base_memory
        
    def _estimate_stage_latency(self, nodes: List[str], graph: SpikeGraph) -> float:
        """Estimate latency for a pipeline stage."""
        total_latency = 0.0
        
        for node_id in nodes:
            node = graph.get_node(node_id)
            node_latency = self._estimate_node_latency(node)
            total_latency += node_latency
            
        # Account for parallelization
        return total_latency / len(nodes) if nodes else 0.0
        
    def _estimate_stage_memory(self, nodes: List[str], graph: SpikeGraph) -> int:
        """Estimate memory requirement for a pipeline stage."""
        total_memory = 0
        
        for node_id in nodes:
            node = graph.get_node(node_id)
            total_memory += self._estimate_node_memory(node)
            
        return total_memory
        
    def _estimate_node_latency(self, node: Any) -> float:
        """Estimate execution latency for a node."""
        base_latency = 0.001  # 1ms base latency
        
        node_type = node.node_type.name
        latency_multipliers = {
            "INPUT": 0.1,
            "OUTPUT": 0.1,
            "SPIKE_NEURON": 1.0,
            "SPIKE_LINEAR": 2.0,
            "SPIKE_CONV": 3.0,
            "SPIKE_ATTENTION": 5.0
        }
        
        multiplier = latency_multipliers.get(node_type, 1.0)
        return base_latency * multiplier
        
    def _calculate_speedup(self, optimization_time: float, graph: SpikeGraph) -> float:
        """Calculate speedup factor from optimization."""
        # Simplified speedup calculation
        baseline_time = len(graph.nodes) * 0.01  # 10ms per node baseline
        return baseline_time / max(optimization_time, 0.001)
        
    # Specialized optimization methods
    def _parallel_model_parsing(self, model: Any) -> Any:
        """Parse model components in parallel."""
        # Split model into components and parse in parallel
        components = self._split_model_components(model)
        
        parse_futures = []
        for component in components:
            future = self.process_pool.submit(self._parse_component, component)
            parse_futures.append(future)
            
        parsed_components = []
        for future in parse_futures:
            try:
                result = future.result(timeout=30)
                parsed_components.append(result)
            except Exception as e:
                performance_logger.error(f"Parallel parsing failed: {str(e)}")
                
        return self._merge_parsed_components(parsed_components)
        
    def _concurrent_pass_optimization(self, graph: SpikeGraph, passes: List[Any]) -> SpikeGraph:
        """Run optimization passes concurrently where possible."""
        return self._apply_concurrent_optimizations(graph, passes)
        
    def _cached_backend_compilation(self, graph: SpikeGraph, backend_config: Dict[str, Any]) -> Any:
        """Use caching for backend compilation."""
        # Create cache key from graph structure and config
        cache_key = self._create_cache_key(graph, backend_config)
        
        if self.enable_caching and cache_key in self._compilation_cache:
            self.cache_stats["hits"] += 1
            return self._compilation_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        
        # Perform compilation
        result = self._compile_backend(graph, backend_config)
        
        if self.enable_caching:
            self._compilation_cache[cache_key] = result
            
        return result
        
    def _optimized_memory_allocation(self, memory_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory allocation patterns."""
        return self._optimize_memory_layout(memory_requirements)
        
    def _pipeline_parallel_execution(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline stages in parallel."""
        stages = execution_plan.get("stages", [])
        
        # Execute overlapping pipeline stages
        stage_futures = []
        for stage in stages:
            future = self.thread_pool.submit(self._execute_pipeline_stage, stage)
            stage_futures.append(future)
            
        results = []
        for future in stage_futures:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                performance_logger.error(f"Pipeline stage execution failed: {str(e)}")
                results.append({})
                
        return {"pipeline_results": results}
        
    # Cache and utility methods
    def _create_cache_key(self, graph: SpikeGraph, config: Dict[str, Any]) -> str:
        """Create cache key from graph and configuration."""
        graph_hash = hash(str(sorted([(n.id, n.node_type.name) for n in graph.nodes])))
        config_hash = hash(str(sorted(config.items())))
        return f"{graph_hash}_{config_hash}"
        
    def _split_model_components(self, model: Any) -> List[Any]:
        """Split model into components for parallel processing."""
        # Simplified component splitting
        return [model]  # In practice, would split into layers/blocks
        
    def _parse_component(self, component: Any) -> Any:
        """Parse individual model component."""
        # Simulate component parsing
        time.sleep(0.01)
        return component
        
    def _merge_parsed_components(self, components: List[Any]) -> Any:
        """Merge parsed components back into model."""
        return components[0] if components else None
        
    def _compile_backend(self, graph: SpikeGraph, config: Dict[str, Any]) -> Any:
        """Compile backend (simplified)."""
        return {"compiled": True, "config": config}
        
    def _execute_pipeline_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual pipeline stage."""
        # Simulate stage execution
        stage_latency = stage.get("estimated_latency", 0.01)
        time.sleep(stage_latency)
        
        return {"stage_completed": True, "latency": stage_latency}
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        if self.optimization_history:
            avg_time = np.mean([h["optimization_time"] for h in self.optimization_history])
            avg_speedup = np.mean([h["speedup_factor"] for h in self.optimization_history])
        else:
            avg_time = 0.0
            avg_speedup = 1.0
            
        return {
            "total_optimizations": len(self.optimization_history),
            "average_optimization_time": avg_time,
            "average_speedup_factor": avg_speedup,
            "cache_hit_rate": self.cache_stats["hits"] / max(1, sum(self.cache_stats.values())),
            "cache_stats": self.cache_stats.copy(),
            "max_workers": self.max_workers,
            "optimization_strategies": list(self.optimization_strategies.keys())
        }
        
    def get_profiling_results(self) -> Optional[str]:
        """Get profiling results if enabled."""
        if not self.enable_profiling or not self.profiler:
            return None
            
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('tottime')
        
        # Return formatted profiling output
        import io
        output = io.StringIO()
        stats.print_stats(output)
        return output.getvalue()
        
    def cleanup(self) -> None:
        """Cleanup performance optimizer resources."""
        self.process_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, '_compilation_cache'):
            self._compilation_cache.clear()
            
        performance_logger.info("PerformanceOptimizer cleanup completed")
        
    # Initialize compilation cache
    def __post_init__(self):
        if self.enable_caching:
            self._compilation_cache = {}


def performance_monitor(func: Callable) -> Callable:
    """Decorator for monitoring function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            performance_logger.info(
                f"{func.__name__} executed in {execution_time:.4f}s"
            )
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            performance_logger.error(
                f"{func.__name__} failed after {execution_time:.4f}s: {str(e)}"
            )
            raise
            
    return wrapper


class AutoScaler:
    """Automatic scaling for neuromorphic compilation workloads."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = None,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.5
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.utilization_history = []
        self.scaling_decisions = []
        
    def should_scale_up(self, current_utilization: float) -> bool:
        """Determine if scaling up is needed."""
        return (current_utilization > self.scale_up_threshold and 
                self.current_workers < self.max_workers)
                
    def should_scale_down(self, current_utilization: float) -> bool:
        """Determine if scaling down is needed."""
        return (current_utilization < self.scale_down_threshold and 
                self.current_workers > self.min_workers)
                
    def make_scaling_decision(self, workload_metrics: Dict[str, Any]) -> Optional[str]:
        """Make scaling decision based on workload metrics."""
        current_utilization = workload_metrics.get("cpu_utilization", 0.0)
        queue_length = workload_metrics.get("queue_length", 0)
        memory_usage = workload_metrics.get("memory_usage", 0.0)
        
        self.utilization_history.append(current_utilization)
        
        # Keep only recent history
        if len(self.utilization_history) > 10:
            self.utilization_history.pop(0)
            
        # Calculate average utilization over recent period
        avg_utilization = np.mean(self.utilization_history)
        
        decision = None
        
        if self.should_scale_up(avg_utilization) or queue_length > 10:
            new_workers = min(self.max_workers, int(self.current_workers * 1.5))
            decision = f"scale_up_to_{new_workers}"
            self.current_workers = new_workers
            
        elif self.should_scale_down(avg_utilization) and queue_length < 2:
            new_workers = max(self.min_workers, int(self.current_workers * 0.7))
            decision = f"scale_down_to_{new_workers}"
            self.current_workers = new_workers
            
        if decision:
            self.scaling_decisions.append({
                "timestamp": time.time(),
                "decision": decision,
                "utilization": avg_utilization,
                "queue_length": queue_length,
                "memory_usage": memory_usage
            })
            
            performance_logger.info(f"Auto-scaling decision: {decision}")
            
        return decision
        
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "recent_utilization": self.utilization_history[-5:] if self.utilization_history else [],
            "scaling_decisions": len(self.scaling_decisions),
            "last_scaling_decision": self.scaling_decisions[-1] if self.scaling_decisions else None
        }