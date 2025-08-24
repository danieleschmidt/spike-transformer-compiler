"""Advanced scaling and performance optimization."""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
import queue
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import weakref


class ScalingStrategy(Enum):
    """Different scaling strategies."""
    VERTICAL = "vertical"        # Scale up (more resources per instance)
    HORIZONTAL = "horizontal"    # Scale out (more instances)
    HYBRID = "hybrid"           # Combination of both
    ELASTIC = "elastic"         # Dynamic scaling based on load


class LoadBalancerType(Enum):
    """Load balancer algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    LEAST_RESPONSE_TIME = "least_response_time"


@dataclass
class WorkerNode:
    """Worker node information."""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    max_capacity: int = 10
    active_tasks: int = 0
    last_heartbeat: float = 0.0
    status: str = "idle"  # idle, busy, overloaded, offline
    
    def __post_init__(self):
        self.last_heartbeat = time.time()


@dataclass
class CompilationTask:
    """Compilation task representation."""
    task_id: str
    model_hash: str
    input_shape: Tuple[int, ...]
    target: str
    priority: int = 5  # 1-10, higher is more priority
    created_at: float = field(default_factory=time.time)
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedCompilationManager:
    """Manage distributed compilation across multiple nodes."""
    
    def __init__(self, 
                 max_workers: int = None,
                 load_balancer: LoadBalancerType = LoadBalancerType.RESOURCE_BASED,
                 enable_caching: bool = True):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.load_balancer_type = load_balancer
        self.enable_caching = enable_caching
        
        self.logger = logging.getLogger(__name__)
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, CompilationTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Performance monitoring
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_duration": 0.0,
            "peak_concurrency": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Caching system
        if self.enable_caching:
            self.compilation_cache = CompilationCache(max_size=1000)
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="CompilerWorker"
        )
        
        # Background tasks
        self._running = False
        self._background_tasks = []
    
    async def start(self):
        """Start the distributed compilation system."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize worker nodes (for this single-machine setup, create virtual nodes)
        await self._initialize_worker_nodes()
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._load_balancer()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._cleanup_task())
        ]
        
        self.logger.info(f"Distributed compilation manager started with {len(self.worker_nodes)} nodes")
    
    async def stop(self):
        """Stop the distributed compilation system."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Distributed compilation manager stopped")
    
    async def submit_compilation(self, 
                               model: Any,
                               input_shape: Tuple[int, ...],
                               target: str,
                               priority: int = 5,
                               **kwargs) -> str:
        """Submit a compilation task."""
        # Generate task ID and model hash
        task_id = self._generate_task_id()
        model_hash = self._compute_model_hash(model, input_shape, target)
        
        # Check cache first
        if self.enable_caching:
            cached_result = await self.compilation_cache.get(model_hash)
            if cached_result:
                self.metrics["cache_hits"] += 1
                self.completed_tasks[task_id] = cached_result
                self.logger.info(f"Task {task_id} served from cache")
                return task_id
            else:
                self.metrics["cache_misses"] += 1
        
        # Estimate resource requirements
        resource_reqs = self._estimate_resource_requirements(model, input_shape)
        
        # Create task
        task = CompilationTask(
            task_id=task_id,
            model_hash=model_hash,
            input_shape=input_shape,
            target=target,
            priority=priority,
            estimated_duration=resource_reqs.get("estimated_duration", 1.0),
            resource_requirements=resource_reqs,
            metadata={"model": model, **kwargs}
        )
        
        # Add to queue (priority queue orders by priority)
        await self.task_queue.put((-priority, time.time(), task))
        self.active_tasks[task_id] = task
        self.metrics["total_tasks"] += 1
        
        self.logger.info(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = 60.0) -> Any:
        """Get compilation result for a task."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                result = self.completed_tasks.pop(task_id)
                if isinstance(result, Exception):
                    raise result
                return result
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a compilation task."""
        if task_id in self.completed_tasks:
            return {"status": "completed", "result_available": True}
        elif task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "status": "active",
                "priority": task.priority,
                "created_at": task.created_at,
                "estimated_duration": task.estimated_duration,
                "queue_position": await self._get_queue_position(task_id)
            }
        else:
            return {"status": "not_found"}
    
    async def _initialize_worker_nodes(self):
        """Initialize virtual worker nodes for this system."""
        for i in range(self.max_workers):
            node_id = f"worker_{i:02d}"
            node = WorkerNode(
                node_id=node_id,
                host="localhost",
                port=8000 + i,
                capabilities={
                    "targets": ["simulation", "loihi3"],
                    "max_model_size": 1000000,
                    "supported_dtypes": ["float32", "int8"]
                },
                max_capacity=5  # Max concurrent compilations per worker
            )
            self.worker_nodes[node_id] = node
    
    async def _heartbeat_monitor(self):
        """Monitor worker node health."""
        while self._running:
            try:
                current_time = time.time()
                
                for node in self.worker_nodes.values():
                    # Simulate heartbeat (in real system, would ping actual nodes)
                    node.last_heartbeat = current_time
                    
                    # Update node status based on load
                    if node.active_tasks >= node.max_capacity:
                        node.status = "overloaded"
                    elif node.active_tasks > node.max_capacity * 0.8:
                        node.status = "busy"
                    else:
                        node.status = "idle"
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(1)
    
    async def _load_balancer(self):
        """Load balancer to distribute tasks to nodes."""
        while self._running:
            try:
                if not self.task_queue.empty():
                    # Get next task
                    _, _, task = await self.task_queue.get()
                    
                    # Find best node for the task
                    node = self._select_best_node(task)
                    
                    if node:
                        # Execute task on selected node
                        await self._execute_task_on_node(task, node)
                    else:
                        # No available nodes, put task back in queue
                        await self.task_queue.put((-task.priority, time.time(), task))
                        await asyncio.sleep(1)  # Wait before retrying
                else:
                    await asyncio.sleep(0.1)  # No tasks, wait briefly
                    
            except Exception as e:
                self.logger.error(f"Load balancer error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task_on_node(self, task: CompilationTask, node: WorkerNode):
        """Execute a compilation task on a specific node."""
        node.active_tasks += 1
        node.current_load = node.active_tasks / node.max_capacity
        
        try:
            # Submit to thread pool for actual execution
            future = self.thread_pool.submit(self._compile_task, task)
            
            # Wait for completion (non-blocking)
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            # Store result
            self.completed_tasks[task.task_id] = result
            self.metrics["completed_tasks"] += 1
            
            # Cache result if enabled
            if self.enable_caching and not isinstance(result, Exception):
                await self.compilation_cache.set(task.model_hash, result)
            
            self.logger.info(f"Task {task.task_id} completed on {node.node_id}")
            
        except Exception as e:
            self.completed_tasks[task.task_id] = e
            self.metrics["failed_tasks"] += 1
            self.logger.error(f"Task {task.task_id} failed on {node.node_id}: {e}")
            
        finally:
            node.active_tasks -= 1
            node.current_load = node.active_tasks / node.max_capacity
            self.active_tasks.pop(task.task_id, None)
    
    def _compile_task(self, task: CompilationTask) -> Any:
        """Actually compile the model (runs in thread pool)."""
        try:
            from .compiler import SpikeCompiler
            
            # Extract model and parameters from task
            model = task.metadata["model"]
            
            # Create compiler
            compiler = SpikeCompiler(target=task.target, verbose=False)
            
            # Compile model
            compiled_model = compiler.compile(model, input_shape=task.input_shape)
            
            return compiled_model
            
        except Exception as e:
            self.logger.error(f"Compilation error for task {task.task_id}: {e}")
            raise e
    
    def _select_best_node(self, task: CompilationTask) -> Optional[WorkerNode]:
        """Select the best node for a task based on load balancing strategy."""
        available_nodes = [
            node for node in self.worker_nodes.values()
            if node.status in ["idle", "busy"] and node.active_tasks < node.max_capacity
        ]
        
        if not available_nodes:
            return None
        
        if self.load_balancer_type == LoadBalancerType.ROUND_ROBIN:
            # Simple round robin
            return min(available_nodes, key=lambda n: n.active_tasks)
            
        elif self.load_balancer_type == LoadBalancerType.LEAST_CONNECTIONS:
            # Node with fewest active tasks
            return min(available_nodes, key=lambda n: n.active_tasks)
            
        elif self.load_balancer_type == LoadBalancerType.RESOURCE_BASED:
            # Node with lowest resource utilization
            return min(available_nodes, key=lambda n: n.current_load)
            
        elif self.load_balancer_type == LoadBalancerType.LEAST_RESPONSE_TIME:
            # For now, just use least connections
            return min(available_nodes, key=lambda n: n.active_tasks)
            
        else:
            # Default to least connections
            return min(available_nodes, key=lambda n: n.active_tasks)
    
    async def _metrics_collector(self):
        """Collect performance metrics."""
        while self._running:
            try:
                # Update peak concurrency
                current_concurrency = sum(node.active_tasks for node in self.worker_nodes.values())
                self.metrics["peak_concurrency"] = max(self.metrics["peak_concurrency"], current_concurrency)
                
                # Calculate average duration (simplified)
                if self.metrics["completed_tasks"] > 0:
                    # This is a simplified calculation
                    self.metrics["average_duration"] = 1.0  # Would calculate from actual task durations
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_task(self):
        """Clean up old completed tasks and cache."""
        while self._running:
            try:
                current_time = time.time()
                
                # Clean up old completed tasks (keep for 1 hour)
                expired_tasks = [
                    task_id for task_id in self.completed_tasks
                    if current_time - self.active_tasks.get(task_id, CompilationTask("", "", (), "", 0)).created_at > 3600
                ]
                
                for task_id in expired_tasks:
                    self.completed_tasks.pop(task_id, None)
                
                # Clean up cache if enabled
                if self.enable_caching:
                    await self.compilation_cache.cleanup()
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    async def _get_queue_position(self, task_id: str) -> int:
        """Get position of task in queue."""
        # This is a simplified implementation
        return self.task_queue.qsize()
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        return f"task_{int(time.time() * 1000000)}"
    
    def _compute_model_hash(self, model: Any, input_shape: Tuple[int, ...], target: str) -> str:
        """Compute hash for model caching."""
        # Create a hash based on model representation and parameters
        model_str = f"{type(model).__name__}_{input_shape}_{target}"
        
        # Add model-specific information if available
        if hasattr(model, 'input_size') and hasattr(model, 'output_size'):
            model_str += f"_{model.input_size}_{model.output_size}"
        
        return hashlib.md5(model_str.encode()).hexdigest()
    
    def _estimate_resource_requirements(self, model: Any, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Estimate resource requirements for compilation."""
        # Simplified resource estimation
        total_params = 1
        for dim in input_shape:
            total_params *= dim
        
        # Add model-specific parameters if available
        if hasattr(model, 'input_size') and hasattr(model, 'output_size'):
            total_params += model.input_size * model.output_size
        
        # Estimate duration based on model complexity
        estimated_duration = max(0.1, min(10.0, total_params / 10000))  # 0.1-10 seconds
        
        return {
            "memory_mb": max(10, total_params // 1000),
            "cpu_cores": 1,
            "estimated_duration": estimated_duration,
            "model_complexity": "simple" if total_params < 1000 else "complex"
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        active_tasks = sum(node.active_tasks for node in self.worker_nodes.values())
        
        return {
            **self.metrics,
            "current_active_tasks": active_tasks,
            "worker_nodes": len(self.worker_nodes),
            "queue_size": self.task_queue.qsize(),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if self.enable_caching else 0.0
            ),
            "success_rate": (
                self.metrics["completed_tasks"] / max(1, self.metrics["completed_tasks"] + self.metrics["failed_tasks"])
            ),
            "node_utilization": [
                {
                    "node_id": node.node_id,
                    "current_load": node.current_load,
                    "active_tasks": node.active_tasks,
                    "status": node.status
                }
                for node in self.worker_nodes.values()
            ]
        }


class CompilationCache:
    """High-performance compilation cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        self.cache: Dict[str, Tuple[Any, float]] = {}  # hash -> (result, timestamp)
        self.access_order: List[str] = []  # LRU tracking
        self._lock = threading.RLock()
    
    async def get(self, model_hash: str) -> Optional[Any]:
        """Get cached result."""
        with self._lock:
            if model_hash in self.cache:
                result, timestamp = self.cache[model_hash]
                
                # Check if expired
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[model_hash]
                    self.access_order.remove(model_hash)
                    return None
                
                # Update LRU order
                self.access_order.remove(model_hash)
                self.access_order.append(model_hash)
                
                return result
            
            return None
    
    async def set(self, model_hash: str, result: Any):
        """Cache a compilation result."""
        with self._lock:
            current_time = time.time()
            
            # Remove oldest entries if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            # Add new entry
            self.cache[model_hash] = (result, current_time)
            self.access_order.append(model_hash)
    
    async def cleanup(self):
        """Clean up expired cache entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self.cache.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "ttl_seconds": self.ttl_seconds
            }


class AutoScaler:
    """Automatic scaling based on load metrics."""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = 16,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.logger = logging.getLogger(__name__)
        self.scaling_history: List[Dict[str, Any]] = []
    
    async def evaluate_scaling(self, manager: DistributedCompilationManager) -> Optional[str]:
        """Evaluate if scaling is needed."""
        metrics = manager.get_performance_metrics()
        
        current_workers = metrics["worker_nodes"]
        avg_utilization = sum(node["current_load"] for node in metrics["node_utilization"]) / current_workers
        queue_size = metrics["queue_size"]
        
        scaling_decision = None
        
        # Scale up conditions
        if (avg_utilization > self.scale_up_threshold or queue_size > current_workers * 2) and current_workers < self.max_workers:
            scaling_decision = "scale_up"
            new_workers = min(current_workers * 2, self.max_workers)
            
        # Scale down conditions
        elif avg_utilization < self.scale_down_threshold and queue_size == 0 and current_workers > self.min_workers:
            scaling_decision = "scale_down"
            new_workers = max(current_workers // 2, self.min_workers)
        
        if scaling_decision:
            self.scaling_history.append({
                "timestamp": time.time(),
                "decision": scaling_decision,
                "old_workers": current_workers,
                "new_workers": new_workers,
                "avg_utilization": avg_utilization,
                "queue_size": queue_size
            })
            
            self.logger.info(f"Scaling decision: {scaling_decision} ({current_workers} -> {new_workers} workers)")
        
        return scaling_decision


# High-level API for easy usage
class ScalableCompiler:
    """High-level scalable compiler interface."""
    
    def __init__(self, **kwargs):
        self.manager = DistributedCompilationManager(**kwargs)
        self.auto_scaler = AutoScaler()
        self._started = False
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """Start the scalable compilation system."""
        if not self._started:
            await self.manager.start()
            self._started = True
    
    async def stop(self):
        """Stop the scalable compilation system."""
        if self._started:
            await self.manager.stop()
            self._started = False
    
    async def compile_async(self, 
                          model: Any,
                          input_shape: Tuple[int, ...],
                          target: str = "simulation",
                          priority: int = 5,
                          timeout: float = 60.0,
                          **kwargs) -> Any:
        """Compile model asynchronously with automatic scaling."""
        if not self._started:
            await self.start()
        
        # Submit compilation task
        task_id = await self.manager.submit_compilation(
            model=model,
            input_shape=input_shape,
            target=target,
            priority=priority,
            **kwargs
        )
        
        # Get result
        result = await self.manager.get_result(task_id, timeout=timeout)
        
        # Check if scaling is needed
        scaling_decision = await self.auto_scaler.evaluate_scaling(self.manager)
        if scaling_decision:
            # In a real system, would trigger actual scaling here
            pass
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        if not self._started:
            return {"status": "stopped"}
        
        return {
            "status": "running",
            "performance_metrics": self.manager.get_performance_metrics(),
            "scaling_history": self.auto_scaler.scaling_history[-10:],  # Last 10 scaling decisions
        }


async def demo_scalable_compilation():
    """Demonstrate scalable compilation system."""
    from .mock_models import create_test_model
    
    print("🚀 Scalable Compilation System Demo")
    print("=" * 40)
    
    async with ScalableCompiler(max_workers=4, enable_caching=True) as compiler:
        
        # Submit multiple compilation tasks
        tasks = []
        for i in range(5):
            model = create_test_model("simple", input_size=10 + i, output_size=5)
            
            print(f"Submitting task {i+1}...")
            task = asyncio.create_task(compiler.compile_async(
                model=model,
                input_shape=(1, 10 + i),
                target="simulation",
                priority=5 - i  # Higher priority for earlier tasks
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        print("Waiting for compilations to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Show results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"✓ Completed {successful}/{len(results)} compilations")
        
        # Show performance metrics
        status = compiler.get_status()
        metrics = status["performance_metrics"]
        print(f"\n📊 Performance Metrics:")
        print(f"  Total tasks: {metrics['total_tasks']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        print(f"  Peak concurrency: {metrics['peak_concurrency']}")
        print(f"  Current queue size: {metrics['queue_size']}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_scalable_compilation())