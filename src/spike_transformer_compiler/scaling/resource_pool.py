"""Advanced resource pooling and auto-scaling for neuromorphic compilation."""

import asyncio
import time
import threading
import psutil
import queue
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from collections import deque, defaultdict
from pathlib import Path
import weakref
import gc
import json

from ..logging_config import compiler_logger
from ..performance import PerformanceProfiler, ResourceMonitor
from ..exceptions import ResourceError, ConfigurationError
from ..config import get_compiler_config


@dataclass
class ResourceRequest:
    """Represents a resource allocation request."""
    request_id: str
    resource_type: str
    amount: int
    priority: int = 0
    max_wait_time: float = 300.0  # 5 minutes
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return time.time() - self.created_at > self.max_wait_time


@dataclass
class ResourcePool:
    """Resource pool configuration and state."""
    pool_name: str
    resource_type: str
    min_size: int
    max_size: int
    current_size: int = 0
    available: int = 0
    in_use: int = 0
    idle_resources: deque = field(default_factory=deque)
    active_resources: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_scaled: float = field(default_factory=time.time)
    scale_cooldown: float = 30.0  # 30 seconds
    
    def can_scale_up(self) -> bool:
        """Check if pool can scale up."""
        return (self.current_size < self.max_size and 
                time.time() - self.last_scaled > self.scale_cooldown)
    
    def can_scale_down(self) -> bool:
        """Check if pool can scale down."""
        return (self.current_size > self.min_size and 
                time.time() - self.last_scaled > self.scale_cooldown and
                len(self.idle_resources) > 1)
    
    def utilization(self) -> float:
        """Get current utilization ratio."""
        if self.current_size == 0:
            return 0.0
        return self.in_use / self.current_size


class AdvancedResourcePool:
    """Advanced resource pool with auto-scaling and load balancing."""
    
    def __init__(self, enable_auto_scaling: bool = True):
        self.enable_auto_scaling = enable_auto_scaling
        self.pools: Dict[str, ResourcePool] = {}
        self.pending_requests: queue.PriorityQueue = queue.PriorityQueue()
        self.request_history: deque = deque(maxlen=10000)
        
        # Thread safety
        self.pool_lock = threading.RLock()
        self.scaling_lock = threading.Lock()
        
        # Auto-scaling configuration
        self.scale_up_threshold = 0.8  # Scale up at 80% utilization
        self.scale_down_threshold = 0.3  # Scale down at 30% utilization
        self.scale_up_factor = 1.5  # Increase by 50%
        self.scale_down_factor = 0.8  # Decrease by 20%
        
        # Background services
        self.resource_monitor = ResourceMonitor()
        self.performance_profiler = PerformanceProfiler(enable_advanced=True)
        
        # Auto-scaling thread
        self.scaling_thread = None
        self.shutdown_event = threading.Event()
        
        if self.enable_auto_scaling:
            self.start_auto_scaling()
        
        compiler_logger.logger.info("Advanced resource pool initialized")
    
    def create_pool(
        self,
        pool_name: str,
        resource_type: str,
        min_size: int = 1,
        max_size: int = 10,
        initial_size: Optional[int] = None
    ) -> None:
        """Create a new resource pool."""
        if initial_size is None:
            initial_size = min_size
        
        with self.pool_lock:
            if pool_name in self.pools:
                raise ConfigurationError(f"Pool '{pool_name}' already exists")
            
            pool = ResourcePool(
                pool_name=pool_name,
                resource_type=resource_type,
                min_size=min_size,
                max_size=max_size,
                current_size=0
            )
            
            self.pools[pool_name] = pool
            
            # Pre-populate pool
            self._scale_pool(pool_name, initial_size)
        
        compiler_logger.logger.info(f"Created resource pool '{pool_name}': {resource_type}, size {initial_size}")
    
    def _scale_pool(self, pool_name: str, target_size: int) -> int:
        """Scale pool to target size."""
        pool = self.pools[pool_name]
        
        if target_size > pool.current_size:
            # Scale up
            added = self._add_resources(pool, target_size - pool.current_size)
            compiler_logger.logger.info(f"Scaled up pool '{pool_name}' by {added} resources")
            return added
        elif target_size < pool.current_size:
            # Scale down
            removed = self._remove_resources(pool, pool.current_size - target_size)
            compiler_logger.logger.info(f"Scaled down pool '{pool_name}' by {removed} resources")
            return -removed
        
        return 0
    
    def _add_resources(self, pool: ResourcePool, count: int) -> int:
        """Add resources to pool."""
        added = 0
        for _ in range(count):
            if pool.current_size >= pool.max_size:
                break
            
            try:
                resource = self._create_resource(pool.resource_type)
                if resource:
                    resource_id = f"{pool.pool_name}_{pool.current_size}_{int(time.time() * 1000)}"
                    pool.idle_resources.append((resource_id, resource))
                    pool.current_size += 1
                    pool.available += 1
                    added += 1
            except Exception as e:
                compiler_logger.logger.error(f"Failed to create resource: {e}")
                break
        
        pool.last_scaled = time.time()
        return added
    
    def _remove_resources(self, pool: ResourcePool, count: int) -> int:
        """Remove idle resources from pool."""
        removed = 0
        for _ in range(count):
            if pool.current_size <= pool.min_size or not pool.idle_resources:
                break
            
            try:
                resource_id, resource = pool.idle_resources.popleft()
                self._destroy_resource(resource)
                pool.current_size -= 1
                pool.available -= 1
                removed += 1
            except Exception as e:
                compiler_logger.logger.error(f"Failed to remove resource: {e}")
                break
        
        pool.last_scaled = time.time()
        return removed
    
    def _create_resource(self, resource_type: str) -> Any:
        """Factory method to create resources."""
        if resource_type == "thread_executor":
            return ThreadPoolExecutor(max_workers=1)
        elif resource_type == "process_executor":
            return ProcessPoolExecutor(max_workers=1)
        elif resource_type == "memory_buffer":
            import numpy as np
            return np.empty(1024 * 1024, dtype=np.float32)  # 4MB buffer
        elif resource_type == "compilation_worker":
            return CompilationWorker()
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")
    
    def _destroy_resource(self, resource: Any) -> None:
        """Clean up resource."""
        try:
            if hasattr(resource, 'shutdown'):
                resource.shutdown(wait=True)
            elif hasattr(resource, 'close'):
                resource.close()
            # For memory buffers, just let GC handle it
        except Exception as e:
            compiler_logger.logger.warning(f"Error destroying resource: {e}")
    
    def acquire_resource(
        self,
        pool_name: str,
        timeout: float = 60.0,
        priority: int = 0
    ) -> Optional[Tuple[str, Any]]:
        """Acquire a resource from the pool."""
        request_id = f"req_{int(time.time() * 1000)}_{threading.get_ident()}"
        
        with self.pool_lock:
            if pool_name not in self.pools:
                raise ValueError(f"Pool '{pool_name}' does not exist")
            
            pool = self.pools[pool_name]
            
            # Try immediate allocation
            if pool.idle_resources:
                resource_id, resource = pool.idle_resources.popleft()
                pool.active_resources[resource_id] = resource
                pool.available -= 1
                pool.in_use += 1
                
                compiler_logger.logger.debug(f"Acquired resource {resource_id} from pool {pool_name}")
                return resource_id, resource
        
        # No immediate resource available, check auto-scaling
        if self.enable_auto_scaling:
            self._trigger_scaling_if_needed(pool_name)
        
        # Wait for resource to become available
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)  # Small delay
            
            with self.pool_lock:
                pool = self.pools[pool_name]
                if pool.idle_resources:
                    resource_id, resource = pool.idle_resources.popleft()
                    pool.active_resources[resource_id] = resource
                    pool.available -= 1
                    pool.in_use += 1
                    
                    compiler_logger.logger.debug(f"Acquired resource {resource_id} from pool {pool_name} after wait")
                    return resource_id, resource
        
        # Timeout reached
        compiler_logger.logger.warning(f"Resource acquisition timeout for pool {pool_name}")
        return None
    
    def release_resource(self, pool_name: str, resource_id: str) -> bool:
        """Release a resource back to the pool."""
        with self.pool_lock:
            if pool_name not in self.pools:
                return False
            
            pool = self.pools[pool_name]
            
            if resource_id not in pool.active_resources:
                return False
            
            resource = pool.active_resources.pop(resource_id)
            pool.idle_resources.append((resource_id, resource))
            pool.available += 1
            pool.in_use -= 1
            
            compiler_logger.logger.debug(f"Released resource {resource_id} to pool {pool_name}")
            return True
    
    def _trigger_scaling_if_needed(self, pool_name: str) -> None:
        """Check if scaling is needed and trigger it."""
        with self.scaling_lock:
            pool = self.pools[pool_name]
            utilization = pool.utilization()
            
            # Scale up if high utilization
            if (utilization > self.scale_up_threshold and 
                pool.can_scale_up()):
                
                target_size = min(
                    pool.max_size,
                    int(pool.current_size * self.scale_up_factor)
                )
                self._scale_pool(pool_name, target_size)
                
            # Scale down if low utilization
            elif (utilization < self.scale_down_threshold and 
                  pool.can_scale_down()):
                
                target_size = max(
                    pool.min_size,
                    int(pool.current_size * self.scale_down_factor)
                )
                self._scale_pool(pool_name, target_size)
    
    def start_auto_scaling(self) -> None:
        """Start background auto-scaling service."""
        def auto_scaling_loop():
            while not self.shutdown_event.wait(10):  # Check every 10 seconds
                try:
                    self._auto_scale_all_pools()
                except Exception as e:
                    compiler_logger.logger.error(f"Auto-scaling error: {e}")
        
        self.scaling_thread = threading.Thread(target=auto_scaling_loop, daemon=True)
        self.scaling_thread.start()
        compiler_logger.logger.info("Auto-scaling service started")
    
    def _auto_scale_all_pools(self) -> None:
        """Auto-scale all pools based on current utilization."""
        for pool_name in list(self.pools.keys()):
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Adjust scaling thresholds based on system load
                if cpu_percent > 90 or memory_percent > 90:
                    # System under stress, be conservative
                    continue
                
                self._trigger_scaling_if_needed(pool_name)
                
            except Exception as e:
                compiler_logger.logger.error(f"Error auto-scaling pool {pool_name}: {e}")
    
    def get_pool_stats(self, pool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for pools."""
        with self.pool_lock:
            if pool_name:
                if pool_name not in self.pools:
                    return {}
                
                pool = self.pools[pool_name]
                return {
                    'pool_name': pool.pool_name,
                    'resource_type': pool.resource_type,
                    'current_size': pool.current_size,
                    'available': pool.available,
                    'in_use': pool.in_use,
                    'utilization': pool.utilization(),
                    'min_size': pool.min_size,
                    'max_size': pool.max_size,
                    'uptime': time.time() - pool.creation_time,
                    'last_scaled': time.time() - pool.last_scaled
                }
            else:
                # All pools stats
                stats = {}
                for name, pool in self.pools.items():
                    stats[name] = self.get_pool_stats(name)
                return stats
    
    def cleanup(self) -> None:
        """Cleanup all resources and stop services."""
        # Stop auto-scaling
        self.shutdown_event.set()
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        # Cleanup all pools
        with self.pool_lock:
            for pool_name, pool in list(self.pools.items()):
                # Release active resources
                for resource in pool.active_resources.values():
                    self._destroy_resource(resource)
                
                # Release idle resources
                while pool.idle_resources:
                    _, resource = pool.idle_resources.popleft()
                    self._destroy_resource(resource)
                
                del self.pools[pool_name]
        
        # Cleanup monitors
        if hasattr(self.resource_monitor, 'cleanup'):
            self.resource_monitor.cleanup()
        if hasattr(self.performance_profiler, 'cleanup'):
            self.performance_profiler.cleanup()
        
        compiler_logger.logger.info("Resource pool cleanup completed")


class CompilationWorker:
    """Worker for compilation tasks."""
    
    def __init__(self):
        self.worker_id = f"worker_{int(time.time() * 1000)}"
        self.busy = False
        self.tasks_completed = 0
        self.created_at = time.time()
    
    def compile_model(self, model, **kwargs):
        """Compile a model (placeholder)."""
        self.busy = True
        try:
            # Simulate compilation work
            time.sleep(0.1)
            self.tasks_completed += 1
            return f"compiled_model_{self.tasks_completed}"
        finally:
            self.busy = False
    
    def is_busy(self) -> bool:
        """Check if worker is busy."""
        return self.busy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'tasks_completed': self.tasks_completed,
            'uptime': time.time() - self.created_at,
            'busy': self.busy
        }


class LoadBalancer:
    """Load balancer for distributing compilation tasks."""
    
    def __init__(self, resource_pool: AdvancedResourcePool):
        self.resource_pool = resource_pool
        self.task_queue = queue.Queue()
        self.completed_tasks = {}
        self.task_stats = defaultdict(int)
        self.balancing_strategies = {
            'round_robin': self._round_robin_balance,
            'least_loaded': self._least_loaded_balance,
            'weighted': self._weighted_balance
        }
        self.current_strategy = 'least_loaded'
    
    def submit_task(
        self,
        task_func: Callable,
        *args,
        pool_name: str = "compilation_workers",
        priority: int = 0,
        timeout: float = 300.0,
        **kwargs
    ) -> str:
        """Submit a task for load-balanced execution."""
        task_id = f"task_{int(time.time() * 1000)}_{threading.get_ident()}"
        
        task = {
            'task_id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'pool_name': pool_name,
            'priority': priority,
            'timeout': timeout,
            'submitted_at': time.time()
        }
        
        self.task_queue.put((priority, task))
        self.task_stats['submitted'] += 1
        
        compiler_logger.logger.debug(f"Submitted task {task_id} to load balancer")
        return task_id
    
    def process_tasks(self) -> None:
        """Process tasks from queue with load balancing."""
        while True:
            try:
                _, task = self.task_queue.get(timeout=1.0)
                self._execute_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                compiler_logger.logger.error(f"Error processing task: {e}")
    
    def _execute_task(self, task: Dict[str, Any]) -> None:
        """Execute a single task."""
        task_id = task['task_id']
        
        try:
            # Acquire resource
            resource_result = self.resource_pool.acquire_resource(
                task['pool_name'],
                timeout=task['timeout']
            )
            
            if not resource_result:
                self.task_stats['failed'] += 1
                self.completed_tasks[task_id] = {
                    'status': 'failed',
                    'error': 'Resource acquisition timeout',
                    'completed_at': time.time()
                }
                return
            
            resource_id, resource = resource_result
            
            try:
                # Execute task
                start_time = time.time()
                result = task['func'](*task['args'], **task['kwargs'])
                end_time = time.time()
                
                self.task_stats['completed'] += 1
                self.completed_tasks[task_id] = {
                    'status': 'completed',
                    'result': result,
                    'duration': end_time - start_time,
                    'completed_at': end_time,
                    'resource_id': resource_id
                }
                
                compiler_logger.logger.debug(f"Task {task_id} completed in {end_time - start_time:.3f}s")
                
            finally:
                # Release resource
                self.resource_pool.release_resource(task['pool_name'], resource_id)
                
        except Exception as e:
            self.task_stats['failed'] += 1
            self.completed_tasks[task_id] = {
                'status': 'failed',
                'error': str(e),
                'completed_at': time.time()
            }
            compiler_logger.logger.error(f"Task {task_id} failed: {e}")
    
    def _round_robin_balance(self, available_resources: List[Any]) -> Any:
        """Round-robin resource selection."""
        # Simple implementation - would maintain state for true round-robin
        return available_resources[0] if available_resources else None
    
    def _least_loaded_balance(self, available_resources: List[Any]) -> Any:
        """Select least loaded resource."""
        if not available_resources:
            return None
        
        # For compilation workers, select least busy
        least_loaded = min(
            available_resources,
            key=lambda r: getattr(r[1], 'tasks_completed', 0) if hasattr(r[1], 'tasks_completed') else 0
        )
        return least_loaded
    
    def _weighted_balance(self, available_resources: List[Any]) -> Any:
        """Weighted resource selection based on performance."""
        # Placeholder for weighted selection
        return self._least_loaded_balance(available_resources)
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result by ID."""
        return self.completed_tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'queue_size': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'task_stats': dict(self.task_stats),
            'current_strategy': self.current_strategy,
            'available_strategies': list(self.balancing_strategies.keys())
        }


# Global resource pool instance
_resource_pool: Optional[AdvancedResourcePool] = None


def get_resource_pool() -> AdvancedResourcePool:
    """Get global resource pool instance."""
    global _resource_pool
    if _resource_pool is None:
        config = get_compiler_config()
        _resource_pool = AdvancedResourcePool(
            enable_auto_scaling=config.enable_auto_scaling
        )
        
        # Initialize default pools
        _resource_pool.create_pool(
            "compilation_workers",
            "compilation_worker",
            min_size=2,
            max_size=8
        )
        _resource_pool.create_pool(
            "thread_executors",
            "thread_executor", 
            min_size=1,
            max_size=4
        )
        _resource_pool.create_pool(
            "memory_buffers",
            "memory_buffer",
            min_size=5,
            max_size=20
        )
    
    return _resource_pool


def cleanup_resource_pool() -> None:
    """Cleanup global resource pool."""
    global _resource_pool
    if _resource_pool:
        _resource_pool.cleanup()
        _resource_pool = None