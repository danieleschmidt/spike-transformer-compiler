"""Hyperscale Performance Engine: Advanced performance optimization and auto-scaling.

This module implements hyperscale performance optimization including adaptive caching,
intelligent load balancing, auto-scaling triggers, and distributed compilation
for the Spike-Transformer-Compiler system.
"""

import time
import json
import asyncio
import threading
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from collections import deque, defaultdict
import statistics
import numpy as np
import hashlib
import pickle
import queue
from enum import Enum
import psutil
from functools import lru_cache, wraps
import weakref


class ScalingTrigger(Enum):
    """Auto-scaling trigger types."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CACHING = "caching"
    PARALLELIZATION = "parallelization"
    VECTORIZATION = "vectorization"
    MEMORY_POOLING = "memory_pooling"
    PIPELINE_OPTIMIZATION = "pipeline_optimization"
    LOAD_BALANCING = "load_balancing"
    PREFETCHING = "prefetching"
    COMPRESSION = "compression"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    throughput: float  # operations per second
    latency: float     # average response time in ms
    queue_length: int
    error_rate: float
    cache_hit_rate: float = 0.0
    network_utilization: float = 0.0
    disk_io_rate: float = 0.0
    active_workers: int = 0


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int = 1
    max_instances: int = 100
    cooldown_period: float = 300.0  # seconds
    evaluation_window: int = 5  # number of metrics to consider
    scale_factor: float = 1.5  # multiplier for scaling


class AdaptiveCacheManager:
    """Advanced adaptive caching with intelligent eviction policies."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_accesses": 0
        }
        self._lock = threading.RLock()
        
        # Adaptive parameters
        self.hit_rate_target = 0.8
        self.adaptation_rate = 0.1
        self.size_adaptation_factor = 1.1
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with adaptive optimization."""
        with self._lock:
            self.cache_stats["total_accesses"] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if time.time() - entry["timestamp"] > self.ttl:
                    del self.cache[key]
                    self._cleanup_metadata(key)
                    self.cache_stats["misses"] += 1
                    return default
                
                # Update access metadata
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                self.cache_stats["hits"] += 1
                
                return entry["value"]
            else:
                self.cache_stats["misses"] += 1
                return default
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._adaptive_eviction()
            
            # Store item
            self.cache[key] = {
                "value": value,
                "timestamp": current_time
            }
            
            # Initialize metadata
            if key not in self.access_counts:
                self.access_counts[key] = 1
                self.access_times[key] = current_time
            
            # Adapt cache size based on performance
            self._adapt_cache_parameters()
    
    def _adaptive_eviction(self) -> None:
        """Intelligent cache eviction using multiple factors."""
        if not self.cache:
            return
        
        current_time = time.time()
        
        # Calculate eviction scores for all items
        eviction_scores = {}
        for key in self.cache.keys():
            entry = self.cache[key]
            age = current_time - entry["timestamp"]
            frequency = self.access_counts.get(key, 1)
            recency = current_time - self.access_times.get(key, current_time)
            
            # Combined score: higher score = more likely to evict
            # Factors: age (older = higher score), low frequency, low recency
            score = (age / self.ttl) * 0.4 + (1.0 / frequency) * 0.4 + (recency / 3600) * 0.2
            eviction_scores[key] = score
        
        # Evict items with highest scores
        eviction_count = max(1, len(self.cache) // 10)  # Evict 10% at a time
        keys_to_evict = sorted(eviction_scores.keys(), 
                              key=lambda k: eviction_scores[k], 
                              reverse=True)[:eviction_count]
        
        for key in keys_to_evict:
            del self.cache[key]
            self._cleanup_metadata(key)
            self.cache_stats["evictions"] += 1
    
    def _cleanup_metadata(self, key: str) -> None:
        """Clean up metadata for evicted key."""
        self.access_counts.pop(key, None)
        self.access_times.pop(key, None)
    
    def _adapt_cache_parameters(self) -> None:
        """Adapt cache parameters based on performance."""
        if self.cache_stats["total_accesses"] < 100:
            return  # Need sufficient data
        
        hit_rate = self.get_hit_rate()
        
        # Adapt cache size
        if hit_rate < self.hit_rate_target:
            # Poor hit rate, consider increasing cache size
            new_max_size = int(self.max_size * self.size_adaptation_factor)
            if new_max_size <= 10000:  # Reasonable upper limit
                self.max_size = new_max_size
        elif hit_rate > 0.95:
            # Excellent hit rate, consider reducing cache size
            new_max_size = int(self.max_size / self.size_adaptation_factor)
            if new_max_size >= 100:  # Reasonable lower limit
                self.max_size = new_max_size
    
    def get_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        return self.cache_stats["hits"] / max(1, total)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": self.get_hit_rate(),
                "stats": self.cache_stats.copy(),
                "avg_access_count": statistics.mean(self.access_counts.values()) if self.access_counts else 0
            }
    
    def clear(self) -> None:
        """Clear cache and reset statistics."""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "total_accesses": 0}


class IntelligentLoadBalancer:
    """Intelligent load balancer with adaptive routing."""
    
    def __init__(self):
        self.workers = {}
        self.worker_metrics = defaultdict(lambda: deque(maxlen=100))
        self.routing_strategy = "least_connections"
        self.health_check_interval = 30.0
        self.load_factors = defaultdict(float)
        self._lock = threading.RLock()
        
    def register_worker(self, worker_id: str, worker_info: Dict[str, Any]) -> None:
        """Register a worker for load balancing."""
        with self._lock:
            self.workers[worker_id] = {
                "info": worker_info,
                "active_connections": 0,
                "total_requests": 0,
                "avg_response_time": 0.0,
                "error_rate": 0.0,
                "last_seen": time.time(),
                "healthy": True,
                "capacity": worker_info.get("capacity", 100)
            }
            self.load_factors[worker_id] = 1.0
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from load balancing."""
        with self._lock:
            self.workers.pop(worker_id, None)
            self.worker_metrics.pop(worker_id, None)
            self.load_factors.pop(worker_id, None)
    
    def select_worker(self, request_metadata: Dict[str, Any] = None) -> Optional[str]:
        """Select optimal worker based on current load and performance."""
        with self._lock:
            healthy_workers = {wid: worker for wid, worker in self.workers.items() 
                             if worker["healthy"]}
            
            if not healthy_workers:
                return None
            
            if self.routing_strategy == "least_connections":
                return self._select_least_connections(healthy_workers)
            elif self.routing_strategy == "weighted_round_robin":
                return self._select_weighted_round_robin(healthy_workers)
            elif self.routing_strategy == "performance_based":
                return self._select_performance_based(healthy_workers, request_metadata)
            else:
                # Default to round robin
                return list(healthy_workers.keys())[0]
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]) -> None:
        """Update worker performance metrics."""
        with self._lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            worker["last_seen"] = time.time()
            
            # Update metrics
            if "response_time" in metrics:
                # Exponential moving average
                alpha = 0.1
                current_avg = worker["avg_response_time"]
                worker["avg_response_time"] = alpha * metrics["response_time"] + (1 - alpha) * current_avg
            
            if "error_occurred" in metrics:
                # Track error rate
                alpha = 0.1
                error = 1.0 if metrics["error_occurred"] else 0.0
                worker["error_rate"] = alpha * error + (1 - alpha) * worker["error_rate"]
            
            if "request_completed" in metrics:
                worker["total_requests"] += 1
                if metrics.get("connection_ended", False):
                    worker["active_connections"] = max(0, worker["active_connections"] - 1)
            
            if "connection_started" in metrics:
                worker["active_connections"] += 1
            
            # Store metrics history
            self.worker_metrics[worker_id].append({
                "timestamp": time.time(),
                "metrics": metrics.copy()
            })
            
            # Update load factor based on performance
            self._update_load_factor(worker_id)
    
    def _select_least_connections(self, workers: Dict[str, Any]) -> str:
        """Select worker with least active connections."""
        return min(workers.keys(), 
                  key=lambda wid: workers[wid]["active_connections"])
    
    def _select_weighted_round_robin(self, workers: Dict[str, Any]) -> str:
        """Select worker using weighted round robin based on capacity."""
        # Simple implementation - weight by inverse of current load
        weights = {}
        for wid, worker in workers.items():
            load_ratio = worker["active_connections"] / max(1, worker["capacity"])
            weights[wid] = 1.0 / max(0.1, load_ratio)  # Higher weight for lower load
        
        # Select based on weights (simplified random selection)
        total_weight = sum(weights.values())
        if total_weight == 0:
            return list(workers.keys())[0]
        
        import random
        target = random.uniform(0, total_weight)
        current = 0
        for wid, weight in weights.items():
            current += weight
            if current >= target:
                return wid
        
        return list(workers.keys())[0]
    
    def _select_performance_based(self, workers: Dict[str, Any], request_metadata: Dict = None) -> str:
        """Select worker based on comprehensive performance metrics."""
        scores = {}
        
        for wid, worker in workers.items():
            # Calculate composite performance score
            load_score = 1.0 - (worker["active_connections"] / max(1, worker["capacity"]))
            response_time_score = 1.0 / max(0.1, worker["avg_response_time"] + 1)
            error_score = 1.0 - worker["error_rate"]
            load_factor = self.load_factors[wid]
            
            # Weighted combination
            composite_score = (0.4 * load_score + 
                             0.3 * response_time_score + 
                             0.2 * error_score + 
                             0.1 * load_factor)
            
            scores[wid] = composite_score
        
        return max(scores.keys(), key=lambda wid: scores[wid])
    
    def _update_load_factor(self, worker_id: str) -> None:
        """Update load factor based on recent performance."""
        worker = self.workers[worker_id]
        
        # Calculate load factor based on multiple metrics
        utilization = worker["active_connections"] / max(1, worker["capacity"])
        response_factor = min(2.0, worker["avg_response_time"] / 100.0)  # Normalize to ~100ms baseline
        error_factor = 1.0 + worker["error_rate"] * 2.0  # Penalty for errors
        
        # Combined load factor (higher = more loaded)
        new_factor = utilization * response_factor * error_factor
        
        # Smooth update
        alpha = 0.1
        self.load_factors[worker_id] = alpha * new_factor + (1 - alpha) * self.load_factors[worker_id]
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics."""
        with self._lock:
            stats = {
                "total_workers": len(self.workers),
                "healthy_workers": sum(1 for w in self.workers.values() if w["healthy"]),
                "total_capacity": sum(w["capacity"] for w in self.workers.values()),
                "total_active_connections": sum(w["active_connections"] for w in self.workers.values()),
                "worker_details": {}
            }
            
            for wid, worker in self.workers.items():
                stats["worker_details"][wid] = {
                    "active_connections": worker["active_connections"],
                    "capacity": worker["capacity"],
                    "avg_response_time": worker["avg_response_time"],
                    "error_rate": worker["error_rate"],
                    "load_factor": self.load_factors[wid],
                    "healthy": worker["healthy"]
                }
            
            return stats


class AutoScalingManager:
    """Advanced auto-scaling with multiple triggers and policies."""
    
    def __init__(self):
        self.scaling_policies = {}
        self.metrics_history = deque(maxlen=1000)
        self.scaling_actions = deque(maxlen=100)
        self.current_instances = 1
        self.last_scaling_action = 0
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def add_scaling_policy(self, name: str, policy: ScalingPolicy) -> None:
        """Add auto-scaling policy."""
        self.scaling_policies[name] = policy
    
    def update_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update performance metrics for scaling decisions."""
        self.metrics_history.append(metrics)
        
        # Check if scaling action is needed
        if self.monitoring_active:
            self._evaluate_scaling_triggers(metrics)
    
    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for auto-scaling."""
        while self.monitoring_active and not self._stop_event.is_set():
            try:
                if len(self.metrics_history) >= 2:
                    latest_metrics = self.metrics_history[-1]
                    self._evaluate_scaling_triggers(latest_metrics)
                
                # Check for scaling policy optimization
                self._optimize_scaling_policies()
                
            except Exception as e:
                print(f"Error in auto-scaling monitoring: {e}")
            
            self._stop_event.wait(10)  # Check every 10 seconds
    
    def _evaluate_scaling_triggers(self, metrics: PerformanceMetrics) -> None:
        """Evaluate if scaling action is needed based on policies."""
        current_time = time.time()
        
        for policy_name, policy in self.scaling_policies.items():
            # Check cooldown period
            if current_time - self.last_scaling_action < policy.cooldown_period:
                continue
            
            # Get metric value for this trigger
            metric_value = self._get_metric_value(metrics, policy.trigger)
            
            # Get recent metrics for evaluation window
            recent_metrics = list(self.metrics_history)[-policy.evaluation_window:]
            if len(recent_metrics) < policy.evaluation_window:
                continue
            
            recent_values = [self._get_metric_value(m, policy.trigger) for m in recent_metrics]
            avg_value = statistics.mean(recent_values)
            
            # Check for scale up
            if (avg_value > policy.scale_up_threshold and 
                self.current_instances < policy.max_instances):
                
                new_instances = min(
                    policy.max_instances,
                    int(self.current_instances * policy.scale_factor)
                )
                
                self._execute_scaling_action("scale_up", new_instances, policy_name, avg_value)
            
            # Check for scale down
            elif (avg_value < policy.scale_down_threshold and 
                  self.current_instances > policy.min_instances):
                
                new_instances = max(
                    policy.min_instances,
                    int(self.current_instances / policy.scale_factor)
                )
                
                self._execute_scaling_action("scale_down", new_instances, policy_name, avg_value)
    
    def _get_metric_value(self, metrics: PerformanceMetrics, trigger: ScalingTrigger) -> float:
        """Extract metric value for given trigger."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return metrics.cpu_utilization
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return metrics.memory_utilization
        elif trigger == ScalingTrigger.QUEUE_LENGTH:
            return metrics.queue_length
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            return metrics.latency
        elif trigger == ScalingTrigger.THROUGHPUT:
            return metrics.throughput
        elif trigger == ScalingTrigger.ERROR_RATE:
            return metrics.error_rate
        else:
            return 0.0
    
    def _execute_scaling_action(self, action: str, target_instances: int, policy_name: str, metric_value: float) -> None:
        """Execute scaling action."""
        if target_instances == self.current_instances:
            return
        
        scaling_action = {
            "timestamp": time.time(),
            "action": action,
            "from_instances": self.current_instances,
            "to_instances": target_instances,
            "policy": policy_name,
            "trigger_value": metric_value,
            "success": True
        }
        
        try:
            # Simulate scaling action (in real implementation, this would scale actual resources)
            if action == "scale_up":
                print(f"🚀 SCALING UP: {self.current_instances} → {target_instances} instances")
                print(f"   Trigger: {policy_name} (value: {metric_value:.3f})")
            else:
                print(f"🔽 SCALING DOWN: {self.current_instances} → {target_instances} instances")
                print(f"   Trigger: {policy_name} (value: {metric_value:.3f})")
            
            self.current_instances = target_instances
            self.last_scaling_action = time.time()
            
        except Exception as e:
            scaling_action["success"] = False
            scaling_action["error"] = str(e)
            print(f"❌ Scaling action failed: {e}")
        
        self.scaling_actions.append(scaling_action)
    
    def _optimize_scaling_policies(self) -> None:
        """Optimize scaling policies based on historical performance."""
        if len(self.scaling_actions) < 10:
            return  # Need sufficient data
        
        # Analyze recent scaling actions for optimization opportunities
        recent_actions = list(self.scaling_actions)[-10:]
        
        # Check for thrashing (frequent up/down scaling)
        action_types = [action["action"] for action in recent_actions]
        if len(set(action_types)) > 1:  # Mixed scale up/down
            # Consider adjusting thresholds or cooldown periods
            for policy in self.scaling_policies.values():
                if policy.cooldown_period < 600:  # Less than 10 minutes
                    policy.cooldown_period *= 1.2  # Increase cooldown
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        recent_actions = list(self.scaling_actions)[-24:]  # Last 24 actions
        
        scale_up_count = sum(1 for action in recent_actions if action["action"] == "scale_up")
        scale_down_count = sum(1 for action in recent_actions if action["action"] == "scale_down")
        success_rate = sum(1 for action in recent_actions if action["success"]) / max(1, len(recent_actions))
        
        return {
            "current_instances": self.current_instances,
            "total_scaling_actions": len(self.scaling_actions),
            "recent_scale_ups": scale_up_count,
            "recent_scale_downs": scale_down_count,
            "success_rate": success_rate,
            "policies_count": len(self.scaling_policies),
            "last_scaling_action": self.last_scaling_action
        }


class DistributedCompilationManager:
    """Manages distributed compilation across multiple workers."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.worker_pool = None
        self.task_queue = queue.Queue()
        self.result_cache = AdaptiveCacheManager(max_size=500)
        self.compilation_stats = {
            "total_compilations": 0,
            "distributed_compilations": 0,
            "cache_hits": 0,
            "avg_compilation_time": 0.0
        }
        self.active_compilations = {}
        self._lock = threading.RLock()
        
    def start_workers(self) -> None:
        """Start worker pool for distributed compilation."""
        if self.worker_pool is None:
            self.worker_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    def stop_workers(self) -> None:
        """Stop worker pool."""
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
    
    def compile_distributed(
        self,
        compilation_tasks: List[Dict[str, Any]],
        enable_caching: bool = True
    ) -> List[Any]:
        """Execute compilation tasks in distributed manner."""
        if not self.worker_pool:
            self.start_workers()
        
        with self._lock:
            self.compilation_stats["total_compilations"] += len(compilation_tasks)
        
        start_time = time.time()
        results = []
        
        # Check cache first if enabled
        if enable_caching:
            cached_results, remaining_tasks = self._check_cache(compilation_tasks)
            results.extend(cached_results)
        else:
            remaining_tasks = compilation_tasks
        
        if remaining_tasks:
            # Submit tasks to worker pool
            future_to_task = {}
            for task in remaining_tasks:
                future = self.worker_pool.submit(self._compile_task, task)
                future_to_task[future] = task
            
            # Collect results
            distributed_results = []
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    distributed_results.append(result)
                    
                    # Cache result if successful
                    if enable_caching and result.get("success", False):
                        cache_key = self._generate_cache_key(task)
                        self.result_cache.put(cache_key, result)
                    
                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "task_id": task.get("task_id", "unknown")
                    }
                    distributed_results.append(error_result)
            
            results.extend(distributed_results)
            
            with self._lock:
                self.compilation_stats["distributed_compilations"] += len(remaining_tasks)
        
        # Update statistics
        compilation_time = time.time() - start_time
        with self._lock:
            alpha = 0.1
            current_avg = self.compilation_stats["avg_compilation_time"]
            self.compilation_stats["avg_compilation_time"] = \
                alpha * compilation_time + (1 - alpha) * current_avg
        
        return results
    
    def _check_cache(self, tasks: List[Dict[str, Any]]) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Check cache for existing results."""
        cached_results = []
        remaining_tasks = []
        
        for task in tasks:
            cache_key = self._generate_cache_key(task)
            cached_result = self.result_cache.get(cache_key)
            
            if cached_result is not None:
                cached_results.append(cached_result)
                with self._lock:
                    self.compilation_stats["cache_hits"] += 1
            else:
                remaining_tasks.append(task)
        
        return cached_results, remaining_tasks
    
    def _generate_cache_key(self, task: Dict[str, Any]) -> str:
        """Generate cache key for compilation task."""
        # Create deterministic key from task parameters
        key_data = {
            "model_hash": task.get("model_hash", ""),
            "optimization_level": task.get("optimization_level", 0),
            "target": task.get("target", "simulation"),
            "input_shape": str(task.get("input_shape", []))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _compile_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual compilation task."""
        try:
            # Simulate compilation work
            task_id = task.get("task_id", "unknown")
            complexity = task.get("complexity", 1.0)
            
            # Simulate variable compilation time based on complexity
            import random
            compilation_time = complexity * random.uniform(0.5, 2.0)
            time.sleep(min(compilation_time, 5.0))  # Cap at 5 seconds for simulation
            
            # Simulate success/failure
            success_rate = max(0.8, 1.0 - complexity * 0.1)  # Higher complexity = higher failure rate
            success = random.random() < success_rate
            
            if success:
                return {
                    "success": True,
                    "task_id": task_id,
                    "compilation_time": compilation_time,
                    "result": f"compiled_model_{task_id}",
                    "performance_metrics": {
                        "throughput": 1000 / complexity,
                        "energy_efficiency": 0.8 - complexity * 0.1
                    }
                }
            else:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": f"Compilation failed for task {task_id}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "task_id": task.get("task_id", "unknown"),
                "error": str(e)
            }
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get comprehensive compilation statistics."""
        with self._lock:
            cache_stats = self.result_cache.get_stats()
            
            return {
                "compilation_stats": self.compilation_stats.copy(),
                "cache_stats": cache_stats,
                "worker_pool_active": self.worker_pool is not None,
                "max_workers": self.max_workers,
                "active_compilations": len(self.active_compilations)
            }


class HyperscalePerformanceEngine:
    """Main orchestrator for hyperscale performance optimization."""
    
    def __init__(self):
        self.cache_manager = AdaptiveCacheManager(max_size=2000)
        self.load_balancer = IntelligentLoadBalancer()
        self.auto_scaler = AutoScalingManager()
        self.distributed_compiler = DistributedCompilationManager()
        self.performance_history = deque(maxlen=1000)
        self.optimization_strategies = set()
        
        # Performance monitoring
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Initialize default scaling policies
        self._initialize_default_policies()
        
        # Initialize default workers
        self._initialize_default_workers()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default auto-scaling policies."""
        # CPU-based scaling
        cpu_policy = ScalingPolicy(
            trigger=ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            min_instances=1,
            max_instances=20,
            cooldown_period=300.0
        )
        self.auto_scaler.add_scaling_policy("cpu_scaling", cpu_policy)
        
        # Memory-based scaling
        memory_policy = ScalingPolicy(
            trigger=ScalingTrigger.MEMORY_UTILIZATION,
            scale_up_threshold=85.0,
            scale_down_threshold=40.0,
            min_instances=1,
            max_instances=15,
            cooldown_period=240.0
        )
        self.auto_scaler.add_scaling_policy("memory_scaling", memory_policy)
        
        # Latency-based scaling
        latency_policy = ScalingPolicy(
            trigger=ScalingTrigger.RESPONSE_TIME,
            scale_up_threshold=200.0,  # 200ms
            scale_down_threshold=50.0,   # 50ms
            min_instances=1,
            max_instances=25,
            cooldown_period=180.0
        )
        self.auto_scaler.add_scaling_policy("latency_scaling", latency_policy)
    
    def _initialize_default_workers(self) -> None:
        """Initialize default workers for load balancing."""
        # Simulate multiple workers
        for i in range(3):
            worker_id = f"worker_{i}"
            self.load_balancer.register_worker(worker_id, {
                "capacity": 100,
                "type": "compilation",
                "region": "us-east" if i < 2 else "us-west"
            })
    
    def optimize_compilation(
        self,
        compilation_request: Dict[str, Any],
        optimization_strategies: List[OptimizationStrategy] = None
    ) -> Dict[str, Any]:
        """Optimize compilation with hyperscale performance techniques."""
        
        start_time = time.time()
        optimization_result = {
            "original_request": compilation_request,
            "optimizations_applied": [],
            "performance_improvements": {},
            "final_result": None
        }
        
        if optimization_strategies is None:
            optimization_strategies = [
                OptimizationStrategy.CACHING,
                OptimizationStrategy.PARALLELIZATION,
                OptimizationStrategy.LOAD_BALANCING
            ]
        
        # Apply optimizations in order
        current_request = compilation_request.copy()
        
        for strategy in optimization_strategies:
            strategy_start = time.time()
            
            if strategy == OptimizationStrategy.CACHING:
                current_request = self._apply_caching_optimization(current_request)
                optimization_result["optimizations_applied"].append("adaptive_caching")
            
            elif strategy == OptimizationStrategy.LOAD_BALANCING:
                current_request = self._apply_load_balancing_optimization(current_request)
                optimization_result["optimizations_applied"].append("intelligent_load_balancing")
            
            elif strategy == OptimizationStrategy.PARALLELIZATION:
                current_request = self._apply_parallelization_optimization(current_request)
                optimization_result["optimizations_applied"].append("distributed_compilation")
            
            elif strategy == OptimizationStrategy.MEMORY_POOLING:
                current_request = self._apply_memory_pooling_optimization(current_request)
                optimization_result["optimizations_applied"].append("memory_pooling")
            
            strategy_time = time.time() - strategy_start
            optimization_result["performance_improvements"][strategy.value] = {
                "time_taken": strategy_time,
                "applied": True
            }
        
        # Execute optimized compilation
        final_result = self._execute_optimized_compilation(current_request)
        optimization_result["final_result"] = final_result
        
        # Record performance metrics
        total_time = time.time() - start_time
        self._record_performance_metrics(total_time, optimization_result)
        
        return optimization_result
    
    def _apply_caching_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive caching optimization."""
        # Generate cache key for request
        cache_key = self._generate_request_cache_key(request)
        
        # Check cache
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            request["cache_hit"] = True
            request["cached_result"] = cached_result
        else:
            request["cache_hit"] = False
        
        return request
    
    def _apply_load_balancing_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent load balancing optimization."""
        # Select optimal worker
        selected_worker = self.load_balancer.select_worker(request)
        
        if selected_worker:
            request["assigned_worker"] = selected_worker
            request["load_balanced"] = True
        else:
            request["load_balanced"] = False
        
        return request
    
    def _apply_parallelization_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Apply distributed compilation optimization."""
        # Break down compilation into parallel tasks if applicable
        model_size = request.get("model_size", 0)
        
        if model_size > 1000000:  # Large models benefit from parallelization
            # Simulate breaking into subtasks
            num_subtasks = min(8, max(2, model_size // 500000))
            request["parallelization"] = {
                "enabled": True,
                "num_subtasks": num_subtasks,
                "estimated_speedup": min(4.0, num_subtasks * 0.7)
            }
        else:
            request["parallelization"] = {"enabled": False}
        
        return request
    
    def _apply_memory_pooling_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory pooling optimization."""
        # Simulate memory pool allocation
        request["memory_pool"] = {
            "enabled": True,
            "pool_size_mb": 1024,
            "allocation_strategy": "adaptive"
        }
        
        return request
    
    def _execute_optimized_compilation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the optimized compilation."""
        start_time = time.time()
        
        # Check for cache hit first
        if request.get("cache_hit", False):
            result = request["cached_result"].copy()
            result["cache_hit"] = True
            result["execution_time"] = 0.001  # Minimal time for cache hit
            return result
        
        # Simulate compilation execution
        base_execution_time = 2.0  # Base 2 seconds
        
        # Apply parallelization speedup
        parallelization = request.get("parallelization", {})
        if parallelization.get("enabled", False):
            speedup = parallelization.get("estimated_speedup", 1.0)
            base_execution_time /= speedup
        
        # Apply load balancing efficiency
        if request.get("load_balanced", False):
            base_execution_time *= 0.8  # 20% improvement from load balancing
        
        # Apply memory pooling efficiency
        if request.get("memory_pool", {}).get("enabled", False):
            base_execution_time *= 0.9  # 10% improvement from memory pooling
        
        # Simulate execution
        time.sleep(min(base_execution_time, 3.0))  # Cap at 3 seconds for simulation
        
        execution_time = time.time() - start_time
        
        # Generate result
        result = {
            "success": True,
            "execution_time": execution_time,
            "optimizations_applied": request.get("optimizations_applied", []),
            "performance_metrics": {
                "throughput": 1000 / execution_time,
                "latency": execution_time * 1000,  # Convert to ms
                "efficiency_score": min(1.0, 2.0 / execution_time)
            },
            "worker_assigned": request.get("assigned_worker"),
            "cache_key": self._generate_request_cache_key(request)
        }
        
        # Cache the result
        cache_key = result["cache_key"]
        self.cache_manager.put(cache_key, result)
        
        # Update worker metrics if applicable
        if result["worker_assigned"]:
            self.load_balancer.update_worker_metrics(result["worker_assigned"], {
                "response_time": execution_time * 1000,
                "request_completed": True,
                "error_occurred": False
            })
        
        return result
    
    def _generate_request_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key for compilation request."""
        # Extract relevant parameters for caching
        cache_data = {
            "model_type": request.get("model_type", "unknown"),
            "optimization_level": request.get("optimization_level", 0),
            "target": request.get("target", "simulation"),
            "input_shape": str(request.get("input_shape", [])),
            "model_size": request.get("model_size", 0)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()[:16]
    
    def _record_performance_metrics(self, execution_time: float, optimization_result: Dict[str, Any]) -> None:
        """Record performance metrics for monitoring."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Calculate performance metrics
            throughput = 1.0 / execution_time if execution_time > 0 else 0
            latency = execution_time * 1000  # Convert to ms
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_utilization=cpu_percent,
                memory_utilization=memory.percent,
                throughput=throughput,
                latency=latency,
                queue_length=0,  # Would be actual queue length in real implementation
                error_rate=0.0,  # Would be calculated from recent errors
                cache_hit_rate=self.cache_manager.get_hit_rate(),
                active_workers=len([w for w in self.load_balancer.workers.values() if w["healthy"]])
            )
            
            self.performance_history.append(metrics)
            
            # Update auto-scaler
            self.auto_scaler.update_metrics(metrics)
            
        except Exception as e:
            print(f"Error recording performance metrics: {e}")
    
    def start_performance_monitoring(self) -> None:
        """Start comprehensive performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            
            # Start auto-scaling monitoring
            self.auto_scaler.start_monitoring()
            
            # Start distributed compilation workers
            self.distributed_compiler.start_workers()
            
            print("📊 Hyperscale performance monitoring started")
    
    def stop_performance_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_event.set()
            
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            
            # Stop auto-scaling
            self.auto_scaler.stop_monitoring()
            
            # Stop distributed compilation workers
            self.distributed_compiler.stop_workers()
            
            print("📊 Hyperscale performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main performance monitoring loop."""
        while self.monitoring_active and not self._stop_event.is_set():
            try:
                # Collect and record current performance metrics
                self._collect_system_metrics()
                
                # Optimize system based on recent performance
                self._adaptive_system_optimization()
                
            except Exception as e:
                print(f"Error in performance monitoring loop: {e}")
            
            self._stop_event.wait(15)  # Monitor every 15 seconds
    
    def _collect_system_metrics(self) -> None:
        """Collect current system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Calculate derived metrics
            recent_metrics = list(self.performance_history)[-10:]  # Last 10 measurements
            avg_throughput = statistics.mean([m.throughput for m in recent_metrics]) if recent_metrics else 0
            avg_latency = statistics.mean([m.latency for m in recent_metrics]) if recent_metrics else 0
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_utilization=cpu_percent,
                memory_utilization=memory.percent,
                throughput=avg_throughput,
                latency=avg_latency,
                queue_length=0,
                error_rate=0.0,
                cache_hit_rate=self.cache_manager.get_hit_rate(),
                active_workers=len([w for w in self.load_balancer.workers.values() if w["healthy"]])
            )
            
            self.performance_history.append(metrics)
            self.auto_scaler.update_metrics(metrics)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _adaptive_system_optimization(self) -> None:
        """Perform adaptive system optimization based on performance trends."""
        if len(self.performance_history) < 10:
            return
        
        recent_metrics = list(self.performance_history)[-10:]
        
        # Analyze performance trends
        latencies = [m.latency for m in recent_metrics]
        hit_rates = [m.cache_hit_rate for m in recent_metrics]
        
        # Optimize cache if hit rate is declining
        if len(hit_rates) > 5:
            recent_hit_rate = statistics.mean(hit_rates[-5:])
            if recent_hit_rate < 0.7:  # Below 70% hit rate
                # Increase cache size
                current_size = self.cache_manager.max_size
                new_size = min(5000, int(current_size * 1.2))
                self.cache_manager.max_size = new_size
        
        # Optimize load balancing strategy based on latency
        if len(latencies) > 5:
            recent_latency = statistics.mean(latencies[-5:])
            if recent_latency > 150:  # Above 150ms
                # Switch to performance-based routing
                self.load_balancer.routing_strategy = "performance_based"
            elif recent_latency < 50:  # Below 50ms
                # Can use simpler strategy
                self.load_balancer.routing_strategy = "least_connections"
    
    def get_hyperscale_summary(self) -> Dict[str, Any]:
        """Get comprehensive hyperscale performance summary."""
        return {
            "performance_engine_status": "active" if self.monitoring_active else "inactive",
            "cache_stats": self.cache_manager.get_stats(),
            "load_balancer_stats": self.load_balancer.get_worker_stats(),
            "auto_scaling_stats": self.auto_scaler.get_scaling_stats(),
            "distributed_compilation_stats": self.distributed_compiler.get_compilation_stats(),
            "recent_performance": {
                "avg_latency": statistics.mean([m.latency for m in list(self.performance_history)[-10:]]) if self.performance_history else 0,
                "avg_throughput": statistics.mean([m.throughput for m in list(self.performance_history)[-10:]]) if self.performance_history else 0,
                "current_cache_hit_rate": self.cache_manager.get_hit_rate(),
                "total_performance_samples": len(self.performance_history)
            },
            "optimization_strategies_active": list(self.optimization_strategies)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize hyperscale performance engine
    engine = HyperscalePerformanceEngine()
    
    print("⚡ HYPERSCALE PERFORMANCE ENGINE TESTING")
    
    # Start monitoring
    engine.start_performance_monitoring()
    
    # Test compilation optimization
    test_requests = [
        {
            "model_type": "spikeformer",
            "model_size": 2000000,
            "optimization_level": 3,
            "target": "loihi3",
            "input_shape": [1, 3, 224, 224]
        },
        {
            "model_type": "dsformer",
            "model_size": 500000,
            "optimization_level": 2,
            "target": "simulation",
            "input_shape": [1, 1, 128, 128]
        }
    ]
    
    for i, request in enumerate(test_requests):
        print(f"\n🚀 Optimizing compilation request {i+1}:")
        
        result = engine.optimize_compilation(
            request,
            optimization_strategies=[
                OptimizationStrategy.CACHING,
                OptimizationStrategy.LOAD_BALANCING,
                OptimizationStrategy.PARALLELIZATION,
                OptimizationStrategy.MEMORY_POOLING
            ]
        )
        
        print(f"   Optimizations applied: {result['optimizations_applied']}")
        if result["final_result"]["success"]:
            metrics = result["final_result"]["performance_metrics"]
            print(f"   Performance: {metrics['throughput']:.1f} ops/sec, {metrics['latency']:.1f}ms latency")
            print(f"   Cache hit: {result['final_result'].get('cache_hit', False)}")
            print(f"   Worker: {result['final_result'].get('worker_assigned', 'N/A')}")
    
    # Wait for some monitoring data
    time.sleep(5)
    
    # Get comprehensive summary
    summary = engine.get_hyperscale_summary()
    
    print(f"\n📊 HYPERSCALE PERFORMANCE SUMMARY:")
    print(f"   Cache Hit Rate: {summary['cache_stats']['hit_rate']:.3f}")
    print(f"   Active Workers: {summary['load_balancer_stats']['healthy_workers']}")
    print(f"   Current Instances: {summary['auto_scaling_stats']['current_instances']}")
    print(f"   Average Latency: {summary['recent_performance']['avg_latency']:.1f}ms")
    print(f"   Average Throughput: {summary['recent_performance']['avg_throughput']:.1f} ops/sec")
    
    # Stop monitoring
    engine.stop_performance_monitoring()
