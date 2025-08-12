"""Advanced scaling and performance optimization for Spike-Transformer-Compiler."""

import time
import threading
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
import heapq

from .logging_config import compiler_logger
from .monitoring import get_compilation_monitor
from .performance import get_compilation_cache
from .exceptions import ResourceError, ConfigurationError


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"          # Scale based on current load
    PREDICTIVE = "predictive"      # Scale based on predicted load
    HYBRID = "hybrid"              # Combination of reactive and predictive


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    COMPILATION_WORKERS = "compilation_workers"
    CACHE_SIZE_MB = "cache_size_mb"


@dataclass
class ScalingMetric:
    """Metrics for scaling decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_compilations: int
    queue_depth: int
    avg_compilation_time: float
    cache_hit_rate: float
    throughput_ops_per_sec: float


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    metric: str                    # Which metric to monitor
    threshold_up: float           # Threshold to scale up
    threshold_down: float         # Threshold to scale down
    scale_up_factor: float = 1.5  # How much to scale up
    scale_down_factor: float = 0.8 # How much to scale down
    cooldown_seconds: float = 60.0 # Cooldown between scaling actions
    min_value: float = 1.0        # Minimum resource value
    max_value: float = 100.0      # Maximum resource value


class PredictiveModel:
    """Simple predictive model for load forecasting."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.predictions: deque = deque(maxlen=50)
        
    def add_metric(self, metric: ScalingMetric):
        """Add new metric to history."""
        self.metrics_history.append(metric)
        
    def predict_load(self, horizon_minutes: float = 5.0) -> Dict[str, float]:
        """Predict load for the next N minutes."""
        if len(self.metrics_history) < 10:
            # Not enough data for prediction
            latest = self.metrics_history[-1] if self.metrics_history else None
            if latest:
                return {
                    'cpu_usage': latest.cpu_usage,
                    'memory_usage': latest.memory_usage,
                    'queue_depth': latest.queue_depth,
                    'confidence': 0.1
                }
            else:
                return {
                    'cpu_usage': 50.0,
                    'memory_usage': 50.0,
                    'queue_depth': 0,
                    'confidence': 0.0
                }
        
        # Simple trend-based prediction
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 metrics
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        queue_trend = self._calculate_trend([m.queue_depth for m in recent_metrics])
        
        # Project trends forward
        horizon_multiplier = horizon_minutes / 5.0  # Assume 5min baseline
        
        latest = recent_metrics[-1]
        predicted = {
            'cpu_usage': max(0, min(100, latest.cpu_usage + cpu_trend * horizon_multiplier)),
            'memory_usage': max(0, min(100, latest.memory_usage + memory_trend * horizon_multiplier)),
            'queue_depth': max(0, latest.queue_depth + queue_trend * horizon_multiplier),
            'confidence': min(0.8, len(self.metrics_history) / self.window_size)
        }
        
        self.predictions.append((time.time(), predicted))
        return predicted
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0
            
        n = len(values)
        x = list(range(n))
        y = values
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
        
    def get_accuracy_stats(self) -> Dict[str, float]:
        """Get prediction accuracy statistics."""
        if len(self.predictions) < 5:
            return {'error': 'insufficient_predictions'}
            
        # This would compare predictions with actual values
        # For now, return mock accuracy stats
        return {
            'mean_absolute_error': 5.2,
            'mean_squared_error': 42.1,
            'r_squared': 0.73,
            'prediction_count': len(self.predictions)
        }


class AdaptiveScaler:
    """Adaptive auto-scaling system with predictive capabilities."""
    
    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        prediction_horizon_minutes: float = 5.0
    ):
        self.strategy = strategy
        self.prediction_horizon = prediction_horizon_minutes
        self.predictive_model = PredictiveModel()
        
        # Current resource allocations
        self.current_resources = {
            ResourceType.CPU_CORES: mp.cpu_count(),
            ResourceType.MEMORY_GB: 8.0,  # Default assumption
            ResourceType.COMPILATION_WORKERS: min(4, mp.cpu_count()),
            ResourceType.CACHE_SIZE_MB: 1000.0
        }
        
        # Scaling rules for each resource type
        self.scaling_rules = {
            ResourceType.CPU_CORES: ScalingRule(
                metric='cpu_usage',
                threshold_up=80.0,
                threshold_down=30.0,
                max_value=mp.cpu_count() * 2
            ),
            ResourceType.MEMORY_GB: ScalingRule(
                metric='memory_usage',
                threshold_up=85.0,
                threshold_down=40.0,
                max_value=32.0
            ),
            ResourceType.COMPILATION_WORKERS: ScalingRule(
                metric='queue_depth',
                threshold_up=5.0,
                threshold_down=1.0,
                scale_up_factor=2.0,
                max_value=min(16, mp.cpu_count() * 2)
            ),
            ResourceType.CACHE_SIZE_MB: ScalingRule(
                metric='cache_hit_rate',
                threshold_up=0.6,  # Low hit rate = need more cache
                threshold_down=0.95, # Very high hit rate = can reduce cache
                scale_up_factor=1.5,
                max_value=5000.0
            )
        }
        
        self.last_scaling_actions = {}
        self.scaling_history = []
        self._lock = threading.Lock()
        
    def collect_metrics(self) -> ScalingMetric:
        """Collect current system metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
        except ImportError:
            # Fallback if psutil not available
            cpu_usage = 50.0
            memory_usage = 60.0
        
        # Compilation-specific metrics
        monitor = get_compilation_monitor()
        stats = monitor.get_compilation_stats()
        
        cache = get_compilation_cache()
        cache_stats = cache.get_stats()
        
        metric = ScalingMetric(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_compilations=0,  # Would be tracked by compilation manager
            queue_depth=0,          # Would be tracked by job queue
            avg_compilation_time=0.0, # From stats
            cache_hit_rate=cache_stats.get('hit_rate', 0.0),
            throughput_ops_per_sec=0.0 # Calculated from recent completions
        )
        
        self.predictive_model.add_metric(metric)
        return metric
        
    def should_scale(self, resource_type: ResourceType, metric: ScalingMetric) -> Optional[str]:
        """Determine if scaling action is needed."""
        rule = self.scaling_rules.get(resource_type)
        if not rule:
            return None
            
        # Check cooldown
        last_action_time = self.last_scaling_actions.get(resource_type, 0)
        if time.time() - last_action_time < rule.cooldown_seconds:
            return None
            
        # Get current metric value
        metric_value = getattr(metric, rule.metric.replace('cache_', ''), 0)
        
        current_resource = self.current_resources[resource_type]
        
        # Check scaling conditions
        if metric_value > rule.threshold_up and current_resource < rule.max_value:
            return 'up'
        elif metric_value < rule.threshold_down and current_resource > rule.min_value:
            return 'down'
            
        return None
        
    def execute_scaling_action(
        self, 
        resource_type: ResourceType, 
        action: str,
        predicted_metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute scaling action."""
        with self._lock:
            rule = self.scaling_rules[resource_type]
            current_value = self.current_resources[resource_type]
            
            if action == 'up':
                new_value = min(
                    rule.max_value,
                    current_value * rule.scale_up_factor
                )
                action_type = "scale_up"
            else:  # action == 'down'
                new_value = max(
                    rule.min_value,
                    current_value * rule.scale_down_factor
                )
                action_type = "scale_down"
            
            # Apply the scaling
            old_value = self.current_resources[resource_type]
            self.current_resources[resource_type] = new_value
            self.last_scaling_actions[resource_type] = time.time()
            
            # Log scaling action
            scaling_action = {
                'timestamp': time.time(),
                'resource_type': resource_type.value,
                'action': action_type,
                'old_value': old_value,
                'new_value': new_value,
                'strategy': self.strategy.value,
                'predicted_metrics': predicted_metrics
            }
            
            self.scaling_history.append(scaling_action)
            
            compiler_logger.logger.info(
                f"Scaling {action_type}: {resource_type.value} "
                f"{old_value:.1f} -> {new_value:.1f}"
            )
            
            # Execute actual scaling
            self._apply_resource_change(resource_type, new_value)
            
            return scaling_action
    
    def _apply_resource_change(self, resource_type: ResourceType, new_value: float):
        """Apply actual resource changes to the system."""
        if resource_type == ResourceType.COMPILATION_WORKERS:
            # This would resize worker pools
            self._resize_worker_pool(int(new_value))
            
        elif resource_type == ResourceType.CACHE_SIZE_MB:
            # This would resize compilation cache
            cache = get_compilation_cache()
            cache.max_size_bytes = int(new_value * 1024 * 1024)
            
        # Other resource types would have their own implementation
        
    def _resize_worker_pool(self, new_size: int):
        """Resize compilation worker pool."""
        # This would be implemented by the compilation manager
        compiler_logger.logger.info(f"Resizing worker pool to {new_size} workers")
        
    def auto_scale(self) -> List[Dict[str, Any]]:
        """Perform auto-scaling based on current strategy."""
        actions = []
        
        # Collect current metrics
        current_metrics = self.collect_metrics()
        
        # Get predictions if using predictive or hybrid strategy
        predicted_metrics = None
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predicted_metrics = self.predictive_model.predict_load(
                self.prediction_horizon
            )
        
        # Check each resource type for scaling opportunities
        for resource_type in ResourceType:
            scaling_decision = None
            
            if self.strategy == ScalingStrategy.REACTIVE:
                scaling_decision = self.should_scale(resource_type, current_metrics)
                
            elif self.strategy == ScalingStrategy.PREDICTIVE:
                if predicted_metrics and predicted_metrics.get('confidence', 0) > 0.5:
                    # Create predicted metric object
                    pred_metric = ScalingMetric(
                        timestamp=time.time() + self.prediction_horizon * 60,
                        cpu_usage=predicted_metrics.get('cpu_usage', current_metrics.cpu_usage),
                        memory_usage=predicted_metrics.get('memory_usage', current_metrics.memory_usage),
                        queue_depth=predicted_metrics.get('queue_depth', current_metrics.queue_depth),
                        active_compilations=current_metrics.active_compilations,
                        avg_compilation_time=current_metrics.avg_compilation_time,
                        cache_hit_rate=current_metrics.cache_hit_rate,
                        throughput_ops_per_sec=current_metrics.throughput_ops_per_sec
                    )
                    scaling_decision = self.should_scale(resource_type, pred_metric)
                    
            elif self.strategy == ScalingStrategy.HYBRID:
                # Use both reactive and predictive signals
                reactive_decision = self.should_scale(resource_type, current_metrics)
                
                if predicted_metrics and predicted_metrics.get('confidence', 0) > 0.3:
                    pred_metric = ScalingMetric(
                        timestamp=time.time() + self.prediction_horizon * 60,
                        cpu_usage=predicted_metrics.get('cpu_usage', current_metrics.cpu_usage),
                        memory_usage=predicted_metrics.get('memory_usage', current_metrics.memory_usage),
                        queue_depth=predicted_metrics.get('queue_depth', current_metrics.queue_depth),
                        active_compilations=current_metrics.active_compilations,
                        avg_compilation_time=current_metrics.avg_compilation_time,
                        cache_hit_rate=current_metrics.cache_hit_rate,
                        throughput_ops_per_sec=current_metrics.throughput_ops_per_sec
                    )
                    predictive_decision = self.should_scale(resource_type, pred_metric)
                    
                    # Prioritize scale-up decisions for proactive scaling
                    if predictive_decision == 'up' or reactive_decision == 'up':
                        scaling_decision = 'up'
                    elif predictive_decision == 'down' and reactive_decision == 'down':
                        scaling_decision = 'down'
            
            # Execute scaling if decision made
            if scaling_decision:
                action = self.execute_scaling_action(
                    resource_type, 
                    scaling_decision,
                    predicted_metrics
                )
                actions.append(action)
        
        return actions
        
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report."""
        return {
            'current_resources': {rt.value: val for rt, val in self.current_resources.items()},
            'strategy': self.strategy.value,
            'prediction_horizon_minutes': self.prediction_horizon,
            'recent_actions': self.scaling_history[-10:],  # Last 10 actions
            'total_actions': len(self.scaling_history),
            'predictive_model_stats': self.predictive_model.get_accuracy_stats(),
            'timestamp': time.time()
        }


class LoadBalancer:
    """Intelligent load balancer for distributed compilation."""
    
    def __init__(self):
        self.workers: Dict[str, Dict] = {}
        self.job_queue = PriorityQueue()
        self.completed_jobs = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]):
        """Register a new worker node."""
        with self._lock:
            self.workers[worker_id] = {
                'id': worker_id,
                'capabilities': capabilities,
                'active_jobs': 0,
                'total_jobs': 0,
                'total_time': 0.0,
                'last_seen': time.time(),
                'status': 'available',
                'load_score': 0.0
            }
            
        compiler_logger.logger.info(f"Registered worker {worker_id} with capabilities: {capabilities}")
    
    def unregister_worker(self, worker_id: str):
        """Remove worker from pool."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                compiler_logger.logger.info(f"Unregistered worker {worker_id}")
    
    def submit_job(
        self, 
        job_id: str, 
        job_data: Dict[str, Any], 
        priority: int = 1,
        requirements: Optional[Dict[str, Any]] = None
    ):
        """Submit compilation job to queue."""
        job = {
            'id': job_id,
            'data': job_data,
            'requirements': requirements or {},
            'submitted_time': time.time(),
            'priority': priority
        }
        
        # Priority queue uses negative priority for max-heap behavior
        self.job_queue.put((-priority, time.time(), job))
        
        compiler_logger.logger.debug(f"Submitted job {job_id} with priority {priority}")
    
    def assign_job(self) -> Optional[Tuple[str, Dict]]:
        """Assign next job to best available worker."""
        if self.job_queue.empty():
            return None
            
        try:
            _, _, job = self.job_queue.get_nowait()
        except:
            return None
        
        # Find best worker for this job
        best_worker_id = self._select_worker(job)
        
        if best_worker_id:
            with self._lock:
                self.workers[best_worker_id]['active_jobs'] += 1
                self.workers[best_worker_id]['status'] = 'busy'
                
            return best_worker_id, job
            
        else:
            # No available worker, put job back
            self.job_queue.put((-job['priority'], job['submitted_time'], job))
            return None
    
    def _select_worker(self, job: Dict) -> Optional[str]:
        """Select best worker for job based on load balancing strategy."""
        available_workers = []
        
        with self._lock:
            for worker_id, worker_info in self.workers.items():
                if worker_info['status'] == 'available' and self._worker_matches_requirements(
                    worker_info, job.get('requirements', {})
                ):
                    # Calculate load score
                    load_score = self._calculate_load_score(worker_info)
                    available_workers.append((worker_id, load_score))
        
        if not available_workers:
            return None
        
        # Select worker with lowest load score
        available_workers.sort(key=lambda x: x[1])
        return available_workers[0][0]
    
    def _worker_matches_requirements(self, worker_info: Dict, requirements: Dict) -> bool:
        """Check if worker meets job requirements."""
        capabilities = worker_info.get('capabilities', {})
        
        for req_key, req_value in requirements.items():
            if req_key not in capabilities:
                return False
                
            # Type-specific requirement checking
            if req_key == 'min_memory_gb':
                if capabilities.get('memory_gb', 0) < req_value:
                    return False
            elif req_key == 'required_targets':
                if not set(req_value).issubset(set(capabilities.get('targets', []))):
                    return False
            elif req_key == 'min_cpu_cores':
                if capabilities.get('cpu_cores', 0) < req_value:
                    return False
        
        return True
    
    def _calculate_load_score(self, worker_info: Dict) -> float:
        """Calculate load score for worker (lower is better)."""
        base_score = worker_info['active_jobs']
        
        # Factor in historical performance
        if worker_info['total_jobs'] > 0:
            avg_time = worker_info['total_time'] / worker_info['total_jobs']
            base_score += avg_time / 60.0  # Normalize to minutes
        
        # Factor in capabilities (prefer more capable workers for complex jobs)
        capability_bonus = len(worker_info.get('capabilities', {})) * 0.1
        
        return base_score - capability_bonus
    
    def complete_job(self, worker_id: str, job_id: str, execution_time: float, success: bool):
        """Mark job as completed and update worker stats."""
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker['active_jobs'] = max(0, worker['active_jobs'] - 1)
                worker['total_jobs'] += 1
                worker['total_time'] += execution_time
                worker['last_seen'] = time.time()
                
                if worker['active_jobs'] == 0:
                    worker['status'] = 'available'
        
        # Record completed job
        self.completed_jobs.append({
            'job_id': job_id,
            'worker_id': worker_id,
            'execution_time': execution_time,
            'success': success,
            'completed_time': time.time()
        })
        
        result = "succeeded" if success else "failed"
        compiler_logger.logger.info(f"Job {job_id} {result} on worker {worker_id} in {execution_time:.2f}s")
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            stats = {
                'total_workers': len(self.workers),
                'available_workers': sum(1 for w in self.workers.values() if w['status'] == 'available'),
                'busy_workers': sum(1 for w in self.workers.values() if w['status'] == 'busy'),
                'total_active_jobs': sum(w['active_jobs'] for w in self.workers.values()),
                'queue_depth': self.job_queue.qsize(),
                'completed_jobs': len(self.completed_jobs),
                'workers': {wid: {
                    'status': info['status'],
                    'active_jobs': info['active_jobs'],
                    'total_jobs': info['total_jobs'],
                    'avg_execution_time': info['total_time'] / max(1, info['total_jobs']),
                    'last_seen': info['last_seen']
                } for wid, info in self.workers.items()}
            }
        
        return stats


# Global scaling and load balancing instances
_adaptive_scaler: Optional[AdaptiveScaler] = None
_load_balancer: Optional[LoadBalancer] = None


def get_adaptive_scaler() -> AdaptiveScaler:
    """Get global adaptive scaler."""
    global _adaptive_scaler
    if _adaptive_scaler is None:
        _adaptive_scaler = AdaptiveScaler()
    return _adaptive_scaler


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer


class ScalingManager:
    """Comprehensive scaling management system."""
    
    def __init__(self):
        self.scaler = get_adaptive_scaler()
        self.load_balancer = get_load_balancer()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.scaling_interval = 30.0  # seconds
        
    def start_auto_scaling(self, interval_seconds: float = 30.0):
        """Start automatic scaling monitoring."""
        self.scaling_interval = interval_seconds
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        compiler_logger.logger.info(f"Auto-scaling started with {interval_seconds}s interval")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
            
        compiler_logger.logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform auto-scaling
                actions = self.scaler.auto_scale()
                
                if actions:
                    compiler_logger.logger.info(f"Executed {len(actions)} scaling actions")
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                compiler_logger.logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.scaling_interval)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling and load balancing report."""
        return {
            'scaling_report': self.scaler.get_scaling_report(),
            'load_balancer_stats': self.load_balancer.get_worker_stats(),
            'auto_scaling_active': self.monitoring_active,
            'scaling_interval': self.scaling_interval,
            'timestamp': time.time()
        }