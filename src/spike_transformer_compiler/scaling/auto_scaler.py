"""Advanced auto-scaling system for neuromorphic compilation workloads."""

import time
import threading
import asyncio
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import weakref

from ..logging_config import compiler_logger
from ..performance import PerformanceProfiler, ResourceMonitor
from ..exceptions import ResourceError
from .resource_pool import AdvancedResourcePool, ResourceRequest


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    active_tasks: int
    completed_tasks_per_minute: float
    average_task_duration: float
    resource_pool_utilization: Dict[str, float] = field(default_factory=dict)
    system_load: float = 0.0
    prediction_confidence: float = 0.0


@dataclass 
class ScalingDecision:
    """Represents a scaling decision."""
    action: str  # 'scale_up', 'scale_down', 'no_action'
    pool_name: str
    target_size: int
    current_size: int
    confidence: float
    reason: str
    metrics_snapshot: ScalingMetrics


class PredictiveScaler:
    """Predictive auto-scaler using machine learning techniques."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.prediction_model = None
        self.feature_weights = {
            'cpu_utilization': 0.3,
            'memory_utilization': 0.2,
            'queue_length': 0.3,
            'active_tasks': 0.2
        }
        
        # Enhanced ML-driven features
        self.workload_patterns = {}
        self.seasonality_detector = WorkloadSeasonalityDetector()
        self.anomaly_detector = SimpleAnomalyDetector()
        self.multi_objective_optimizer = MultiObjectiveResourceOptimizer()
        
    def add_metrics(self, metrics: ScalingMetrics) -> None:
        """Add metrics to history for learning."""
        self.metrics_history.append(metrics)
        
        # Update model if we have enough data
        if len(self.metrics_history) >= 100:
            self._update_prediction_model()
    
    def _update_prediction_model(self) -> None:
        """Update the prediction model with recent data."""
        try:
            # Simple moving average predictor
            if len(self.metrics_history) < 10:
                return
            
            recent_metrics = list(self.metrics_history)[-50:]  # Last 50 data points
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_utilization for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_utilization for m in recent_metrics])
            queue_trend = self._calculate_trend([m.queue_length for m in recent_metrics])
            
            self.prediction_model = {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend, 
                'queue_trend': queue_trend,
                'updated_at': time.time()
            }
            
        except Exception as e:
            compiler_logger.logger.warning(f"Failed to update prediction model: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    
    def predict_load(self, horizon_minutes: float = 5.0) -> Dict[str, float]:
        """Predict future load based on historical data."""
        if not self.prediction_model or len(self.metrics_history) < 10:
            # Fallback to current values
            latest = self.metrics_history[-1] if self.metrics_history else None
            if not latest:
                return {'cpu_utilization': 50.0, 'memory_utilization': 50.0, 'queue_length': 0.0}
            
            return {
                'cpu_utilization': latest.cpu_utilization,
                'memory_utilization': latest.memory_utilization, 
                'queue_length': float(latest.queue_length)
            }
        
        # Predict based on trends
        latest = self.metrics_history[-1]
        prediction = {
            'cpu_utilization': max(0, min(100, 
                latest.cpu_utilization + self.prediction_model['cpu_trend'] * horizon_minutes)),
            'memory_utilization': max(0, min(100,
                latest.memory_utilization + self.prediction_model['memory_trend'] * horizon_minutes)),
            'queue_length': max(0,
                latest.queue_length + self.prediction_model['queue_trend'] * horizon_minutes)
        }
        
        return prediction
    
    def get_confidence(self) -> float:
        """Get confidence in current predictions."""
        if not self.prediction_model or len(self.metrics_history) < 20:
            return 0.3  # Low confidence
        
        # Calculate prediction accuracy from recent history
        recent_predictions = []
        actual_values = []
        
        for i in range(-10, -1):  # Last 10 predictions
            if i + len(self.metrics_history) < 0:
                continue
            
            # Simulate prediction from history
            past_metrics = list(self.metrics_history)[:i]
            if len(past_metrics) < 10:
                continue
            
            # This is a simplified confidence calculation
            recent_predictions.append(past_metrics[-1].cpu_utilization)
            actual_values.append(self.metrics_history[i].cpu_utilization)
        
        if not recent_predictions:
            return 0.3
        
        # Calculate mean absolute error
        errors = [abs(p - a) for p, a in zip(recent_predictions, actual_values)]
        mae = np.mean(errors)
        
        # Convert to confidence (lower error = higher confidence)
        confidence = max(0.1, min(1.0, 1.0 - mae / 100.0))
        return confidence


class WorkloadSeasonalityDetector:
    """Detects seasonal patterns in workload for predictive scaling."""
    
    def __init__(self):
        self.hourly_patterns = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        self.weekly_patterns = defaultdict(list)
        
    def add_observation(self, timestamp: float, workload_intensity: float) -> None:
        """Add workload observation."""
        dt = time.localtime(timestamp)
        
        self.hourly_patterns[dt.tm_hour].append(workload_intensity)
        self.daily_patterns[dt.tm_mday].append(workload_intensity)
        self.weekly_patterns[dt.tm_wday].append(workload_intensity)
        
    def predict_workload(self, future_timestamp: float) -> float:
        """Predict workload intensity based on seasonal patterns."""
        dt = time.localtime(future_timestamp)
        
        # Get historical patterns
        hourly_avg = np.mean(self.hourly_patterns[dt.tm_hour]) if self.hourly_patterns[dt.tm_hour] else 50.0
        daily_avg = np.mean(self.daily_patterns[dt.tm_mday]) if self.daily_patterns[dt.tm_mday] else 50.0
        weekly_avg = np.mean(self.weekly_patterns[dt.tm_wday]) if self.weekly_patterns[dt.tm_wday] else 50.0
        
        # Weighted combination
        return (hourly_avg * 0.5 + daily_avg * 0.3 + weekly_avg * 0.2)


class SimpleAnomalyDetector:
    """Simple statistical anomaly detector for resource usage."""
    
    def __init__(self, window_size: int = 50, threshold_sigma: float = 2.5):
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self.recent_values = deque(maxlen=window_size)
        
    def add_value(self, value: float) -> None:
        """Add new value to the detector."""
        self.recent_values.append(value)
        
    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        """Check if value is anomalous. Returns (is_anomaly, severity)."""
        if len(self.recent_values) < 10:
            return False, 0.0
            
        mean_val = np.mean(self.recent_values)
        std_val = np.std(self.recent_values)
        
        if std_val == 0:
            return False, 0.0
            
        z_score = abs(value - mean_val) / std_val
        is_anomaly = z_score > self.threshold_sigma
        severity = min(1.0, z_score / (self.threshold_sigma * 2))
        
        return is_anomaly, severity


class MultiObjectiveResourceOptimizer:
    """Multi-objective optimizer balancing performance, cost, and energy."""
    
    def __init__(self):
        self.objectives = {
            'performance': {'weight': 0.4, 'target': 'maximize'},
            'cost_efficiency': {'weight': 0.4, 'target': 'maximize'},
            'energy_efficiency': {'weight': 0.2, 'target': 'maximize'}
        }
        
    def optimize_resource_allocation(
        self, 
        current_allocation: Dict[str, int],
        workload_demand: float,
        constraints: Dict[str, Any]
    ) -> Dict[str, int]:
        """Optimize resource allocation across multiple objectives."""
        
        # Generate candidate allocations
        candidates = self._generate_candidates(current_allocation, workload_demand, constraints)
        
        # Evaluate each candidate
        best_candidate = current_allocation
        best_score = -float('inf')
        
        for candidate in candidates:
            score = self._evaluate_candidate(candidate, workload_demand)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                
        return best_candidate
    
    def _generate_candidates(
        self, 
        current: Dict[str, int], 
        demand: float, 
        constraints: Dict[str, Any]
    ) -> List[Dict[str, int]]:
        """Generate candidate resource allocations."""
        candidates = [current.copy()]
        
        # Generate variations
        for pool_name, current_size in current.items():
            max_size = constraints.get(f'{pool_name}_max', current_size * 2)
            min_size = constraints.get(f'{pool_name}_min', max(1, current_size // 2))
            
            # Conservative increase
            if current_size < max_size:
                candidate = current.copy()
                candidate[pool_name] = min(max_size, int(current_size * 1.2))
                candidates.append(candidate)
                
            # Conservative decrease
            if current_size > min_size:
                candidate = current.copy()
                candidate[pool_name] = max(min_size, int(current_size * 0.8))
                candidates.append(candidate)
                
        return candidates[:10]  # Limit candidates
    
    def _evaluate_candidate(self, candidate: Dict[str, int], demand: float) -> float:
        """Evaluate candidate allocation using multi-objective scoring."""
        total_resources = sum(candidate.values())
        
        # Performance score (ability to handle demand)
        performance_score = min(1.0, total_resources / max(1.0, demand))
        
        # Cost efficiency (resources per unit capacity)
        cost_efficiency = 1.0 / max(1.0, total_resources) if total_resources > 0 else 0.0
        
        # Energy efficiency (simplified model)
        energy_efficiency = 1.0 - (total_resources * 0.01)  # Penalty for more resources
        energy_efficiency = max(0.0, energy_efficiency)
        
        # Weighted combination
        score = (
            performance_score * self.objectives['performance']['weight'] +
            cost_efficiency * self.objectives['cost_efficiency']['weight'] +
            energy_efficiency * self.objectives['energy_efficiency']['weight']
        )
        
        return score


class AdvancedAutoScaler:
    """Advanced auto-scaler with predictive scaling and load balancing."""
    
    def __init__(self, resource_pool: AdvancedResourcePool):
        self.resource_pool = resource_pool
        self.predictive_scaler = PredictiveScaler()
        self.resource_monitor = ResourceMonitor()
        
        # Configuration
        self.scaling_config = {
            'scale_up_cpu_threshold': 75.0,
            'scale_down_cpu_threshold': 30.0,
            'scale_up_memory_threshold': 80.0,
            'scale_down_memory_threshold': 40.0,
            'scale_up_queue_threshold': 5,
            'scale_down_queue_threshold': 1,
            'min_scaling_interval': 60.0,  # 1 minute
            'prediction_horizon': 5.0,  # 5 minutes
            'confidence_threshold': 0.7
        }
        
        # State tracking
        self.scaling_history: deque = deque(maxlen=100)
        self.last_scaling_decisions = {}
        self.scaling_lock = threading.Lock()
        
        # Background services
        self.monitoring_active = False
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.task_queue_lengths = deque(maxlen=60)  # 1 hour at 1-minute intervals
        self.task_completion_times = deque(maxlen=1000)
        
        compiler_logger.logger.info("Advanced auto-scaler initialized")
    
    def start_monitoring(self) -> None:
        """Start background monitoring and auto-scaling."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.resource_monitor.start_monitoring()
        
        def monitoring_loop():
            while not self.shutdown_event.wait(30):  # Check every 30 seconds
                try:
                    self._collect_metrics_and_scale()
                except Exception as e:
                    compiler_logger.logger.error(f"Auto-scaling monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        compiler_logger.logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.resource_monitor.stop_monitoring()
        compiler_logger.logger.info("Auto-scaler monitoring stopped")
    
    def _collect_metrics_and_scale(self) -> None:
        """Collect current metrics and make scaling decisions."""
        try:
            # Collect system metrics
            current_usage = self.resource_monitor.get_current_usage()
            pool_stats = self.resource_pool.get_pool_stats()
            
            # Create metrics snapshot
            metrics = ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=current_usage.get('process_cpu_percent', 0),
                memory_utilization=current_usage.get('system_memory_percent', 0),
                queue_length=len(getattr(self, 'pending_tasks', [])),
                active_tasks=sum(pool['in_use'] for pool in pool_stats.values()),
                completed_tasks_per_minute=self._calculate_completion_rate(),
                average_task_duration=self._calculate_average_duration(),
                resource_pool_utilization={
                    name: pool['utilization'] for name, pool in pool_stats.items()
                },
                system_load=psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            )
            
            # Add to prediction model
            self.predictive_scaler.add_metrics(metrics)
            
            # Make scaling decisions for each pool
            for pool_name in pool_stats.keys():
                decision = self._make_scaling_decision(pool_name, metrics)
                if decision.action != 'no_action':
                    self._execute_scaling_decision(decision)
            
        except Exception as e:
            compiler_logger.logger.error(f"Error in metrics collection: {e}")
    
    def _calculate_completion_rate(self) -> float:
        """Calculate task completion rate per minute."""
        if len(self.task_completion_times) < 2:
            return 0.0
        
        recent_completions = [
            t for t in self.task_completion_times 
            if time.time() - t < 60.0  # Last minute
        ]
        
        return len(recent_completions)
    
    def _calculate_average_duration(self) -> float:
        """Calculate average task duration."""
        # This would need to be integrated with actual task tracking
        return 30.0  # Placeholder: 30 seconds average
    
    def _make_scaling_decision(
        self, 
        pool_name: str, 
        current_metrics: ScalingMetrics
    ) -> ScalingDecision:
        """Make scaling decision for a specific pool."""
        pool_stats = self.resource_pool.get_pool_stats(pool_name)
        if not pool_stats:
            return ScalingDecision(
                action='no_action',
                pool_name=pool_name,
                target_size=0,
                current_size=0,
                confidence=0.0,
                reason="Pool not found",
                metrics_snapshot=current_metrics
            )
        
        current_size = pool_stats['current_size']
        utilization = pool_stats['utilization']
        
        # Check if enough time has passed since last scaling
        if pool_name in self.last_scaling_decisions:
            time_since_last = time.time() - self.last_scaling_decisions[pool_name]
            if time_since_last < self.scaling_config['min_scaling_interval']:
                return ScalingDecision(
                    action='no_action',
                    pool_name=pool_name,
                    target_size=current_size,
                    current_size=current_size,
                    confidence=1.0,
                    reason=f"Cooling down ({time_since_last:.1f}s < {self.scaling_config['min_scaling_interval']}s)",
                    metrics_snapshot=current_metrics
                )
        
        # Get predictions
        predictions = self.predictive_scaler.predict_load(
            self.scaling_config['prediction_horizon']
        )
        confidence = self.predictive_scaler.get_confidence()
        
        # Determine scaling action
        action = 'no_action'
        target_size = current_size
        reason = "No scaling needed"
        
        # Scale up conditions
        scale_up_indicators = [
            utilization > 0.8,  # High pool utilization
            current_metrics.cpu_utilization > self.scaling_config['scale_up_cpu_threshold'],
            current_metrics.memory_utilization > self.scaling_config['scale_up_memory_threshold'],
            current_metrics.queue_length > self.scaling_config['scale_up_queue_threshold'],
            predictions['cpu_utilization'] > self.scaling_config['scale_up_cpu_threshold']
        ]
        
        # Scale down conditions
        scale_down_indicators = [
            utilization < 0.3,  # Low pool utilization
            current_metrics.cpu_utilization < self.scaling_config['scale_down_cpu_threshold'],
            current_metrics.memory_utilization < self.scaling_config['scale_down_memory_threshold'],
            current_metrics.queue_length < self.scaling_config['scale_down_queue_threshold'],
            predictions['cpu_utilization'] < self.scaling_config['scale_down_cpu_threshold']
        ]
        
        # Decision logic with confidence weighting
        scale_up_score = sum(scale_up_indicators) * confidence
        scale_down_score = sum(scale_down_indicators) * confidence
        
        if scale_up_score >= 2.0 and current_size < pool_stats['max_size']:
            action = 'scale_up'
            target_size = min(pool_stats['max_size'], int(current_size * 1.5))
            reason = f"Scale up: score={scale_up_score:.1f}, utilization={utilization:.1f}"
            
        elif scale_down_score >= 2.0 and current_size > pool_stats['min_size']:
            action = 'scale_down'
            target_size = max(pool_stats['min_size'], int(current_size * 0.8))
            reason = f"Scale down: score={scale_down_score:.1f}, utilization={utilization:.1f}"
        
        return ScalingDecision(
            action=action,
            pool_name=pool_name,
            target_size=target_size,
            current_size=current_size,
            confidence=confidence,
            reason=reason,
            metrics_snapshot=current_metrics
        )
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision."""
        with self.scaling_lock:
            try:
                if decision.action == 'scale_up':
                    self.resource_pool._scale_pool(decision.pool_name, decision.target_size)
                elif decision.action == 'scale_down':
                    self.resource_pool._scale_pool(decision.pool_name, decision.target_size)
                
                # Record decision
                self.last_scaling_decisions[decision.pool_name] = time.time()
                self.scaling_history.append(decision)
                
                compiler_logger.logger.info(
                    f"Executed scaling decision: {decision.action} pool '{decision.pool_name}' "
                    f"from {decision.current_size} to {decision.target_size} "
                    f"(confidence: {decision.confidence:.2f}, reason: {decision.reason})"
                )
                
            except Exception as e:
                compiler_logger.logger.error(f"Failed to execute scaling decision: {e}")
    
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get current scaling recommendations without executing them."""
        recommendations = []
        
        try:
            current_usage = self.resource_monitor.get_current_usage()
            pool_stats = self.resource_pool.get_pool_stats()
            
            metrics = ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=current_usage.get('process_cpu_percent', 0),
                memory_utilization=current_usage.get('system_memory_percent', 0),
                queue_length=0,
                active_tasks=sum(pool['in_use'] for pool in pool_stats.values()),
                completed_tasks_per_minute=self._calculate_completion_rate(),
                average_task_duration=self._calculate_average_duration(),
                resource_pool_utilization={
                    name: pool['utilization'] for name, pool in pool_stats.items()
                }
            )
            
            for pool_name in pool_stats.keys():
                decision = self._make_scaling_decision(pool_name, metrics)
                
                recommendations.append({
                    'pool_name': pool_name,
                    'current_size': decision.current_size,
                    'recommended_action': decision.action,
                    'target_size': decision.target_size,
                    'confidence': decision.confidence,
                    'reason': decision.reason,
                    'utilization': pool_stats[pool_name]['utilization']
                })
                
        except Exception as e:
            compiler_logger.logger.error(f"Error generating scaling recommendations: {e}")
        
        return recommendations
    
    def force_scaling_action(
        self,
        pool_name: str,
        action: str,
        target_size: Optional[int] = None
    ) -> bool:
        """Force a scaling action (for manual intervention)."""
        try:
            pool_stats = self.resource_pool.get_pool_stats(pool_name)
            if not pool_stats:
                return False
            
            current_size = pool_stats['current_size']
            
            if action == 'scale_up':
                new_target = target_size or min(pool_stats['max_size'], int(current_size * 1.5))
            elif action == 'scale_down':
                new_target = target_size or max(pool_stats['min_size'], int(current_size * 0.8))
            else:
                return False
            
            # Create manual decision
            metrics = ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=0,
                memory_utilization=0,
                queue_length=0,
                active_tasks=0,
                completed_tasks_per_minute=0,
                average_task_duration=0
            )
            
            decision = ScalingDecision(
                action=action,
                pool_name=pool_name,
                target_size=new_target,
                current_size=current_size,
                confidence=1.0,
                reason="Manual intervention",
                metrics_snapshot=metrics
            )
            
            self._execute_scaling_decision(decision)
            return True
            
        except Exception as e:
            compiler_logger.logger.error(f"Failed to force scaling action: {e}")
            return False
    
    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        history = list(self.scaling_history)[-limit:]
        return [
            {
                'timestamp': decision.metrics_snapshot.timestamp,
                'pool_name': decision.pool_name,
                'action': decision.action,
                'from_size': decision.current_size,
                'to_size': decision.target_size,
                'confidence': decision.confidence,
                'reason': decision.reason,
                'cpu_utilization': decision.metrics_snapshot.cpu_utilization,
                'memory_utilization': decision.metrics_snapshot.memory_utilization
            }
            for decision in history
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        return {
            'monitoring_active': self.monitoring_active,
            'total_scaling_decisions': len(self.scaling_history),
            'scaling_config': self.scaling_config,
            'prediction_confidence': self.predictive_scaler.get_confidence(),
            'metrics_history_size': len(self.predictive_scaler.metrics_history),
            'last_scaling_times': dict(self.last_scaling_decisions),
            'recent_recommendations': self.get_scaling_recommendations()
        }
    
    def cleanup(self) -> None:
        """Cleanup auto-scaler resources."""
        self.stop_monitoring()
        compiler_logger.logger.info("Auto-scaler cleanup completed")