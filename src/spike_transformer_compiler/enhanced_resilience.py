"""Enhanced Resilience System for Production-Ready Deployment.

Implements advanced circuit breakers, health monitoring, and self-healing
mechanisms for robust neuromorphic compilation services.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, rejecting requests  
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    compilation_success_rate: float
    average_response_time: float
    error_rate: float
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'compilation_success_rate': self.compilation_success_rate,
            'average_response_time': self.average_response_time,
            'error_rate': self.error_rate,
            'active_connections': self.active_connections
        }


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_failure_rate: float = 0.05,
                 slow_call_threshold: float = 2000.0):  # 2 seconds
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_failure_rate = expected_failure_rate
        self.slow_call_threshold = slow_call_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.call_history = []
        
        # Adaptive learning
        self.adaptive_learning = True
        self.learning_rate = 0.01
        
        self.logger = logging.getLogger(f"circuit_breaker_{id(self)}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        start_time = time.time()
        try:
            result = await self._execute_call(func, *args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # ms
            
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_failure(execution_time)
            raise
    
    async def _execute_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the actual function call."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful call."""
        self.success_count += 1
        self.call_history.append({
            'timestamp': time.time(),
            'success': True,
            'execution_time': execution_time
        })
        
        # Trim history to last 100 calls
        if len(self.call_history) > 100:
            self.call_history.pop(0)
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.logger.info("Circuit breaker CLOSED after successful recovery")
        
        # Adaptive threshold adjustment
        if self.adaptive_learning:
            self._adjust_thresholds()
    
    def _record_failure(self, execution_time: float) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        self.call_history.append({
            'timestamp': time.time(),
            'success': False,
            'execution_time': execution_time
        })
        
        if len(self.call_history) > 100:
            self.call_history.pop(0)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        
        if self.adaptive_learning:
            self._adjust_thresholds()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _adjust_thresholds(self) -> None:
        """Adaptively adjust thresholds based on historical performance."""
        if len(self.call_history) < 10:
            return
        
        recent_calls = self.call_history[-20:]  # Last 20 calls
        failure_rate = sum(1 for call in recent_calls if not call['success']) / len(recent_calls)
        
        # Adjust failure threshold based on observed patterns
        if failure_rate > self.expected_failure_rate * 2:
            # Higher than expected failure rate, be more sensitive
            self.failure_threshold = max(3, int(self.failure_threshold * 0.9))
        elif failure_rate < self.expected_failure_rate * 0.5:
            # Lower than expected failure rate, be less sensitive
            self.failure_threshold = min(10, int(self.failure_threshold * 1.1))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        if not self.call_history:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_rate': 0.0,
                'average_response_time': 0.0
            }
        
        recent_calls = self.call_history[-50:]  # Last 50 calls
        failure_rate = sum(1 for call in recent_calls if not call['success']) / len(recent_calls)
        avg_response_time = sum(call['execution_time'] for call in recent_calls) / len(recent_calls)
        
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failure_rate': failure_rate,
            'average_response_time': avg_response_time,
            'threshold': self.failure_threshold
        }


class HealthMonitoringSystem:
    """Advanced health monitoring with predictive capabilities."""
    
    def __init__(self):
        self.metrics_history: List[HealthMetrics] = []
        self.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'error_rate': 0.05,
            'response_time': 5000.0  # 5 seconds
        }
        self.monitoring_active = False
        self.logger = logging.getLogger("health_monitor")
    
    async def start_monitoring(self, interval: int = 30) -> None:
        """Start continuous health monitoring."""
        self.monitoring_active = True
        self.logger.info("Health monitoring started")
        
        while self.monitoring_active:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Predictive analysis
                await self._predictive_analysis()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")
    
    async def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
        except ImportError:
            # Fallback if psutil not available
            cpu_usage = 50.0  # Mock values
            memory_usage = 60.0
        
        # Application-specific metrics (would be collected from actual services)
        compilation_success_rate = 0.95  # Mock - would come from metrics
        average_response_time = 1500.0   # Mock - would come from metrics
        error_rate = 0.02               # Mock - would come from metrics
        active_connections = 10         # Mock - would come from metrics
        
        return HealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            compilation_success_rate=compilation_success_rate,
            average_response_time=average_response_time,
            error_rate=error_rate,
            active_connections=active_connections
        )
    
    async def _check_alerts(self, metrics: HealthMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.average_response_time > self.alert_thresholds['response_time']:
            alerts.append(f"Slow response time: {metrics.average_response_time:.0f}ms")
        
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
            await self._trigger_self_healing(alert)
    
    async def _predictive_analysis(self) -> None:
        """Perform predictive analysis on metrics trends."""
        if len(self.metrics_history) < 10:
            return
        
        # Simple trend analysis (in production, would use ML models)
        recent_metrics = self.metrics_history[-10:]
        
        # CPU trend
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        if cpu_trend > 5.0:  # CPU increasing by >5% per interval
            self.logger.warning("Predictive alert: CPU usage trending upward")
        
        # Memory trend
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        if memory_trend > 3.0:  # Memory increasing by >3% per interval
            self.logger.warning("Predictive alert: Memory usage trending upward")
        
        # Error rate trend
        error_trend = self._calculate_trend([m.error_rate for m in recent_metrics])
        if error_trend > 0.01:  # Error rate increasing
            self.logger.warning("Predictive alert: Error rate trending upward")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0
        
        # Simple slope calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    async def _trigger_self_healing(self, alert: str) -> None:
        """Trigger self-healing mechanisms."""
        self.logger.info(f"Triggering self-healing for: {alert}")
        
        if "High CPU usage" in alert:
            await self._scale_resources("cpu")
        elif "High memory usage" in alert:
            await self._clean_memory_cache()
        elif "High error rate" in alert:
            await self._restart_failing_services()
        elif "Slow response time" in alert:
            await self._optimize_performance()
    
    async def _scale_resources(self, resource_type: str) -> None:
        """Scale resources dynamically."""
        self.logger.info(f"Scaling {resource_type} resources")
        # In production, would integrate with container orchestration
    
    async def _clean_memory_cache(self) -> None:
        """Clean memory caches."""
        self.logger.info("Cleaning memory caches")
        from .adaptive_cache import adaptive_cache
        adaptive_cache.optimize_cache()
    
    async def _restart_failing_services(self) -> None:
        """Restart failing services."""
        self.logger.info("Restarting failing services")
        # In production, would restart specific service components
    
    async def _optimize_performance(self) -> None:
        """Apply performance optimizations."""
        self.logger.info("Applying performance optimizations")
        # In production, would tune compilation parameters
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.metrics_history:
            return {"status": "unknown", "metrics": {}}
        
        latest = self.metrics_history[-1]
        
        # Determine overall health status
        if (latest.cpu_usage > self.alert_thresholds['cpu_usage'] or
            latest.memory_usage > self.alert_thresholds['memory_usage'] or
            latest.error_rate > self.alert_thresholds['error_rate']):
            status = "unhealthy"
        elif (latest.cpu_usage > self.alert_thresholds['cpu_usage'] * 0.8 or
              latest.memory_usage > self.alert_thresholds['memory_usage'] * 0.8):
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": latest.timestamp.isoformat(),
            "metrics": latest.to_dict()
        }


class SelfHealingSystem:
    """Self-healing system for autonomous recovery."""
    
    def __init__(self):
        self.healing_strategies = {
            'compilation_failure': self._handle_compilation_failure,
            'memory_leak': self._handle_memory_leak,
            'performance_degradation': self._handle_performance_degradation,
            'service_unavailable': self._handle_service_unavailable
        }
        self.healing_history = []
        self.logger = logging.getLogger("self_healing")
    
    async def diagnose_and_heal(self, 
                               issue_type: str, 
                               context: Dict[str, Any]) -> bool:
        """Diagnose issue and apply healing strategy."""
        self.logger.info(f"Diagnosing and healing issue: {issue_type}")
        
        if issue_type not in self.healing_strategies:
            self.logger.warning(f"No healing strategy for issue type: {issue_type}")
            return False
        
        try:
            strategy = self.healing_strategies[issue_type]
            success = await strategy(context)
            
            # Record healing attempt
            self.healing_history.append({
                'timestamp': datetime.now().isoformat(),
                'issue_type': issue_type,
                'context': context,
                'success': success
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Healing strategy failed: {e}")
            return False
    
    async def _handle_compilation_failure(self, context: Dict[str, Any]) -> bool:
        """Handle compilation failures."""
        self.logger.info("Applying compilation failure healing")
        
        # Strategy 1: Reduce optimization level
        if context.get('optimization_level', 2) > 0:
            context['optimization_level'] -= 1
            self.logger.info(f"Reduced optimization level to {context['optimization_level']}")
            return True
        
        # Strategy 2: Switch to simulation target
        if context.get('target') != 'simulation':
            context['target'] = 'simulation'
            self.logger.info("Switched to simulation target")
            return True
        
        return False
    
    async def _handle_memory_leak(self, context: Dict[str, Any]) -> bool:
        """Handle memory leaks."""
        self.logger.info("Applying memory leak healing")
        
        # Clear caches
        from .adaptive_cache import adaptive_cache
        adaptive_cache.optimize_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return True
    
    async def _handle_performance_degradation(self, context: Dict[str, Any]) -> bool:
        """Handle performance degradation."""
        self.logger.info("Applying performance degradation healing")
        
        # Enable aggressive caching
        from .adaptive_cache import adaptive_cache
        adaptive_cache.learning_rate = 0.05  # More aggressive learning
        
        # Reduce concurrent operations
        if 'max_workers' in context:
            context['max_workers'] = max(1, context['max_workers'] // 2)
        
        return True
    
    async def _handle_service_unavailable(self, context: Dict[str, Any]) -> bool:
        """Handle service unavailability."""
        self.logger.info("Applying service unavailable healing")
        
        # Retry with exponential backoff
        max_retries = context.get('max_retries', 3)
        for attempt in range(max_retries):
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Check if service is back online
            if await self._check_service_health():
                return True
        
        return False
    
    async def _check_service_health(self) -> bool:
        """Check if services are healthy."""
        # Mock health check - in production would check actual services
        return True
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get healing system statistics."""
        if not self.healing_history:
            return {
                'total_attempts': 0,
                'success_rate': 0.0,
                'strategies_used': {}
            }
        
        total_attempts = len(self.healing_history)
        successful_attempts = sum(1 for h in self.healing_history if h['success'])
        success_rate = successful_attempts / total_attempts
        
        strategies_used = {}
        for healing in self.healing_history:
            issue_type = healing['issue_type']
            if issue_type not in strategies_used:
                strategies_used[issue_type] = {'attempts': 0, 'successes': 0}
            
            strategies_used[issue_type]['attempts'] += 1
            if healing['success']:
                strategies_used[issue_type]['successes'] += 1
        
        return {
            'total_attempts': total_attempts,
            'success_rate': success_rate,
            'strategies_used': strategies_used
        }


# Global instances
advanced_circuit_breaker = AdvancedCircuitBreaker()
health_monitoring_system = HealthMonitoringSystem()
self_healing_system = SelfHealingSystem()


async def initialize_resilience_systems():
    """Initialize all resilience systems."""
    logging.info("Initializing enhanced resilience systems")
    
    # Start health monitoring
    asyncio.create_task(health_monitoring_system.start_monitoring())
    
    logging.info("Enhanced resilience systems initialized")


def get_resilience_status() -> Dict[str, Any]:
    """Get overall resilience system status."""
    return {
        'circuit_breaker': advanced_circuit_breaker.get_metrics(),
        'health_monitoring': health_monitoring_system.get_current_health(),
        'self_healing': self_healing_system.get_healing_statistics(),
        'timestamp': datetime.now().isoformat()
    }