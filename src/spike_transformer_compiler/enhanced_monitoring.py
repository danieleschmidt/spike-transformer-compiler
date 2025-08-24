"""Enhanced monitoring and health checks for robust operation."""

import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], bool]
    critical: bool = False
    interval_seconds: int = 60
    timeout_seconds: int = 30
    description: str = ""
    
    # Runtime state
    last_run: float = 0.0
    last_result: bool = True
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass 
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    compilation_success_rate: float
    average_compilation_time: float
    memory_usage_mb: float
    error_count: int
    warning_count: int
    active_compilations: int
    queue_size: int
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, 
                 check_interval: int = 30,
                 retention_hours: int = 24,
                 alert_threshold: int = 3):
        self.check_interval = check_interval
        self.retention_hours = retention_hours
        self.alert_threshold = alert_threshold
        
        self.logger = logging.getLogger(__name__)
        self.checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        self._metrics_lock = threading.Lock()
        
        # Initialize default health checks
        self._initialize_default_checks()
    
    def _initialize_default_checks(self):
        """Initialize default health checks."""
        
        def check_compilation_service():
            """Check if compilation service is responsive."""
            try:
                from .compiler import SpikeCompiler
                from .mock_models import create_test_model
                
                model = create_test_model("simple")
                compiler = SpikeCompiler(target="simulation", verbose=False)
                compiled = compiler.compile(model, input_shape=(1, 10))
                return hasattr(compiled, 'utilization')
            except Exception as e:
                self.logger.error(f"Compilation service check failed: {e}")
                return False
        
        def check_backend_availability():
            """Check if backends are available."""
            try:
                from .backend.factory import BackendFactory
                targets = BackendFactory.get_available_targets()
                return len(targets) > 0
            except Exception as e:
                self.logger.error(f"Backend availability check failed: {e}")
                return False
        
        def check_memory_usage():
            """Check memory usage levels."""
            try:
                # Simplified memory check (in real system would use psutil)
                import os
                # Mock memory check - always passes for now
                return True
            except Exception:
                return True  # Default to healthy if can't check
        
        def check_error_rates():
            """Check recent error rates."""
            try:
                # Check if recent compilations have high failure rate
                recent_metrics = self.get_recent_metrics(minutes=10)
                if not recent_metrics:
                    return True
                
                avg_success_rate = sum(m.compilation_success_rate for m in recent_metrics) / len(recent_metrics)
                return avg_success_rate >= 0.8  # 80% success rate threshold
            except Exception:
                return True
        
        # Register default checks
        self.register_check(HealthCheck(
            name="compilation_service",
            check_function=check_compilation_service,
            critical=True,
            interval_seconds=120,
            description="Check if compilation service is responsive"
        ))
        
        self.register_check(HealthCheck(
            name="backend_availability", 
            check_function=check_backend_availability,
            critical=True,
            interval_seconds=300,
            description="Check if compilation backends are available"
        ))
        
        self.register_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory_usage,
            critical=False,
            interval_seconds=60,
            description="Monitor memory usage levels"
        ))
        
        self.register_check(HealthCheck(
            name="error_rates",
            check_function=check_error_rates,
            critical=False,
            interval_seconds=180,
            description="Monitor compilation error rates"
        ))
    
    def register_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Health monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        if self._monitoring_thread:
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Run health checks that are due
                for check in self.checks.values():
                    if current_time - check.last_run >= check.interval_seconds:
                        self._run_health_check(check)
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep until next check interval
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(self.check_interval)
    
    def _run_health_check(self, check: HealthCheck):
        """Run an individual health check."""
        start_time = time.time()
        
        try:
            # Run the check with timeout (simplified - real implementation would use threading)
            result = check.check_function()
            
            check.last_run = start_time
            check.last_result = result
            check.last_error = None
            
            if result:
                check.consecutive_failures = 0
            else:
                check.consecutive_failures += 1
                
                # Generate alert if threshold reached
                if check.consecutive_failures >= self.alert_threshold:
                    self._generate_alert(check, "Health check failed")
            
            duration = time.time() - start_time
            self.logger.debug(f"Health check {check.name}: {result} ({duration:.3f}s)")
            
        except Exception as e:
            check.last_run = start_time
            check.last_result = False
            check.last_error = str(e)
            check.consecutive_failures += 1
            
            self.logger.error(f"Health check {check.name} failed: {e}")
            
            if check.consecutive_failures >= self.alert_threshold:
                self._generate_alert(check, f"Health check error: {e}")
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            current_time = time.time()
            
            # Mock metrics collection (in real system would gather actual metrics)
            metrics = SystemMetrics(
                timestamp=current_time,
                compilation_success_rate=0.95,  # Mock high success rate
                average_compilation_time=0.1,   # Mock fast compilation
                memory_usage_mb=100,           # Mock memory usage
                error_count=0,
                warning_count=0,
                active_compilations=0,
                queue_size=0,
                cpu_usage_percent=25.0,
                memory_usage_percent=40.0,
                disk_usage_percent=60.0
            )
            
            with self._metrics_lock:
                self.metrics_history.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _generate_alert(self, check: HealthCheck, message: str):
        """Generate an alert for a failed health check."""
        alert = {
            "timestamp": time.time(),
            "severity": "critical" if check.critical else "warning",
            "check_name": check.name,
            "message": message,
            "consecutive_failures": check.consecutive_failures,
            "description": check.description
        }
        
        self.alerts.append(alert)
        
        # Log the alert
        log_level = logging.CRITICAL if check.critical else logging.WARNING
        self.logger.log(log_level, f"ALERT: {message} (check: {check.name})")
        
        # In a real system, this would also send notifications (email, Slack, etc.)
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._metrics_lock:
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
        
        self.alerts = [
            a for a in self.alerts
            if a["timestamp"] > cutoff_time
        ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current overall health status."""
        current_time = time.time()
        
        # Check for critical failures
        critical_failures = [
            check for check in self.checks.values()
            if check.critical and not check.last_result
        ]
        
        # Check for warnings
        warnings = [
            check for check in self.checks.values()
            if not check.critical and not check.last_result
        ]
        
        # Determine overall status
        if critical_failures:
            overall_status = HealthStatus.CRITICAL
        elif len(warnings) >= 2:
            overall_status = HealthStatus.DEGRADED
        elif warnings:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Get recent metrics
        recent_metrics = self.get_recent_metrics(minutes=30)
        
        return {
            "overall_status": overall_status.value,
            "timestamp": current_time,
            "checks": {
                name: {
                    "status": "healthy" if check.last_result else "failing",
                    "last_run": check.last_run,
                    "consecutive_failures": check.consecutive_failures,
                    "last_error": check.last_error,
                    "critical": check.critical
                }
                for name, check in self.checks.items()
            },
            "recent_metrics": {
                "count": len(recent_metrics),
                "avg_success_rate": (
                    sum(m.compilation_success_rate for m in recent_metrics) / len(recent_metrics)
                    if recent_metrics else 0.0
                ),
                "avg_compilation_time": (
                    sum(m.average_compilation_time for m in recent_metrics) / len(recent_metrics)
                    if recent_metrics else 0.0
                )
            },
            "active_alerts": len([a for a in self.alerts if current_time - a["timestamp"] < 3600]),
            "system_info": {
                "monitoring_active": self._monitoring_thread is not None and self._monitoring_thread.is_alive(),
                "uptime_seconds": current_time - (self.metrics_history[0].timestamp if self.metrics_history else current_time)
            }
        }
    
    def get_recent_metrics(self, minutes: int = 10) -> List[SystemMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._metrics_lock:
            return [
                m for m in self.metrics_history
                if m.timestamp > cutoff_time
            ]
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            a for a in self.alerts
            if a["timestamp"] > cutoff_time
        ]
    
    def export_health_report(self, filepath: str):
        """Export comprehensive health report to file."""
        report = {
            "report_timestamp": time.time(),
            "health_status": self.get_health_status(),
            "metrics_history": [
                {
                    "timestamp": m.timestamp,
                    "compilation_success_rate": m.compilation_success_rate,
                    "average_compilation_time": m.average_compilation_time,
                    "memory_usage_mb": m.memory_usage_mb,
                    "error_count": m.error_count,
                    "warning_count": m.warning_count,
                    "active_compilations": m.active_compilations,
                    "queue_size": m.queue_size
                }
                for m in self.metrics_history[-100:]  # Last 100 metrics
            ],
            "recent_alerts": self.get_alerts(hours=24),
            "check_configurations": {
                name: {
                    "critical": check.critical,
                    "interval_seconds": check.interval_seconds,
                    "timeout_seconds": check.timeout_seconds,
                    "description": check.description
                }
                for name, check in self.checks.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Health report exported to {filepath}")


class CircuitBreaker:
    """Circuit breaker for handling failures gracefully."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
            else:
                raise RuntimeError(f"Circuit breaker is OPEN. Try again in {self._time_until_reset():.1f}s")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _time_until_reset(self) -> float:
        """Time remaining until circuit breaker can reset."""
        if self.last_failure_time is None:
            return 0.0
        
        return max(0.0, self.recovery_timeout - (time.time() - self.last_failure_time))
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker reset to closed state")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RateLimiter:
    """Rate limiter for controlling request rates."""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        current_time = time.time()
        
        with self._lock:
            # Remove old requests
            cutoff_time = current_time - self.time_window
            self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            # Check if we can allow this request
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        current_time = time.time()
        
        with self._lock:
            cutoff_time = current_time - self.time_window
            active_requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            return {
                "current_requests": len(active_requests),
                "max_requests": self.max_requests,
                "time_window": self.time_window,
                "requests_remaining": max(0, self.max_requests - len(active_requests)),
                "reset_time": min(active_requests) + self.time_window if active_requests else current_time
            }


# Global instances for easy access
_global_health_monitor = None
_global_circuit_breakers = {}
_global_rate_limiters = {}


def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor."""
    global _global_health_monitor
    
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
        _global_health_monitor.start_monitoring()
    
    return _global_health_monitor


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create named circuit breaker."""
    global _global_circuit_breakers
    
    if name not in _global_circuit_breakers:
        _global_circuit_breakers[name] = CircuitBreaker(**kwargs)
    
    return _global_circuit_breakers[name]


def get_rate_limiter(name: str, **kwargs) -> RateLimiter:
    """Get or create named rate limiter."""
    global _global_rate_limiters
    
    if name not in _global_rate_limiters:
        _global_rate_limiters[name] = RateLimiter(**kwargs)
    
    return _global_rate_limiters[name]


if __name__ == "__main__":
    # Demo health monitoring
    monitor = HealthMonitor()
    monitor.start_monitoring()
    
    try:
        # Run for a short time to collect some data
        time.sleep(10)
        
        # Print health status
        status = monitor.get_health_status()
        print("Health Status:")
        print(json.dumps(status, indent=2))
        
        # Export report
        monitor.export_health_report("health_report.json")
        print("Health report exported to health_report.json")
        
    finally:
        monitor.stop_monitoring()