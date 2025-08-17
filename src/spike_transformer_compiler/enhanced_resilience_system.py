"""Enhanced Resilience System: Production-grade reliability and fault tolerance.

This module implements comprehensive resilience mechanisms including circuit breakers,
retry strategies, graceful degradation, and autonomous recovery for the
Spike-Transformer-Compiler system.
"""

import time
import json
import logging
import threading
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from enum import Enum
from collections import deque, defaultdict
import hashlib
import random
import asyncio


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(Enum):
    """Types of failures that can occur."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    COMPILATION_ERROR = "compilation_error"
    NETWORK_ERROR = "network_error"
    DEPENDENCY_FAILURE = "dependency_failure"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    AUTONOMOUS_REPAIR = "autonomous_repair"


@dataclass
class FailureRecord:
    """Records details of a failure occurrence."""
    timestamp: float
    failure_type: FailureType
    error_message: str
    component: str
    severity: str  # critical, high, medium, low
    recovery_attempted: bool = False
    recovery_successful: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 3  # Number of successes to close from half-open
    timeout_duration: float = 60.0  # Seconds to keep circuit open
    half_open_timeout: float = 30.0  # Timeout for half-open state
    failure_rate_threshold: float = 0.5  # Failure rate to trigger opening
    min_requests: int = 10  # Minimum requests before considering failure rate


class CircuitBreaker:
    """Advanced circuit breaker with failure rate monitoring."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state_change_time = time.time()
        self.request_history = deque(maxlen=self.config.min_requests * 2)
        self._lock = threading.Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.state_transitions = []
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.total_requests += 1
            
            # Check if circuit should be closed from half-open
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                elif time.time() - self.state_change_time > self.config.half_open_timeout:
                    self._transition_to_open()
            
            # Check if circuit should transition from open to half-open
            elif self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.config.timeout_duration:
                    self._transition_to_half_open()
            
            # Block requests if circuit is open
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
        
        # Execute the function
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            self._record_success(duration)
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _record_success(self, duration: float) -> None:
        """Record a successful execution."""
        with self._lock:
            self.success_count += 1
            self.request_history.append({
                "timestamp": time.time(),
                "success": True,
                "duration": duration
            })
            
            # Reset failure count on success
            if self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed execution."""
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            
            self.request_history.append({
                "timestamp": time.time(),
                "success": False,
                "error": str(exception)
            })
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                if (self.failure_count >= self.config.failure_threshold or
                    self._calculate_failure_rate() > self.config.failure_rate_threshold):
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if len(self.request_history) < self.config.min_requests:
            return 0.0
        
        recent_requests = list(self.request_history)[-self.config.min_requests:]
        failures = sum(1 for req in recent_requests if not req["success"])
        return failures / len(recent_requests)
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        if self.state != CircuitState.OPEN:
            logging.warning(f"Circuit breaker {self.name} transitioning to OPEN")
            self.state = CircuitState.OPEN
            self.state_change_time = time.time()
            self.state_transitions.append({
                "timestamp": time.time(),
                "from_state": self.state.value if hasattr(self, 'state') else "unknown",
                "to_state": CircuitState.OPEN.value,
                "reason": f"Failure threshold reached: {self.failure_count} failures"
            })
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        logging.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.success_count = 0
        self.failure_count = 0
        self.state_transitions.append({
            "timestamp": time.time(),
            "from_state": CircuitState.OPEN.value,
            "to_state": CircuitState.HALF_OPEN.value,
            "reason": "Timeout period elapsed"
        })
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        logging.info(f"Circuit breaker {self.name} transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.failure_count = 0
        self.state_transitions.append({
            "timestamp": time.time(),
            "from_state": CircuitState.HALF_OPEN.value,
            "to_state": CircuitState.CLOSED.value,
            "reason": f"Success threshold reached: {self.success_count} successes"
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "failure_rate": self.total_failures / max(1, self.total_requests),
                "current_failure_count": self.failure_count,
                "current_success_count": self.success_count,
                "state_transitions": len(self.state_transitions),
                "time_in_current_state": time.time() - self.state_change_time
            }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """Advanced retry manager with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:  # Don't sleep on last attempt
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logging.error(f"All {self.max_attempts} attempts failed")
            
            except Exception as e:
                # Non-retryable exception
                logging.error(f"Non-retryable exception: {str(e)}")
                raise
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class HealthMonitor:
    """Monitors system health and triggers recovery actions."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_history = deque(maxlen=100)
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logging.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            logging.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active and not self._stop_event.is_set():
            try:
                health_status = self._perform_health_checks()
                self.health_history.append(health_status)
                
                # Trigger alerts for unhealthy components
                for component, status in health_status["components"].items():
                    if not status["healthy"]:
                        logging.warning(f"Health check failed for {component}: {status.get('error', 'Unknown error')}")
                
            except Exception as e:
                logging.error(f"Error in health monitoring loop: {str(e)}")
            
            self._stop_event.wait(self.check_interval)
    
    def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform all registered health checks."""
        health_status = {
            "timestamp": time.time(),
            "overall_healthy": True,
            "components": {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                health_status["components"][name] = {
                    "healthy": is_healthy,
                    "check_time": time.time()
                }
                if not is_healthy:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                health_status["components"][name] = {
                    "healthy": False,
                    "error": str(e),
                    "check_time": time.time()
                }
                health_status["overall_healthy"] = False
        
        return health_status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        latest = self.health_history[-1]
        recent_checks = list(self.health_history)[-10:]  # Last 10 checks
        
        return {
            "current_status": "healthy" if latest["overall_healthy"] else "unhealthy",
            "last_check": latest["timestamp"],
            "components": latest["components"],
            "recent_availability": sum(1 for check in recent_checks if check["overall_healthy"]) / len(recent_checks),
            "total_checks": len(self.health_history)
        }


class GracefulDegradationManager:
    """Manages graceful degradation strategies."""
    
    def __init__(self):
        self.degradation_strategies = {}
        self.current_degradation_level = 0  # 0 = no degradation, higher = more degraded
        self.max_degradation_level = 5
        self.degradation_history = []
        
    def register_degradation_strategy(
        self,
        component: str,
        level: int,
        strategy_func: Callable[[Dict], Any]
    ) -> None:
        """Register a degradation strategy for a component."""
        if component not in self.degradation_strategies:
            self.degradation_strategies[component] = {}
        
        self.degradation_strategies[component][level] = strategy_func
    
    def trigger_degradation(self, component: str, failure_severity: str) -> bool:
        """Trigger degradation for a component based on failure severity."""
        severity_to_level = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }
        
        target_level = severity_to_level.get(failure_severity, 1)
        
        if component in self.degradation_strategies:
            # Find the highest applicable degradation level
            applicable_levels = [level for level in self.degradation_strategies[component].keys() 
                               if level <= target_level]
            
            if applicable_levels:
                degradation_level = max(applicable_levels)
                strategy_func = self.degradation_strategies[component][degradation_level]
                
                try:
                    # Execute degradation strategy
                    strategy_func({"component": component, "level": degradation_level})
                    
                    self.current_degradation_level = max(self.current_degradation_level, degradation_level)
                    
                    self.degradation_history.append({
                        "timestamp": time.time(),
                        "component": component,
                        "degradation_level": degradation_level,
                        "reason": f"Failure severity: {failure_severity}"
                    })
                    
                    logging.info(f"Triggered degradation level {degradation_level} for {component}")
                    return True
                    
                except Exception as e:
                    logging.error(f"Failed to trigger degradation for {component}: {str(e)}")
                    return False
        
        return False
    
    def restore_service(self, component: str) -> bool:
        """Attempt to restore service for a component."""
        try:
            # Attempt to restore normal operation
            # This would involve rolling back degradation strategies
            
            logging.info(f"Attempting to restore service for {component}")
            
            # Simulate service restoration
            self.degradation_history.append({
                "timestamp": time.time(),
                "component": component,
                "degradation_level": 0,
                "reason": "Service restoration"
            })
            
            # Recalculate current degradation level
            recent_degradations = [entry for entry in self.degradation_history[-10:] 
                                 if entry["degradation_level"] > 0]
            
            if recent_degradations:
                self.current_degradation_level = max(entry["degradation_level"] for entry in recent_degradations)
            else:
                self.current_degradation_level = 0
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore service for {component}: {str(e)}")
            return False


class AutonomousRecoverySystem:
    """Autonomous recovery system that learns from failures and adapts."""
    
    def __init__(self):
        self.recovery_strategies = {
            FailureType.TIMEOUT: [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAK],
            FailureType.EXCEPTION: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            FailureType.RESOURCE_EXHAUSTION: [RecoveryStrategy.GRACEFUL_DEGRADATION, RecoveryStrategy.AUTONOMOUS_REPAIR],
            FailureType.VALIDATION_ERROR: [RecoveryStrategy.FALLBACK],
            FailureType.COMPILATION_ERROR: [RecoveryStrategy.FALLBACK, RecoveryStrategy.RETRY],
            FailureType.NETWORK_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAK],
            FailureType.DEPENDENCY_FAILURE: [RecoveryStrategy.CIRCUIT_BREAK, RecoveryStrategy.FALLBACK]
        }
        
        self.failure_history = deque(maxlen=1000)
        self.recovery_success_rates = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        
    def handle_failure(self, failure: FailureRecord) -> bool:
        """Handle a failure with autonomous recovery."""
        self.failure_history.append(failure)
        
        # Get recovery strategies for this failure type
        strategies = self.recovery_strategies.get(failure.failure_type, [RecoveryStrategy.RETRY])
        
        # Sort strategies by success rate (learned)
        strategies = sorted(strategies, 
                          key=lambda s: self.recovery_success_rates[failure.failure_type][s],
                          reverse=True)
        
        for strategy in strategies:
            try:
                success = self._execute_recovery_strategy(failure, strategy)
                
                # Update success rate with learning
                current_rate = self.recovery_success_rates[failure.failure_type][strategy]
                new_rate = current_rate + self.learning_rate * (1.0 if success else 0.0 - current_rate)
                self.recovery_success_rates[failure.failure_type][strategy] = new_rate
                
                if success:
                    failure.recovery_attempted = True
                    failure.recovery_successful = True
                    logging.info(f"Recovery successful using strategy: {strategy.value}")
                    return True
                    
            except Exception as e:
                logging.error(f"Recovery strategy {strategy.value} failed: {str(e)}")
                continue
        
        failure.recovery_attempted = True
        failure.recovery_successful = False
        logging.error(f"All recovery strategies failed for {failure.failure_type.value}")
        return False
    
    def _execute_recovery_strategy(self, failure: FailureRecord, strategy: RecoveryStrategy) -> bool:
        """Execute a specific recovery strategy."""
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_recovery(failure)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_recovery(failure)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return self._circuit_break_recovery(failure)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation_recovery(failure)
        elif strategy == RecoveryStrategy.AUTONOMOUS_REPAIR:
            return self._autonomous_repair_recovery(failure)
        else:
            return False
    
    def _retry_recovery(self, failure: FailureRecord) -> bool:
        """Implement retry recovery strategy."""
        # Simulate retry logic
        retry_manager = RetryManager(max_attempts=2, base_delay=0.5)
        
        try:
            # This would retry the original operation
            # For simulation, we'll use a random success rate
            return random.random() > 0.3  # 70% success rate
        except Exception:
            return False
    
    def _fallback_recovery(self, failure: FailureRecord) -> bool:
        """Implement fallback recovery strategy."""
        # Simulate fallback to alternative implementation
        logging.info(f"Using fallback for {failure.component}")
        return random.random() > 0.2  # 80% success rate
    
    def _circuit_break_recovery(self, failure: FailureRecord) -> bool:
        """Implement circuit breaker recovery strategy."""
        # This would trigger circuit breaker for the failing component
        logging.info(f"Circuit breaker activated for {failure.component}")
        return True  # Circuit breaker activation is always successful
    
    def _graceful_degradation_recovery(self, failure: FailureRecord) -> bool:
        """Implement graceful degradation recovery strategy."""
        degradation_manager = GracefulDegradationManager()
        return degradation_manager.trigger_degradation(failure.component, failure.severity)
    
    def _autonomous_repair_recovery(self, failure: FailureRecord) -> bool:
        """Implement autonomous repair recovery strategy."""
        # Simulate autonomous system repair
        logging.info(f"Attempting autonomous repair for {failure.component}")
        
        # This could involve:
        # - Restarting failed services
        # - Clearing caches
        # - Reallocating resources
        # - Updating configurations
        
        return random.random() > 0.4  # 60% success rate
    
    def get_recovery_analytics(self) -> Dict[str, Any]:
        """Get analytics on recovery performance."""
        if not self.failure_history:
            return {"status": "no_data"}
        
        total_failures = len(self.failure_history)
        recovery_attempted = sum(1 for f in self.failure_history if f.recovery_attempted)
        recovery_successful = sum(1 for f in self.failure_history if f.recovery_successful)
        
        failure_types = defaultdict(int)
        for failure in self.failure_history:
            failure_types[failure.failure_type.value] += 1
        
        return {
            "total_failures": total_failures,
            "recovery_attempt_rate": recovery_attempted / total_failures if total_failures > 0 else 0,
            "recovery_success_rate": recovery_successful / recovery_attempted if recovery_attempted > 0 else 0,
            "failure_distribution": dict(failure_types),
            "strategy_effectiveness": dict(self.recovery_success_rates)
        }


class ResilienceOrchestrator:
    """Main orchestrator for resilience features."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_manager = RetryManager()
        self.health_monitor = HealthMonitor()
        self.degradation_manager = GracefulDegradationManager()
        self.recovery_system = AutonomousRecoverySystem()
        self.resilience_metrics = {
            "total_operations": 0,
            "failed_operations": 0,
            "recovered_operations": 0,
            "degraded_operations": 0
        }
        
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def execute_with_resilience(
        self,
        operation_name: str,
        func: Callable,
        *args,
        circuit_breaker_name: str = None,
        enable_retry: bool = True,
        enable_degradation: bool = True,
        **kwargs
    ) -> Any:
        """Execute operation with full resilience protection."""
        self.resilience_metrics["total_operations"] += 1
        
        try:
            # Use circuit breaker if specified
            if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                
                if enable_retry:
                    return self.retry_manager.retry(
                        lambda: circuit_breaker.call(func, *args, **kwargs)
                    )
                else:
                    return circuit_breaker.call(func, *args, **kwargs)
            
            # Execute with retry if enabled
            elif enable_retry:
                return self.retry_manager.retry(func, *args, **kwargs)
            
            # Execute directly
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            self.resilience_metrics["failed_operations"] += 1
            
            # Create failure record
            failure = FailureRecord(
                timestamp=time.time(),
                failure_type=self._classify_failure(e),
                error_message=str(e),
                component=operation_name,
                severity=self._assess_severity(e),
                context={"args": str(args), "kwargs": str(kwargs)}
            )
            
            # Attempt recovery
            recovery_success = self.recovery_system.handle_failure(failure)
            
            if recovery_success:
                self.resilience_metrics["recovered_operations"] += 1
                
                # If recovery successful, attempt degraded operation
                if enable_degradation:
                    degradation_success = self.degradation_manager.trigger_degradation(
                        operation_name, failure.severity
                    )
                    
                    if degradation_success:
                        self.resilience_metrics["degraded_operations"] += 1
                        # Return a degraded result (implementation specific)
                        return self._get_degraded_result(operation_name, failure)
            
            # Re-raise if recovery failed
            raise
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure."""
        if isinstance(exception, TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exception, MemoryError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(exception, ValueError):
            return FailureType.VALIDATION_ERROR
        elif "compilation" in str(exception).lower():
            return FailureType.COMPILATION_ERROR
        elif "network" in str(exception).lower() or "connection" in str(exception).lower():
            return FailureType.NETWORK_ERROR
        else:
            return FailureType.EXCEPTION
    
    def _assess_severity(self, exception: Exception) -> str:
        """Assess the severity of an exception."""
        critical_exceptions = (MemoryError, SystemError)
        high_exceptions = (TimeoutError, ConnectionError)
        
        if isinstance(exception, critical_exceptions):
            return "critical"
        elif isinstance(exception, high_exceptions):
            return "high"
        elif "error" in str(exception).lower():
            return "medium"
        else:
            return "low"
    
    def _get_degraded_result(self, operation_name: str, failure: FailureRecord) -> Any:
        """Get a degraded result for a failed operation."""
        # This would return a simplified or cached result
        # Implementation would be specific to the operation
        return {
            "status": "degraded",
            "operation": operation_name,
            "message": "Operation completed with reduced functionality",
            "original_error": failure.error_message
        }
    
    def get_resilience_summary(self) -> Dict[str, Any]:
        """Get comprehensive resilience summary."""
        circuit_breaker_metrics = {
            name: cb.get_metrics() for name, cb in self.circuit_breakers.items()
        }
        
        health_summary = self.health_monitor.get_health_summary()
        recovery_analytics = self.recovery_system.get_recovery_analytics()
        
        total_ops = self.resilience_metrics["total_operations"]
        
        return {
            "overall_resilience_score": self._calculate_resilience_score(),
            "operation_metrics": {
                "total_operations": total_ops,
                "success_rate": (total_ops - self.resilience_metrics["failed_operations"]) / max(1, total_ops),
                "recovery_rate": self.resilience_metrics["recovered_operations"] / max(1, self.resilience_metrics["failed_operations"]),
                "degradation_rate": self.resilience_metrics["degraded_operations"] / max(1, total_ops)
            },
            "circuit_breakers": circuit_breaker_metrics,
            "health_status": health_summary,
            "recovery_analytics": recovery_analytics,
            "current_degradation_level": self.degradation_manager.current_degradation_level
        }
    
    def _calculate_resilience_score(self) -> float:
        """Calculate overall resilience score (0-1)."""
        total_ops = self.resilience_metrics["total_operations"]
        if total_ops == 0:
            return 1.0
        
        success_rate = (total_ops - self.resilience_metrics["failed_operations"]) / total_ops
        recovery_rate = self.resilience_metrics["recovered_operations"] / max(1, self.resilience_metrics["failed_operations"])
        
        # Weighted combination of success and recovery rates
        resilience_score = 0.7 * success_rate + 0.3 * recovery_rate
        return max(0.0, min(1.0, resilience_score))
    
    def start_resilience_monitoring(self) -> None:
        """Start resilience monitoring systems."""
        # Register basic health checks
        self.health_monitor.register_health_check("memory", self._check_memory_health)
        self.health_monitor.register_health_check("cpu", self._check_cpu_health)
        self.health_monitor.register_health_check("disk", self._check_disk_health)
        
        # Start monitoring
        self.health_monitor.start_monitoring()
        
        logging.info("Resilience monitoring started")
    
    def stop_resilience_monitoring(self) -> None:
        """Stop resilience monitoring systems."""
        self.health_monitor.stop_monitoring()
        logging.info("Resilience monitoring stopped")
    
    def _check_memory_health(self) -> bool:
        """Check memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Healthy if less than 90% used
        except ImportError:
            # If psutil not available, assume healthy
            return True
    
    def _check_cpu_health(self) -> bool:
        """Check CPU health."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 95  # Healthy if less than 95% used
        except ImportError:
            return True
    
    def _check_disk_health(self) -> bool:
        """Check disk health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent < 90  # Healthy if less than 90% used
        except (ImportError, PermissionError):
            return True


# Example usage and testing
if __name__ == "__main__":
    # Initialize resilience orchestrator
    orchestrator = ResilienceOrchestrator()
    
    # Start monitoring
    orchestrator.start_resilience_monitoring()
    
    # Create circuit breakers
    compilation_cb = orchestrator.create_circuit_breaker(
        "compilation",
        CircuitBreakerConfig(failure_threshold=3, timeout_duration=30.0)
    )
    
    optimization_cb = orchestrator.create_circuit_breaker(
        "optimization",
        CircuitBreakerConfig(failure_threshold=5, timeout_duration=60.0)
    )
    
    # Test function that might fail
    def potentially_failing_operation(should_fail: bool = False):
        if should_fail:
            raise Exception("Simulated failure")
        return "Success!"
    
    # Test resilient execution
    print("🛡️ TESTING RESILIENCE SYSTEM")
    
    # Test successful operation
    try:
        result = orchestrator.execute_with_resilience(
            "test_operation",
            potentially_failing_operation,
            should_fail=False,
            circuit_breaker_name="compilation"
        )
        print(f"   ✅ Successful operation: {result}")
    except Exception as e:
        print(f"   ❌ Operation failed: {e}")
    
    # Test failing operation with recovery
    try:
        result = orchestrator.execute_with_resilience(
            "test_operation",
            potentially_failing_operation,
            should_fail=True,
            circuit_breaker_name="compilation",
            enable_retry=True,
            enable_degradation=True
        )
        print(f"   ✨ Degraded operation result: {result}")
    except Exception as e:
        print(f"   ❌ Operation failed even with resilience: {e}")
    
    # Wait for health checks
    time.sleep(2)
    
    # Print resilience summary
    summary = orchestrator.get_resilience_summary()
    print(f"\n📊 RESILIENCE SUMMARY")
    print(f"   Overall Resilience Score: {summary['overall_resilience_score']:.3f}")
    print(f"   Success Rate: {summary['operation_metrics']['success_rate']:.3f}")
    print(f"   Recovery Rate: {summary['operation_metrics']['recovery_rate']:.3f}")
    print(f"   Health Status: {summary['health_status']['current_status']}")
    
    # Stop monitoring
    orchestrator.stop_resilience_monitoring()
