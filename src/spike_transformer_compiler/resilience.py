"""Resilience and fault tolerance mechanisms for Spike-Transformer-Compiler."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback
from contextlib import contextmanager

from .logging_config import compiler_logger
from .exceptions import CompilationError, BackendError, ResourceError


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit tripped, failing fast
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 3           # Successes to close from half-open
    timeout_duration: float = 30.0       # Seconds before trying half-open
    monitoring_period: float = 60.0      # Sliding window in seconds
    max_failures_per_period: int = 10    # Max failures in monitoring period
    

class CircuitBreaker:
    """Circuit breaker for resilient service calls."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.failures_in_period: List[float] = []
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            current_time = time.time()
            
            # Clean old failures from monitoring period
            self._cleanup_old_failures(current_time)
            
            # Check if circuit should be opened
            if self._should_trip(current_time):
                self.state = CircuitState.OPEN
                self.last_failure_time = current_time
            
            # Handle different states
            if self.state == CircuitState.OPEN:
                if current_time - self.last_failure_time >= self.config.timeout_duration:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    compiler_logger.logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Try again in {self.config.timeout_duration - (current_time - self.last_failure_time):.1f}s"
                    )
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(current_time)
            raise
    
    def _should_trip(self, current_time: float) -> bool:
        """Check if circuit should be tripped."""
        if self.state == CircuitState.OPEN:
            return False
            
        # Trip if too many failures in current period
        if len(self.failures_in_period) >= self.config.max_failures_per_period:
            return True
            
        # Trip if consecutive failures exceed threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
            
        return False
    
    def _record_success(self):
        """Record successful execution."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    compiler_logger.logger.info(f"Circuit breaker {self.name} closed")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    def _record_failure(self, current_time: float):
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = current_time
            self.failures_in_period.append(current_time)
            
            compiler_logger.logger.warning(
                f"Circuit breaker {self.name} recorded failure "
                f"({self.failure_count}/{self.config.failure_threshold})"
            )
    
    def _cleanup_old_failures(self, current_time: float):
        """Remove failures outside monitoring period."""
        cutoff_time = current_time - self.config.monitoring_period
        self.failures_in_period = [
            failure_time for failure_time in self.failures_in_period
            if failure_time > cutoff_time
        ]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'failures_in_period': len(self.failures_in_period),
            'last_failure_time': self.last_failure_time
        }
    
    def __enter__(self):
        """Enter context manager - check circuit state."""
        with self._lock:
            current_time = time.time()
            
            # Clean old failures from monitoring period
            self._cleanup_old_failures(current_time)
            
            # Check if circuit should be opened
            if self._should_trip(current_time):
                self.state = CircuitState.OPEN
                self.last_failure_time = current_time
            
            # Handle different states
            if self.state == CircuitState.OPEN:
                if current_time - self.last_failure_time >= self.config.timeout_duration:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    compiler_logger.logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Try again in {self.config.timeout_duration - (current_time - self.last_failure_time):.1f}s"
                    )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - record success or failure."""
        current_time = time.time()
        
        if exc_type is None:
            # Success
            self._record_success()
        else:
            # Failure
            self._record_failure(current_time)
        
        return False  # Don't suppress exceptions


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[type]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            CompilationError, BackendError, ResourceError
        ]


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to functions."""
    retry_config = config or RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    is_retryable = any(
                        isinstance(e, exc_type) 
                        for exc_type in retry_config.retryable_exceptions
                    )
                    
                    if not is_retryable or attempt == retry_config.max_attempts - 1:
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    compiler_logger.logger.warning(
                        f"Attempt {attempt + 1}/{retry_config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
            
        return wrapper
    return decorator


class FallbackStrategy:
    """Base class for fallback strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute fallback strategy."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if fallback is available."""
        return True


class SimulationFallback(FallbackStrategy):
    """Fallback to simulation when hardware is unavailable."""
    
    def __init__(self):
        super().__init__("simulation_fallback")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute compilation with simulation backend."""
        from .backend.factory import BackendFactory
        
        compiler_logger.logger.info("Falling back to simulation backend")
        
        # Create simulation backend
        backend = BackendFactory.create_backend("simulation")
        
        # Execute with modified parameters
        kwargs = kwargs.copy()
        kwargs['target'] = 'simulation'
        kwargs['fallback_mode'] = True
        
        return backend.compile_graph(*args, **kwargs)
    
    def is_available(self) -> bool:
        """Simulation should always be available."""
        return True


class CachedResultFallback(FallbackStrategy):
    """Fallback to cached results when compilation fails."""
    
    def __init__(self):
        super().__init__("cached_result_fallback")
    
    def execute(self, model_hash: str, *args, **kwargs) -> Any:
        """Try to return cached result."""
        from .performance import get_compilation_cache
        
        cache = get_compilation_cache()
        
        # Try to find similar cached result
        cached_result = self._find_similar_cached_result(cache, model_hash, *args, **kwargs)
        
        if cached_result:
            compiler_logger.logger.info(f"Using cached result as fallback for {model_hash[:16]}")
            return cached_result
        
        raise CompilationError("No suitable cached result found for fallback")
    
    def _find_similar_cached_result(self, cache, model_hash: str, *args, **kwargs):
        """Find similar cached result that could serve as fallback."""
        # This would implement logic to find similar compilations
        # For now, just return None
        return None
    
    def is_available(self) -> bool:
        """Check if cache has relevant entries."""
        try:
            from .performance import get_compilation_cache
            cache = get_compilation_cache()
            stats = cache.get_stats()
            return stats.get('memory_entries', 0) > 0
        except Exception:
            return False


class GracefulDegradationFallback(FallbackStrategy):
    """Fallback with reduced functionality."""
    
    def __init__(self):
        super().__init__("graceful_degradation")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute with reduced optimization level."""
        compiler_logger.logger.info("Falling back to reduced optimization compilation")
        
        # Create a simple degraded compilation
        kwargs = kwargs.copy()
        kwargs['degraded_mode'] = True
        
        # Disable advanced features
        kwargs['profile_energy'] = False
        kwargs['secure_mode'] = False
        kwargs['enable_resilience'] = False
        
        # Re-attempt compilation with reduced parameters
        from .compiler import SpikeCompiler
        
        compiler = SpikeCompiler(
            target=kwargs.get('target', 'simulation'),
            optimization_level=0,  # Minimal optimization
            verbose=False
        )
        
        # Extract model and input_shape from args
        model = args[0] if len(args) > 0 else kwargs.get('model')
        input_shape = args[1] if len(args) > 1 else kwargs.get('input_shape')
        
        return compiler.compile(model, input_shape, **kwargs)


class ResilientCompilationManager:
    """Manager for resilient compilation with fallbacks."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_strategies: List[FallbackStrategy] = [
            SimulationFallback(),
            CachedResultFallback(),
            GracefulDegradationFallback()
        ]
        
        # Create circuit breakers for different components
        self._init_circuit_breakers()
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for different services."""
        components = [
            'frontend_parsing',
            'optimization',
            'backend_compilation',
            'loihi3_backend',
            'simulation_backend'
        ]
        
        for component in components:
            self.circuit_breakers[component] = CircuitBreaker(
                name=component,
                config=CircuitBreakerConfig(
                    failure_threshold=3,
                    timeout_duration=30.0
                )
            )
    
    @contextmanager
    def resilient_operation(self, operation_name: str):
        """Context manager for resilient operations."""
        circuit_breaker = self.circuit_breakers.get(
            operation_name,
            CircuitBreaker(operation_name)
        )
        
        try:
            with circuit_breaker:
                yield circuit_breaker
                
        except CircuitBreakerOpenError:
            # Circuit is open, try fallbacks
            compiler_logger.logger.warning(f"Circuit breaker {operation_name} is open, trying fallbacks")
            raise
    
    def compile_with_resilience(self, compiler, model, input_shape, **kwargs) -> Any:
        """Compile with full resilience mechanisms."""
        primary_attempt = True
        last_exception = None
        
        # Try primary compilation path
        try:
            return self._attempt_compilation(compiler, model, input_shape, **kwargs)
            
        except Exception as e:
            last_exception = e
            compiler_logger.logger.warning(f"Primary compilation failed: {e}")
            primary_attempt = False
        
        # Try fallback strategies
        for strategy in self.fallback_strategies:
            if not strategy.is_available():
                continue
                
            try:
                compiler_logger.logger.info(f"Attempting fallback strategy: {strategy.name}")
                result = strategy.execute(model, input_shape, **kwargs)
                
                # Mark as fallback result
                if hasattr(result, 'metadata'):
                    result.metadata = result.metadata or {}
                    result.metadata['fallback_strategy'] = strategy.name
                    result.metadata['degraded_mode'] = True
                
                return result
                
            except Exception as e:
                compiler_logger.logger.warning(f"Fallback strategy {strategy.name} failed: {e}")
                continue
        
        # All strategies failed
        raise CompilationError(
            f"All compilation strategies failed. Primary error: {last_exception}"
        ) from last_exception
    
    def _attempt_compilation(self, compiler, model, input_shape, **kwargs):
        """Attempt compilation with circuit breaker protection."""
        # Use circuit breakers for different stages
        with self.resilient_operation('frontend_parsing'):
            # This would be the actual frontend parsing
            pass
            
        with self.resilient_operation('optimization'):
            # This would be the optimization phase
            pass
            
        with self.resilient_operation('backend_compilation'):
            # This would be the backend compilation
            return compiler.compile(model, input_shape, **kwargs)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health = {
            'overall_status': 'healthy',
            'circuit_breakers': {},
            'fallback_strategies': {},
            'recommendations': []
        }
        
        # Check circuit breaker states
        open_breakers = 0
        for name, cb in self.circuit_breakers.items():
            cb_state = cb.get_state()
            health['circuit_breakers'][name] = cb_state
            
            if cb_state['state'] == 'open':
                open_breakers += 1
        
        # Check fallback strategy availability
        available_fallbacks = 0
        for strategy in self.fallback_strategies:
            is_available = strategy.is_available()
            health['fallback_strategies'][strategy.name] = {
                'available': is_available,
                'name': strategy.name
            }
            if is_available:
                available_fallbacks += 1
        
        # Determine overall health
        if open_breakers > len(self.circuit_breakers) * 0.5:
            health['overall_status'] = 'unhealthy'
            health['recommendations'].append("Multiple circuit breakers open - system may be experiencing issues")
        elif open_breakers > 0:
            health['overall_status'] = 'degraded'
            health['recommendations'].append("Some services are failing - using fallbacks")
        
        if available_fallbacks == 0:
            health['overall_status'] = 'critical'
            health['recommendations'].append("No fallback strategies available - system vulnerable to failures")
        
        return health


# Global resilient manager instance
_resilient_manager: Optional[ResilientCompilationManager] = None


def get_resilient_manager() -> ResilientCompilationManager:
    """Get global resilient compilation manager."""
    global _resilient_manager
    if _resilient_manager is None:
        _resilient_manager = ResilientCompilationManager()
    return _resilient_manager


# Decorators for easy use
def resilient_compilation(func: Callable) -> Callable:
    """Decorator to make compilation functions resilient."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = get_resilient_manager()
        return manager.compile_with_resilience(func, *args, **kwargs)
    return wrapper


def circuit_protected(component_name: str):
    """Decorator to protect functions with circuit breakers."""
    def decorator(func: Callable) -> Callable:
        manager = get_resilient_manager()
        circuit_breaker = manager.circuit_breakers.get(
            component_name,
            CircuitBreaker(component_name)
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    
    return decorator