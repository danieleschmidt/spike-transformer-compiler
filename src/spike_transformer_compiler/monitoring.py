"""Advanced monitoring and health check system."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from pathlib import Path

from .logging_config import compiler_logger
from .exceptions import ConfigurationError


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and store performance metrics."""
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        with self._lock:
            self.counters[name] += value
            self._add_metric_point(name, self.counters[name], tags or {})
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self._add_metric_point(name, value, tags or {})
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        with self._lock:
            self._add_metric_point(f"{name}_duration_ms", duration_ms, tags or {})
    
    def _add_metric_point(self, name: str, value: Union[int, float], tags: Dict[str, str]):
        """Add metric point to time series."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags
        )
        self.metrics[name].append(metric)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {'error': f'Metric {name} not found'}
        
        points = list(self.metrics[name])
        if not points:
            return {'error': 'No data points'}
        
        values = [p.value for p in points]
        
        return {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1],
            'latest_timestamp': points[-1].timestamp,
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metric summaries."""
        return {name: self.get_metric_summary(name) for name in self.metrics.keys()}


class CompilationMonitor:
    """Monitor compilation metrics and health."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.compilation_stats = {
            'total_compilations': 0,
            'successful_compilations': 0,
            'failed_compilations': 0,
        }
        self._stats_lock = threading.Lock()
    
    def record_compilation_start(self, model_hash: str, target: str):
        """Record start of compilation."""
        self.metrics.record_counter('compilations_started_total')
        return time.time()
    
    def record_compilation_success(self, model_hash: str, target: str, start_time: float):
        """Record successful compilation."""
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_counter('compilations_successful_total')
        self.metrics.record_timing('compilation_duration', duration_ms)
        
        with self._stats_lock:
            self.compilation_stats['total_compilations'] += 1
            self.compilation_stats['successful_compilations'] += 1
    
    def record_compilation_failure(self, model_hash: str, target: str, start_time: float, error: Exception):
        """Record failed compilation."""
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_counter('compilations_failed_total') 
        self.metrics.record_timing('compilation_duration', duration_ms)
        
        with self._stats_lock:
            self.compilation_stats['total_compilations'] += 1
            self.compilation_stats['failed_compilations'] += 1
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        with self._stats_lock:
            stats = self.compilation_stats.copy()
        
        total = stats['total_compilations']
        if total > 0:
            stats['success_rate'] = stats['successful_compilations'] / total
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        stats = self.get_compilation_stats()
        
        if stats['total_compilations'] == 0:
            return HealthStatus.HEALTHY
        
        success_rate = stats['success_rate']
        if success_rate >= 0.95:
            return HealthStatus.HEALTHY
        elif success_rate >= 0.8:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""
        return {
            'compilation_stats': self.get_compilation_stats(),
            'health_status': self.get_health_status().value,
            'metrics': self.metrics.get_all_metrics(),
            'timestamp': time.time()
        }


# Global monitoring instance
_compilation_monitor: Optional[CompilationMonitor] = None


def get_compilation_monitor() -> CompilationMonitor:
    """Get global compilation monitor."""
    global _compilation_monitor
    if _compilation_monitor is None:
        _compilation_monitor = CompilationMonitor()
    return _compilation_monitor


class CompilationTracking:
    """Context manager for tracking compilation metrics."""
    
    def __init__(self, model_hash: str, target: str):
        self.model_hash = model_hash
        self.target = target
        self.start_time = None
        self.monitor = get_compilation_monitor()
    
    def __enter__(self):
        self.start_time = self.monitor.record_compilation_start(self.model_hash, self.target)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.monitor.record_compilation_success(self.model_hash, self.target, self.start_time)
        else:
            self.monitor.record_compilation_failure(self.model_hash, self.target, self.start_time, exc_val)