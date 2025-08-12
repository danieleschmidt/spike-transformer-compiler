"""Logging and monitoring configuration for Spike-Transformer-Compiler."""

import logging
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class CompilationMetrics:
    """Metrics collected during compilation."""
    start_time: float
    end_time: Optional[float] = None
    frontend_parsing_time: float = 0.0
    optimization_time: float = 0.0
    backend_compilation_time: float = 0.0
    total_nodes: int = 0
    optimized_nodes: int = 0
    energy_estimate: float = 0.0
    memory_usage: int = 0
    target: str = ""
    model_type: str = ""
    input_shape: tuple = ()
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @property
    def total_time(self) -> float:
        """Get total compilation time."""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0


class CompilerLogger:
    """Enhanced logger for spike transformer compiler."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger("spike_transformer_compiler")
        self.metrics_logger = logging.getLogger("spike_transformer_compiler.metrics")
        self.performance_logger = logging.getLogger("spike_transformer_compiler.performance")
        
        # Set up logging if not already configured
        if not self.logger.handlers:
            self._setup_logging()
        
        self.current_metrics: Optional[CompilationMetrics] = None
        self.metrics_history: list = []
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Main logger
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Metrics logger (separate handler for structured logs)
        metrics_handler = logging.StreamHandler(sys.stderr)
        metrics_formatter = logging.Formatter(
            '%(asctime)s - METRICS - %(message)s'
        )
        metrics_handler.setFormatter(metrics_formatter)
        self.metrics_logger.addHandler(metrics_handler)
        self.metrics_logger.setLevel(logging.INFO)
        
        # Performance logger
        perf_handler = logging.StreamHandler(sys.stderr)
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
        self.performance_logger.setLevel(logging.INFO)
    
    def set_level(self, level: str):
        """Set logging level."""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
    
    def add_file_handler(self, log_file: Path, level: str = "INFO"):
        """Add file handler for persistent logging."""
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        file_handler.setLevel(level_map.get(level.upper(), logging.INFO))
        
        self.logger.addHandler(file_handler)
    
    def start_compilation(self, **context) -> CompilationMetrics:
        """Start tracking compilation metrics."""
        self.current_metrics = CompilationMetrics(
            start_time=time.time(),
            target=context.get('target', ''),
            model_type=context.get('model_type', ''),
            input_shape=context.get('input_shape', ())
        )
        
        self.logger.info(f"Starting compilation for {self.current_metrics.model_type} "
                        f"model with shape {self.current_metrics.input_shape}")
        
        return self.current_metrics
    
    def end_compilation(self, success: bool = True, error: Optional[str] = None):
        """End tracking compilation metrics."""
        if self.current_metrics:
            self.current_metrics.end_time = time.time()
            self.current_metrics.success = success
            self.current_metrics.error_message = error
            
            # Log completion
            if success:
                self.logger.info(f"Compilation completed successfully in "
                               f"{self.current_metrics.total_time:.2f}s")
            else:
                self.logger.error(f"Compilation failed after "
                                f"{self.current_metrics.total_time:.2f}s: {error}")
            
            # Log metrics
            self.metrics_logger.info(json.dumps(self.current_metrics.to_dict()))
            
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
            # Reset current metrics
            self.current_metrics = None
    
    @contextmanager
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.logger.debug(f"Starting {operation}")
        
        try:
            yield
            end_time = time.time()
            duration = end_time - start_time
            
            self.performance_logger.info(f"{operation}: {duration:.3f}s")
            
            # Update current metrics if available
            if self.current_metrics:
                if operation == "frontend_parsing":
                    self.current_metrics.frontend_parsing_time = duration
                elif operation == "optimization":
                    self.current_metrics.optimization_time = duration
                elif operation == "backend_compilation":
                    self.current_metrics.backend_compilation_time = duration
                    
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"{operation} failed after {duration:.3f}s: {str(e)}")
            raise
    
    def log_model_info(self, model, input_shape: tuple):
        """Log model information."""
        try:
            # Count parameters if PyTorch model
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            self.logger.info(f"Model type: {type(model).__name__}")
            self.logger.info(f"Input shape: {input_shape}")
            
        except Exception as e:
            self.logger.warning(f"Could not extract model info: {e}")
    
    def log_compilation_stage(self, stage: str, **info):
        """Log compilation stage information."""
        self.logger.info(f"Stage: {stage}")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_resource_usage(self, **resources):
        """Log resource usage information."""
        self.logger.info("Resource usage:")
        for resource, value in resources.items():
            self.logger.info(f"  {resource}: {value}")
    
    def log_optimization_results(self, before: dict, after: dict):
        """Log optimization pass results."""
        self.logger.info("Optimization results:")
        for metric in ['nodes', 'edges', 'memory_bytes']:
            if metric in before and metric in after:
                reduction = before[metric] - after[metric]
                percentage = (reduction / before[metric] * 100) if before[metric] > 0 else 0
                self.logger.info(f"  {metric}: {before[metric]} -> {after[metric]} "
                               f"(reduced {reduction}, {percentage:.1f}%)")
    
    def get_metrics_summary(self) -> dict:
        """Get summary of compilation metrics."""
        if not self.metrics_history:
            return {"message": "No compilation metrics available"}
        
        successful_compilations = [m for m in self.metrics_history if m.success]
        failed_compilations = [m for m in self.metrics_history if not m.success]
        
        summary = {
            "total_compilations": len(self.metrics_history),
            "successful_compilations": len(successful_compilations),
            "failed_compilations": len(failed_compilations),
            "success_rate": len(successful_compilations) / len(self.metrics_history) * 100,
        }
        
        if successful_compilations:
            times = [m.total_time for m in successful_compilations]
            summary.update({
                "avg_compilation_time": sum(times) / len(times),
                "min_compilation_time": min(times),
                "max_compilation_time": max(times),
            })
        
        return summary
    
    def export_metrics(self, output_file: Path):
        """Export metrics to file."""
        with open(output_file, 'w') as f:
            json.dump({
                "summary": self.get_metrics_summary(),
                "detailed_metrics": [m.to_dict() for m in self.metrics_history]
            }, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_file}")


class HealthMonitor:
    """Monitor system health during compilation."""
    
    def __init__(self):
        self.logger = CompilerLogger()
        self.start_memory = None
        self.peak_memory = 0
    
    def check_system_resources(self) -> dict:
        """Check available system resources."""
        import psutil
        
        resources = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # Log warnings for resource constraints
        if resources["memory_percent"] > 90:
            self.logger.logger.warning("High memory usage detected: {:.1f}%".format(
                resources["memory_percent"]))
        
        if resources["cpu_percent"] > 95:
            self.logger.logger.warning("High CPU usage detected: {:.1f}%".format(
                resources["cpu_percent"]))
        
        return resources
    
    def start_monitoring(self):
        """Start resource monitoring."""
        try:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / (1024**2)  # MB
            self.peak_memory = self.start_memory
        except ImportError:
            self.logger.logger.warning("psutil not available, resource monitoring disabled")
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**2)  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        except ImportError:
            pass
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        stats = {
            "start_memory_mb": self.start_memory or 0,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": (self.peak_memory - (self.start_memory or 0))
        }
        return stats


    def stop_monitoring(self):
        """Stop resource monitoring."""
        # Cleanup monitoring resources if needed
        pass
    
    def log_performance_metrics(self, perf_stats: dict):
        """Log performance metrics."""
        self.logger.logger.info("Performance metrics:")
        for metric, value in perf_stats.items():
            self.logger.logger.info(f"  {metric}: {value}")
    
    def log_resource_summary(self, resource_stats: dict):
        """Log resource summary."""
        self.logger.logger.info("Resource summary:")
        for resource, value in resource_stats.items():
            self.logger.logger.info(f"  {resource}: {value}")
    
    def get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().isoformat()


# Global logger instance
compiler_logger = CompilerLogger()