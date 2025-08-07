"""Performance optimization and caching for Spike-Transformer-Compiler."""

import hashlib
import pickle
import json
import time
import threading
from typing import Any, Dict, Optional, Tuple, List, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import gc
from functools import wraps, lru_cache

from .config import get_compiler_config
from .logging_config import compiler_logger
from .exceptions import ConfigurationError


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def access(self):
        """Mark cache entry as accessed."""
        self.access_count += 1


class CompilationCache:
    """Intelligent caching system for compilation artifacts."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: float = 1000):
        self.config = get_compiler_config()
        self.cache_dir = cache_dir or Path(self.config.cache_directory).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.current_size_bytes = 0
        
        # In-memory cache for frequently accessed items
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        compiler_logger.logger.info(f"Compilation cache initialized: {self.cache_dir}")
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.current_size_bytes = metadata.get('total_size_bytes', 0)
                    self.stats.update(metadata.get('stats', {}))
            except Exception as e:
                compiler_logger.logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        metadata = {
            'total_size_bytes': self.current_size_bytes,
            'stats': self.stats,
            'last_updated': time.time()
        }
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            compiler_logger.logger.warning(f"Failed to save cache metadata: {e}")
    
    def _compute_cache_key(self, model_hash: str, input_shape: tuple, 
                          target: str, optimization_level: int, **kwargs) -> str:
        """Compute cache key from compilation parameters."""
        key_data = {
            'model_hash': model_hash,
            'input_shape': input_shape,
            'target': target,
            'optimization_level': optimization_level,
            'time_steps': kwargs.get('time_steps', 4),
            'compiler_version': '0.1.0'  # Should be dynamic
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _get_model_hash(self, model: Any) -> str:
        """Compute hash of model for caching."""
        try:
            # For PyTorch models, hash the state dict
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                # Convert to bytes for hashing
                model_bytes = pickle.dumps(state_dict)
                return hashlib.sha256(model_bytes).hexdigest()
            else:
                # Fallback: hash the string representation
                model_str = str(model)
                return hashlib.sha256(model_str.encode()).hexdigest()
        except Exception as e:
            compiler_logger.logger.warning(f"Failed to compute model hash: {e}")
            return "unknown"
    
    def get(self, model: Any, input_shape: tuple, target: str, 
            optimization_level: int, **kwargs) -> Optional[Any]:
        """Get cached compilation result."""
        if not self.config.compilation_cache_enabled:
            return None
        
        model_hash = self._get_model_hash(model)
        cache_key = self._compute_cache_key(
            model_hash, input_shape, target, optimization_level, **kwargs
        )
        
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not entry.is_expired():
                    entry.access()
                    self.stats['hits'] += 1
                    compiler_logger.logger.debug(f"Cache hit (memory): {cache_key[:16]}")
                    return entry.data
                else:
                    del self.memory_cache[cache_key]
            
            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        entry_data = pickle.load(f)
                    
                    entry = CacheEntry(**entry_data)
                    if not entry.is_expired():
                        entry.access()
                        
                        # Store in memory cache for faster future access
                        if len(self.memory_cache) < 100:  # Limit memory cache size
                            self.memory_cache[cache_key] = entry
                        
                        self.stats['hits'] += 1
                        self.stats['disk_reads'] += 1
                        compiler_logger.logger.debug(f"Cache hit (disk): {cache_key[:16]}")
                        return entry.data
                    else:
                        # Remove expired entry
                        cache_file.unlink()
                        
                except Exception as e:
                    compiler_logger.logger.warning(f"Failed to load cache entry: {e}")
            
            self.stats['misses'] += 1
            return None
    
    def put(self, model: Any, input_shape: tuple, target: str,
            optimization_level: int, compiled_model: Any, **kwargs) -> None:
        """Cache compilation result."""
        if not self.config.compilation_cache_enabled:
            return
        
        model_hash = self._get_model_hash(model)
        cache_key = self._compute_cache_key(
            model_hash, input_shape, target, optimization_level, **kwargs
        )
        
        try:
            # Estimate size of cached data
            data_bytes = pickle.dumps(compiled_model)
            data_size = len(data_bytes)
            
            # Check if we need to evict entries
            self._ensure_cache_space(data_size)
            
            entry = CacheEntry(
                data=compiled_model,
                timestamp=time.time(),
                size_bytes=data_size,
                ttl_seconds=3600 * 24  # 24 hour TTL
            )
            
            with self.cache_lock:
                # Store in memory cache
                if len(self.memory_cache) < 100:
                    self.memory_cache[cache_key] = entry
                
                # Store on disk
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                entry_data = {
                    'data': compiled_model,
                    'timestamp': entry.timestamp,
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes,
                    'ttl_seconds': entry.ttl_seconds
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry_data, f)
                
                self.current_size_bytes += data_size
                self.stats['disk_writes'] += 1
                
            compiler_logger.logger.debug(f"Cached compilation result: {cache_key[:16]}")
            self._save_cache_metadata()
            
        except Exception as e:
            compiler_logger.logger.warning(f"Failed to cache compilation result: {e}")
    
    def _ensure_cache_space(self, required_bytes: int):
        """Ensure sufficient cache space by evicting entries if needed."""
        if self.current_size_bytes + required_bytes <= self.max_size_bytes:
            return
        
        with self.cache_lock:
            # Get all cache files with their metadata
            cache_files = []
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.name == "cache_metadata.json":
                    continue
                
                try:
                    stat = cache_file.stat()
                    cache_files.append({
                        'file': cache_file,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'atime': stat.st_atime
                    })
                except Exception:
                    continue
            
            # Sort by access time (LRU)
            cache_files.sort(key=lambda x: x['atime'])
            
            # Remove files until we have enough space
            bytes_to_free = self.current_size_bytes + required_bytes - self.max_size_bytes
            bytes_freed = 0
            
            for file_info in cache_files:
                if bytes_freed >= bytes_to_free:
                    break
                
                try:
                    file_info['file'].unlink()
                    bytes_freed += file_info['size']
                    self.current_size_bytes -= file_info['size']
                    self.stats['evictions'] += 1
                    
                    # Remove from memory cache too
                    cache_key = file_info['file'].stem
                    if cache_key in self.memory_cache:
                        del self.memory_cache[cache_key]
                        
                except Exception as e:
                    compiler_logger.logger.warning(f"Failed to evict cache entry: {e}")
            
            compiler_logger.logger.info(f"Evicted {bytes_freed} bytes from cache")
    
    def clear(self):
        """Clear all cached entries."""
        with self.cache_lock:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            
            self.current_size_bytes = 0
            self.stats = {key: 0 for key in self.stats}
            
        compiler_logger.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_entries': len(self.memory_cache),
            'disk_size_mb': self.current_size_bytes / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        self.config = get_compiler_config()
        
    @lru_cache(maxsize=1000)
    def get_optimal_batch_size(self, model_complexity: int, available_memory_gb: float) -> int:
        """Calculate optimal batch size based on model complexity and available memory."""
        # Simple heuristic - more sophisticated models would use actual profiling
        base_batch_size = 32
        
        # Adjust based on model complexity
        if model_complexity > 1000000:  # Very large model
            base_batch_size = 8
        elif model_complexity > 100000:  # Large model
            base_batch_size = 16
        elif model_complexity < 10000:   # Small model
            base_batch_size = 64
        
        # Adjust based on available memory
        memory_factor = min(2.0, available_memory_gb / 4.0)
        optimal_batch_size = int(base_batch_size * memory_factor)
        
        return max(1, optimal_batch_size)
    
    def optimize_compilation_order(self, compilation_requests: List[Dict]) -> List[Dict]:
        """Optimize order of compilation requests for better cache utilization."""
        # Sort by similarity of parameters to improve cache hits
        def similarity_key(request):
            return (
                request.get('target', ''),
                request.get('optimization_level', 0),
                request.get('input_shape', tuple())
            )
        
        return sorted(compilation_requests, key=similarity_key)
    
    def estimate_compilation_time(self, model_complexity: int, target: str, 
                                optimization_level: int) -> float:
        """Estimate compilation time based on model characteristics."""
        # Base time estimates (in seconds)
        base_times = {
            'simulation': 5.0,
            'loihi3': 15.0
        }
        
        base_time = base_times.get(target, 10.0)
        
        # Scale by model complexity
        complexity_factor = max(0.1, model_complexity / 100000)
        
        # Scale by optimization level
        optimization_factors = {0: 0.5, 1: 1.0, 2: 2.0, 3: 4.0}
        optimization_factor = optimization_factors.get(optimization_level, 2.0)
        
        estimated_time = base_time * complexity_factor * optimization_factor
        
        return estimated_time


class ParallelCompilationManager:
    """Manage parallel and concurrent compilation tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.config = get_compiler_config()
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1))
        self.active_compilations: Dict[str, Any] = {}
        self.compilation_lock = threading.Lock()
        
    def compile_batch(self, compilation_requests: List[Dict], 
                     progress_callback: Optional[Callable] = None) -> List[Any]:
        """Compile multiple models in parallel."""
        if not self.config.enable_parallel_compilation:
            # Fall back to sequential compilation
            return self._compile_sequential(compilation_requests, progress_callback)
        
        optimizer = PerformanceOptimizer()
        optimized_requests = optimizer.optimize_compilation_order(compilation_requests)
        
        results = []
        completed = 0
        total = len(optimized_requests)
        
        # Use thread pool for I/O-bound compilation tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all compilation tasks
            future_to_request = {
                executor.submit(self._compile_single, request): request 
                for request in optimized_requests
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                
                try:
                    result = future.result()
                    results.append({
                        'request': request,
                        'result': result,
                        'success': True,
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'request': request,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        return results
    
    def _compile_sequential(self, requests: List[Dict], 
                           progress_callback: Optional[Callable] = None) -> List[Any]:
        """Compile models sequentially as fallback."""
        results = []
        for i, request in enumerate(requests):
            try:
                result = self._compile_single(request)
                results.append({
                    'request': request,
                    'result': result,
                    'success': True,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'request': request,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
            
            if progress_callback:
                progress_callback(i + 1, len(requests))
        
        return results
    
    def _compile_single(self, request: Dict) -> Any:
        """Compile a single model."""
        from .compiler import SpikeCompiler
        
        compiler = SpikeCompiler(
            target=request.get('target', 'simulation'),
            optimization_level=request.get('optimization_level', 2),
            time_steps=request.get('time_steps', 4),
            verbose=False  # Reduce noise in parallel execution
        )
        
        return compiler.compile(
            model=request['model'],
            input_shape=request['input_shape'],
            **request.get('compile_kwargs', {})
        )


class MemoryProfiler:
    """Profile and optimize memory usage during compilation."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_snapshots = []
        
    def start_profiling(self):
        """Start memory profiling."""
        self.peak_memory = 0
        self.current_memory = self._get_current_memory()
        self.memory_snapshots = [('start', self.current_memory)]
        
    def snapshot(self, label: str):
        """Take memory snapshot."""
        current = self._get_current_memory()
        self.memory_snapshots.append((label, current))
        self.peak_memory = max(self.peak_memory, current)
        
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_report(self) -> Dict[str, Any]:
        """Get memory profiling report."""
        return {
            'peak_memory_mb': self.peak_memory,
            'snapshots': self.memory_snapshots,
            'memory_increase_mb': self.memory_snapshots[-1][1] - self.memory_snapshots[0][1] if self.memory_snapshots else 0
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on profiling."""
        suggestions = []
        
        if self.peak_memory > 4000:  # > 4GB
            suggestions.append("Consider reducing model size or using gradient checkpointing")
        
        memory_increase = self.memory_snapshots[-1][1] - self.memory_snapshots[0][1] if len(self.memory_snapshots) > 1 else 0
        if memory_increase > 2000:  # > 2GB increase
            suggestions.append("High memory usage detected - consider memory optimization passes")
        
        return suggestions


# Global cache instance
_compilation_cache: Optional[CompilationCache] = None


def get_compilation_cache() -> CompilationCache:
    """Get global compilation cache."""
    global _compilation_cache
    if _compilation_cache is None:
        _compilation_cache = CompilationCache()
    return _compilation_cache


# Decorators for performance optimization
def cached_compilation(func):
    """Decorator to add caching to compilation functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache = get_compilation_cache()
        
        # Extract relevant parameters for cache key
        if len(args) >= 3:  # Assuming (self, model, input_shape, ...)
            model, input_shape = args[1], args[2]
            target = kwargs.get('target') or (args[0].target if hasattr(args[0], 'target') else 'simulation')
            optimization_level = kwargs.get('optimization_level') or getattr(args[0], 'optimization_level', 2)
            
            # Try to get from cache
            cached_result = cache.get(model, input_shape, target, optimization_level, **kwargs)
            if cached_result is not None:
                return cached_result
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Store in cache
        if len(args) >= 3:
            cache.put(model, input_shape, target, optimization_level, result, **kwargs)
        
        return result
    
    return wrapper


def memory_optimized(func):
    """Decorator to optimize memory usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before execution
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Clean up after execution
            gc.collect()
    
    return wrapper