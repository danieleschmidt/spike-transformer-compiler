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
from contextlib import contextmanager

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
        
        # In-memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
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
        
        # ML-driven intelligent caching features
        self.access_predictor = CacheAccessPredictor()
        self.eviction_optimizer = IntelligentEvictionPolicy()
        self.prefetch_engine = PrefetchEngine()
        self.cache_analytics = CacheAnalytics()
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        compiler_logger.logger.info(f"Compilation cache initialized: {self.cache_dir}")


class CacheAccessPredictor:
    """ML-based predictor for cache access patterns."""
    
    def __init__(self, history_size: int = 10000):
        self.access_history = []
        self.history_size = history_size
        self.feature_weights = {
            'time_since_last_access': 0.3,
            'access_frequency': 0.4,
            'compilation_similarity': 0.3
        }
        
    def record_access(self, cache_key: str, compilation_params: Dict[str, Any]) -> None:
        """Record cache access for learning."""
        access_record = {
            'cache_key': cache_key,
            'timestamp': time.time(),
            'params': compilation_params
        }
        
        self.access_history.append(access_record)
        if len(self.access_history) > self.history_size:
            self.access_history.pop(0)
    
    def predict_access_probability(self, cache_key: str, current_params: Dict[str, Any]) -> float:
        """Predict probability of cache access."""
        if not self.access_history:
            return 0.5  # Default probability
        
        # Find similar compilation patterns
        similar_accesses = self._find_similar_compilations(current_params)
        
        if not similar_accesses:
            return 0.1  # Low probability for unseen patterns
        
        # Calculate features
        recent_accesses = len([a for a in similar_accesses if time.time() - a['timestamp'] < 3600])  # Last hour
        frequency_score = recent_accesses / len(similar_accesses)
        
        # Time decay
        latest_access = max(similar_accesses, key=lambda x: x['timestamp'])
        time_score = 1.0 / (1.0 + (time.time() - latest_access['timestamp']) / 3600)  # Hour-based decay
        
        # Weighted combination
        probability = frequency_score * 0.6 + time_score * 0.4
        return min(1.0, max(0.0, probability))
    
    def _find_similar_compilations(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar compilation patterns."""
        similar = []
        
        for record in self.access_history:
            similarity = self._calculate_similarity(params, record['params'])
            if similarity > 0.7:  # Threshold for similarity
                similar.append(record)
        
        return similar
    
    def _calculate_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between compilation parameters."""
        # Simple similarity based on key parameters
        key_params = ['optimization_level', 'target', 'input_shape']
        
        matches = 0
        total = len(key_params)
        
        for key in key_params:
            if params1.get(key) == params2.get(key):
                matches += 1
        
        return matches / total if total > 0 else 0.0


class IntelligentEvictionPolicy:
    """Intelligent cache eviction policy using multiple factors."""
    
    def __init__(self):
        self.eviction_weights = {
            'access_frequency': 0.4,
            'recency': 0.3,
            'size_cost': 0.2,
            'future_value': 0.1
        }
        
    def select_eviction_candidates(
        self, 
        cache_entries: Dict[str, CacheEntry],
        required_space: int
    ) -> List[str]:
        """Select cache entries for eviction."""
        if not cache_entries:
            return []
        
        # Score all entries
        scored_entries = []
        
        for cache_key, entry in cache_entries.items():
            score = self._calculate_eviction_score(entry)
            scored_entries.append((cache_key, score, entry.size_bytes))
        
        # Sort by eviction score (higher score = more likely to evict)
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Select entries to evict
        eviction_candidates = []
        freed_space = 0
        
        for cache_key, score, size_bytes in scored_entries:
            eviction_candidates.append(cache_key)
            freed_space += size_bytes
            
            if freed_space >= required_space:
                break
        
        return eviction_candidates
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score (higher = more likely to evict)."""
        current_time = time.time()
        
        # Recency score (older = higher eviction score)
        time_since_access = current_time - entry.timestamp
        recency_score = min(1.0, time_since_access / 3600)  # Normalize to 1 hour
        
        # Frequency score (less frequent = higher eviction score)
        frequency_score = 1.0 / max(1, entry.access_count)
        
        # Size cost score (larger = higher eviction score for same value)
        size_score = min(1.0, entry.size_bytes / (10 * 1024 * 1024))  # Normalize to 10MB
        
        # Future value score (expired = higher eviction score)
        future_score = 1.0 if entry.is_expired() else 0.2
        
        # Weighted combination
        total_score = (
            recency_score * self.eviction_weights['recency'] +
            frequency_score * self.eviction_weights['access_frequency'] +
            size_score * self.eviction_weights['size_cost'] +
            future_score * self.eviction_weights['future_value']
        )
        
        return total_score


class PrefetchEngine:
    """Intelligent prefetching engine for compilation artifacts."""
    
    def __init__(self, max_prefetch_threads: int = 2):
        self.max_prefetch_threads = max_prefetch_threads
        self.prefetch_executor = ThreadPoolExecutor(max_workers=max_prefetch_threads)
        self.prefetch_queue = []
        self.prefetch_history = {}
        
    def suggest_prefetch(self, current_params: Dict[str, Any]) -> List[str]:
        """Suggest cache keys to prefetch based on patterns."""
        suggestions = []
        
        # Pattern 1: Sequential model sizes
        if 'input_shape' in current_params:
            shape = current_params['input_shape']
            if isinstance(shape, tuple) and len(shape) >= 2:
                # Suggest similar shapes with different batch sizes
                for batch_size in [1, 4, 8, 16]:
                    if batch_size != shape[0]:
                        suggested_shape = (batch_size,) + shape[1:]
                        suggestions.append(self._generate_cache_key({
                            **current_params,
                            'input_shape': suggested_shape
                        }))
        
        # Pattern 2: Different optimization levels
        current_opt = current_params.get('optimization_level', 2)
        for opt_level in [0, 1, 2, 3]:
            if opt_level != current_opt:
                suggestions.append(self._generate_cache_key({
                    **current_params,
                    'optimization_level': opt_level
                }))
        
        return suggestions[:5]  # Limit suggestions
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        import hashlib
        key_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class CacheAnalytics:
    """Analytics and optimization for cache performance."""
    
    def __init__(self):
        self.performance_history = []
        self.optimization_recommendations = []
        
    def analyze_cache_performance(self, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cache performance and generate insights."""
        hit_rate = cache_stats['hits'] / max(1, cache_stats['hits'] + cache_stats['misses'])
        
        analysis = {
            'hit_rate': hit_rate,
            'performance_grade': self._calculate_performance_grade(hit_rate),
            'recommendations': []
        }
        
        # Generate recommendations
        if hit_rate < 0.6:
            analysis['recommendations'].append(
                "Consider increasing cache size or improving prefetching"
            )
        
        if cache_stats['evictions'] > cache_stats['hits'] * 0.5:
            analysis['recommendations'].append(
                "High eviction rate detected - consider cache size tuning"
            )
        
        return analysis
    
    def _calculate_performance_grade(self, hit_rate: float) -> str:
        """Calculate performance grade based on hit rate."""
        if hit_rate >= 0.9:
            return "A+"
        elif hit_rate >= 0.8:
            return "A"
        elif hit_rate >= 0.7:
            return "B"
        elif hit_rate >= 0.6:
            return "C"
        else:
            return "D"
    
    def get_optimization_suggestions(self, usage_patterns: Dict[str, Any]) -> List[str]:
        """Get optimization suggestions based on usage patterns."""
        suggestions = []
        
        # Analyze memory usage patterns
        if usage_patterns.get('memory_pressure', 0) > 0.8:
            suggestions.append("Implement more aggressive eviction policy")
            suggestions.append("Consider compression for large compilation artifacts")
        
        # Analyze access patterns
        access_variance = usage_patterns.get('access_pattern_variance', 0)
        if access_variance > 0.5:
            suggestions.append("Implement adaptive prefetching based on usage patterns")
        
        return suggestions
    
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


class PerformanceProfiler:
    """Profile performance during compilation."""
    
    def __init__(self):
        self.stage_times = {}
        self.start_times = {}
        self.compilation_start = None
        self.compilation_end = None
        
    def start_compilation_profiling(self):
        """Start profiling compilation."""
        self.compilation_start = time.time()
        self.stage_times.clear()
        self.start_times.clear()
        
    def end_compilation_profiling(self, failed: bool = False):
        """End compilation profiling."""
        self.compilation_end = time.time()
        if failed:
            compiler_logger.logger.warning("Compilation profiling ended due to failure")
        
    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling compilation stages."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.stage_times[stage_name] = end_time - start_time
            
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation performance statistics."""
        total_time = 0
        if self.compilation_start and self.compilation_end:
            total_time = self.compilation_end - self.compilation_start
            
        return {
            'total_compilation_time': total_time,
            'stage_times': self.stage_times.copy(),
            'throughput_nodes_per_sec': 0,  # Could be calculated if node count available
        }
        
    def cleanup(self):
        """Clean up profiler resources."""
        self.stage_times.clear()
        self.start_times.clear()


class ResourceMonitor:
    """Monitor system resources during compilation."""
    
    def __init__(self):
        self.memory_snapshots = []
        self.cpu_snapshots = []
        self.start_memory = None
        self.peak_memory = 0
        
    def log_memory_usage(self, label: str):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            self.memory_snapshots.append((label, memory_mb, time.time()))
            self.peak_memory = max(self.peak_memory, memory_mb)
            
            if self.start_memory is None:
                self.start_memory = memory_mb
                
        except ImportError:
            compiler_logger.logger.warning("psutil not available for memory monitoring")
            
    def log_compilation_failure(self, failure_type: str):
        """Log compilation failure for analysis."""
        compiler_logger.logger.error(f"Compilation failure: {failure_type}")
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.memory_snapshots:
            return {'memory_tracking': 'unavailable'}
            
        current_memory = self.memory_snapshots[-1][1] if self.memory_snapshots else 0
        memory_increase = current_memory - (self.start_memory or 0)
        
        return {
            'start_memory_mb': self.start_memory or 0,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': memory_increase,
            'memory_snapshots': len(self.memory_snapshots)
        }
        
    def cleanup(self):
        """Clean up monitoring resources."""
        self.memory_snapshots.clear()
        self.cpu_snapshots.clear()