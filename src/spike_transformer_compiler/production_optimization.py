"""Production-ready optimization features for maximum performance."""

import os
import sys
import time
import gc
import threading
import multiprocessing
import weakref
import pickle
import zlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import contextlib


class OptimizationProfile(Enum):
    """Optimization profiles for different use cases."""
    DEVELOPMENT = "development"     # Fast compilation, debug friendly
    TESTING = "testing"            # Balanced speed/verification
    PRODUCTION = "production"       # Maximum performance
    LOW_MEMORY = "low_memory"      # Memory-constrained environments
    HIGH_THROUGHPUT = "high_throughput"  # Batch processing optimized


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    profile: OptimizationProfile
    max_memory_mb: int = 1024
    max_cpu_cores: int = None
    enable_jit: bool = True
    enable_vectorization: bool = True
    enable_memory_mapping: bool = True
    cache_size_mb: int = 256
    batch_size: int = 32
    prefetch_threads: int = 2
    
    def __post_init__(self):
        if self.max_cpu_cores is None:
            self.max_cpu_cores = min(multiprocessing.cpu_count(), 8)


class MemoryManager:
    """Advanced memory management for large-scale compilation."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.allocated_objects = weakref.WeakSet()
        self.memory_pools = {}
        self.gc_threshold = max_memory_mb * 0.8  # Trigger GC at 80% memory
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
    
    def allocate_buffer(self, size_bytes: int, pool_name: str = "default") -> Optional[bytearray]:
        """Allocate memory buffer from pool."""
        with self._lock:
            if pool_name not in self.memory_pools:
                self.memory_pools[pool_name] = []
            
            # Check if we have available buffer in pool
            pool = self.memory_pools[pool_name]
            for buffer in pool:
                if len(buffer) >= size_bytes:
                    pool.remove(buffer)
                    return buffer[:size_bytes]
            
            # Check memory limits
            estimated_usage = self._estimate_memory_usage()
            if estimated_usage + size_bytes > self.max_memory_mb * 1024 * 1024:
                self._trigger_garbage_collection()
                
                # Check again after GC
                estimated_usage = self._estimate_memory_usage()
                if estimated_usage + size_bytes > self.max_memory_mb * 1024 * 1024:
                    self.logger.warning(f"Memory limit exceeded, requested {size_bytes} bytes")
                    return None
            
            # Allocate new buffer
            try:
                buffer = bytearray(size_bytes)
                self.allocated_objects.add(buffer)
                return buffer
            except MemoryError:
                self.logger.error(f"Failed to allocate {size_bytes} bytes")
                return None
    
    def deallocate_buffer(self, buffer: bytearray, pool_name: str = "default"):
        """Return buffer to pool for reuse."""
        with self._lock:
            if pool_name not in self.memory_pools:
                self.memory_pools[pool_name] = []
            
            # Reset buffer and add to pool
            buffer[:] = b'\0' * len(buffer)
            self.memory_pools[pool_name].append(buffer)
            
            # Limit pool size to prevent unbounded growth
            max_pool_size = 10
            if len(self.memory_pools[pool_name]) > max_pool_size:
                self.memory_pools[pool_name] = self.memory_pools[pool_name][-max_pool_size:]
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        # Simplified estimation - in production would use more accurate methods
        total_size = 0
        for pool in self.memory_pools.values():
            total_size += sum(len(buffer) for buffer in pool)
        
        # Add estimated overhead
        total_size += len(self.allocated_objects) * 1024  # Rough estimate
        return total_size
    
    def _trigger_garbage_collection(self):
        """Force garbage collection to free memory."""
        self.logger.debug("Triggering garbage collection")
        
        # Clear weak references to deleted objects
        self.allocated_objects = weakref.WeakSet()
        
        # Run Python garbage collection
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            pool_stats = {}
            for name, pool in self.memory_pools.items():
                pool_stats[name] = {
                    "buffers": len(pool),
                    "total_bytes": sum(len(buffer) for buffer in pool)
                }
            
            return {
                "max_memory_mb": self.max_memory_mb,
                "estimated_usage_bytes": self._estimate_memory_usage(),
                "allocated_objects": len(self.allocated_objects),
                "memory_pools": pool_stats,
                "gc_threshold_bytes": self.gc_threshold * 1024 * 1024
            }


class BatchProcessor:
    """Efficient batch processing for multiple compilation tasks."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 max_workers: int = None,
                 memory_manager: Optional[MemoryManager] = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.memory_manager = memory_manager or MemoryManager()
        
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "total_processing_time": 0.0,
            "average_batch_time": 0.0
        }
    
    def process_batch(self, items: List[Any], 
                     processor_func: Callable,
                     **kwargs) -> List[Any]:
        """Process a batch of items efficiently."""
        if not items:
            return []
        
        start_time = time.time()
        
        # Split into optimal batch sizes
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        results = []
        
        if len(batches) == 1:
            # Single batch - process directly
            results = self._process_single_batch(batches[0], processor_func, **kwargs)
        else:
            # Multiple batches - use parallel processing
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(self._process_single_batch, batch, processor_func, **kwargs)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    batch_results = future.result()
                    results.extend(batch_results)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_stats["batches_processed"] += len(batches)
        self.processing_stats["items_processed"] += len(items)
        self.processing_stats["total_processing_time"] += processing_time
        
        if self.processing_stats["batches_processed"] > 0:
            self.processing_stats["average_batch_time"] = (
                self.processing_stats["total_processing_time"] / self.processing_stats["batches_processed"]
            )
        
        self.logger.debug(f"Processed {len(items)} items in {len(batches)} batches ({processing_time:.3f}s)")
        
        return results
    
    def _process_single_batch(self, batch: List[Any], 
                            processor_func: Callable,
                            **kwargs) -> List[Any]:
        """Process a single batch of items."""
        results = []
        
        for item in batch:
            try:
                result = processor_func(item, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process item: {e}")
                results.append(e)  # Include error in results for handling upstream
        
        # Trigger memory cleanup if needed
        if hasattr(self.memory_manager, '_estimate_memory_usage'):
            current_usage = self.memory_manager._estimate_memory_usage()
            if current_usage > self.memory_manager.gc_threshold * 1024 * 1024:
                self.memory_manager._trigger_garbage_collection()
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return {
            **self.processing_stats,
            "throughput_items_per_second": (
                self.processing_stats["items_processed"] / max(1, self.processing_stats["total_processing_time"])
            ),
            "memory_stats": self.memory_manager.get_memory_stats()
        }


class CompressionManager:
    """Handle model compression and decompression for storage/transfer."""
    
    @staticmethod
    def compress_model(model: Any, compression_level: int = 6) -> bytes:
        """Compress a model for storage or transfer."""
        try:
            # Serialize model
            model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress using zlib
            compressed_bytes = zlib.compress(model_bytes, level=compression_level)
            
            return compressed_bytes
            
        except Exception as e:
            logging.error(f"Model compression failed: {e}")
            raise e
    
    @staticmethod
    def decompress_model(compressed_data: bytes) -> Any:
        """Decompress a model from compressed data."""
        try:
            # Decompress
            model_bytes = zlib.decompress(compressed_data)
            
            # Deserialize model
            model = pickle.loads(model_bytes)
            
            return model
            
        except Exception as e:
            logging.error(f"Model decompression failed: {e}")
            raise e
    
    @staticmethod
    def get_compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 0.0
        return compressed_size / original_size


class ProfiledCompiler:
    """Compiler with built-in performance profiling and optimization."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.max_memory_mb)
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size,
            max_workers=config.max_cpu_cores,
            memory_manager=self.memory_manager
        )
        
        self.logger = logging.getLogger(__name__)
        self.compilation_profiles = {}
        self.optimization_cache = {}
    
    def compile_with_profiling(self, 
                             model: Any,
                             input_shape: Tuple[int, ...],
                             target: str = "simulation",
                             **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Compile model with comprehensive profiling."""
        profile_start = time.time()
        
        # Generate profile key
        profile_key = f"{type(model).__name__}_{input_shape}_{target}"
        
        # Check if we have previous profile for this model type
        if profile_key in self.compilation_profiles:
            previous_profile = self.compilation_profiles[profile_key]
            self.logger.debug(f"Using profile data for {profile_key}")
        else:
            previous_profile = None
        
        # Prepare compilation environment based on profile
        compilation_context = self._prepare_compilation_context(previous_profile)
        
        try:
            with compilation_context:
                # Perform actual compilation
                from .compiler import SpikeCompiler
                
                compiler = SpikeCompiler(
                    target=target,
                    optimization_level=3 if self.config.profile == OptimizationProfile.PRODUCTION else 2,
                    verbose=False
                )
                
                # Profile memory usage during compilation
                pre_compilation_memory = self._get_memory_usage()
                
                compiled_model = compiler.compile(model, input_shape, **kwargs)
                
                post_compilation_memory = self._get_memory_usage()
        
        except Exception as e:
            self.logger.error(f"Compilation failed: {e}")
            raise e
        
        # Record profiling data
        compilation_time = time.time() - profile_start
        
        profile_data = {
            "compilation_time": compilation_time,
            "memory_delta": post_compilation_memory - pre_compilation_memory,
            "input_shape": input_shape,
            "target": target,
            "config_profile": self.config.profile.value,
            "timestamp": time.time()
        }
        
        # Update compilation profiles
        self.compilation_profiles[profile_key] = profile_data
        
        # Generate recommendations for future compilations
        recommendations = self._generate_optimization_recommendations(profile_data, previous_profile)
        
        return compiled_model, {
            "profile_data": profile_data,
            "recommendations": recommendations,
            "memory_stats": self.memory_manager.get_memory_stats()
        }
    
    def compile_batch(self, 
                     models_and_shapes: List[Tuple[Any, Tuple[int, ...]]],
                     target: str = "simulation",
                     **kwargs) -> List[Tuple[Any, Dict[str, Any]]]:
        """Compile multiple models efficiently in batch."""
        
        def compile_single(item):
            model, input_shape = item
            return self.compile_with_profiling(model, input_shape, target, **kwargs)
        
        return self.batch_processor.process_batch(
            models_and_shapes,
            compile_single
        )
    
    @contextlib.contextmanager
    def _prepare_compilation_context(self, previous_profile: Optional[Dict[str, Any]]):
        """Prepare optimized compilation context."""
        # Set thread affinity and priority based on profile
        original_priority = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, 'getpriority') else None
        
        try:
            # Optimize for production workloads
            if self.config.profile == OptimizationProfile.PRODUCTION:
                # Disable garbage collection during compilation for speed
                gc.disable()
                
                # Set higher process priority if possible
                if hasattr(os, 'setpriority'):
                    try:
                        os.setpriority(os.PRIO_PROCESS, 0, -1)  # Higher priority
                    except PermissionError:
                        pass  # Ignore if we don't have permission
            
            yield
            
        finally:
            # Restore original settings
            if self.config.profile == OptimizationProfile.PRODUCTION:
                gc.enable()
            
            if original_priority is not None and hasattr(os, 'setpriority'):
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, original_priority)
                except PermissionError:
                    pass
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        # Simplified memory usage estimation
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Convert KB to bytes
        except ImportError:
            # Fallback estimation
            return self.memory_manager._estimate_memory_usage()
    
    def _generate_optimization_recommendations(self, 
                                             current_profile: Dict[str, Any],
                                             previous_profile: Optional[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        # Memory usage recommendations
        memory_mb = current_profile["memory_delta"] / (1024 * 1024)
        if memory_mb > self.config.max_memory_mb * 0.8:
            recommendations.append(
                f"High memory usage detected ({memory_mb:.1f} MB). "
                f"Consider increasing memory limit or using LOW_MEMORY profile."
            )
        
        # Compilation time recommendations
        if current_profile["compilation_time"] > 10.0:  # More than 10 seconds
            recommendations.append(
                "Long compilation time detected. Consider using DEVELOPMENT profile for faster iterations."
            )
        
        # Profile comparison recommendations
        if previous_profile:
            time_delta = current_profile["compilation_time"] - previous_profile["compilation_time"]
            if time_delta > previous_profile["compilation_time"] * 0.2:  # 20% slower
                recommendations.append(
                    "Compilation time increased significantly compared to previous runs. "
                    "Check for model complexity changes or system resource availability."
                )
        
        # Configuration recommendations
        if self.config.profile == OptimizationProfile.DEVELOPMENT and current_profile["compilation_time"] < 1.0:
            recommendations.append(
                "Fast compilation detected. Consider using TESTING or PRODUCTION profile for better optimization."
            )
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "configuration": {
                "profile": self.config.profile.value,
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_cores": self.config.max_cpu_cores,
                "batch_size": self.config.batch_size
            },
            "compilation_profiles": len(self.compilation_profiles),
            "memory_manager": self.memory_manager.get_memory_stats(),
            "batch_processor": self.batch_processor.get_processing_stats(),
            "recent_compilations": list(self.compilation_profiles.values())[-5:]  # Last 5
        }


def create_optimized_compiler(profile: OptimizationProfile = OptimizationProfile.PRODUCTION,
                            max_memory_mb: int = 1024,
                            max_cpu_cores: int = None) -> ProfiledCompiler:
    """Factory function to create optimized compiler with best settings."""
    
    config = PerformanceConfig(
        profile=profile,
        max_memory_mb=max_memory_mb,
        max_cpu_cores=max_cpu_cores,
        enable_jit=profile != OptimizationProfile.DEVELOPMENT,
        enable_vectorization=True,
        enable_memory_mapping=profile == OptimizationProfile.PRODUCTION,
        cache_size_mb=256 if profile == OptimizationProfile.PRODUCTION else 128,
        batch_size=64 if profile == OptimizationProfile.HIGH_THROUGHPUT else 32,
        prefetch_threads=4 if profile == OptimizationProfile.HIGH_THROUGHPUT else 2
    )
    
    return ProfiledCompiler(config)


if __name__ == "__main__":
    # Demo production optimization features
    from .mock_models import create_test_model
    
    print("⚡ Production Optimization Demo")
    print("=" * 40)
    
    # Create optimized compiler
    compiler = create_optimized_compiler(
        profile=OptimizationProfile.PRODUCTION,
        max_memory_mb=512
    )
    
    # Test single compilation with profiling
    model = create_test_model("simple", input_size=100, output_size=50)
    
    print("Testing single compilation with profiling...")
    compiled_model, profile_info = compiler.compile_with_profiling(
        model=model,
        input_shape=(1, 100),
        target="simulation"
    )
    
    print(f"✓ Compilation completed in {profile_info['profile_data']['compilation_time']:.3f}s")
    print(f"  Memory usage: {profile_info['profile_data']['memory_delta'] / 1024:.1f} KB")
    
    if profile_info['recommendations']:
        print("📋 Recommendations:")
        for rec in profile_info['recommendations']:
            print(f"  • {rec}")
    
    # Test batch compilation
    print("\nTesting batch compilation...")
    models_and_shapes = [
        (create_test_model("simple", input_size=10 + i, output_size=5), (1, 10 + i))
        for i in range(3)
    ]
    
    batch_results = compiler.compile_batch(models_and_shapes, target="simulation")
    successful_compilations = sum(1 for result, _ in batch_results if not isinstance(result, Exception))
    
    print(f"✓ Batch compilation: {successful_compilations}/{len(batch_results)} successful")
    
    # Show performance summary
    summary = compiler.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    print(f"  Profile: {summary['configuration']['profile']}")
    print(f"  Memory limit: {summary['configuration']['max_memory_mb']} MB")
    print(f"  CPU cores: {summary['configuration']['max_cpu_cores']}")
    print(f"  Batch size: {summary['configuration']['batch_size']}")
    print(f"  Items processed: {summary['batch_processor']['items_processed']}")
    print(f"  Throughput: {summary['batch_processor']['throughput_items_per_second']:.1f} items/sec")