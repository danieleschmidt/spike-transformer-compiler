"""Performance benchmarks and stress tests.

This module provides comprehensive performance benchmarks and stress tests
for the Spike-Transformer-Compiler system to ensure scalability and
performance requirements are met.
"""

import pytest
import time
import threading
import multiprocessing
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import psutil
import json

# Import systems to benchmark
try:
    from spike_transformer_compiler.hyperscale_performance_engine import (
        HyperscalePerformanceEngine, AdaptiveCacheManager, AutoScalingManager
    )
    from spike_transformer_compiler.enhanced_resilience_system import (
        ResilienceOrchestrator, CircuitBreaker
    )
    from spike_transformer_compiler.comprehensive_security_system import (
        ComprehensiveSecuritySystem, InputValidator
    )
    from spike_transformer_compiler.research_orchestrator import ResearchOrchestrator
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def start_benchmark(self):
        """Start timing the benchmark."""
        self.start_time = time.time()
    
    def end_benchmark(self):
        """End timing the benchmark."""
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        """Get benchmark duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def add_result(self, operation: str, duration: float, success: bool = True, **metadata):
        """Add a result to the benchmark."""
        self.results.append({
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
            **metadata
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        if not self.results:
            return {"error": "No results available"}
        
        durations = [r["duration"] for r in self.results if r["success"]]
        success_count = sum(1 for r in self.results if r["success"])
        
        if not durations:
            return {"error": "No successful operations"}
        
        return {
            "benchmark_name": self.name,
            "total_operations": len(self.results),
            "successful_operations": success_count,
            "success_rate": success_count / len(self.results),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0,
            "operations_per_second": success_count / max(0.001, self.get_duration()),
            "total_benchmark_time": self.get_duration()
        }
    
    def print_results(self):
        """Print benchmark results."""
        stats = self.get_statistics()
        if "error" in stats:
            print(f"\n{self.name}: {stats['error']}")
            return
        
        print(f"\n{self.name} Results:")
        print(f"  Total Operations: {stats['total_operations']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Average Duration: {stats['avg_duration']:.3f}s")
        print(f"  Operations/Second: {stats['operations_per_second']:.1f}")
        print(f"  Min/Max Duration: {stats['min_duration']:.3f}s / {stats['max_duration']:.3f}s")
        print(f"  Total Time: {stats['total_benchmark_time']:.3f}s")


class TestCachePerformance:
    """Test adaptive cache performance under various conditions."""
    
    def test_cache_throughput(self):
        """Test cache throughput with high load."""
        benchmark = PerformanceBenchmark("Cache Throughput")
        cache = AdaptiveCacheManager(max_size=1000)
        
        benchmark.start_benchmark()
        
        # Populate cache
        for i in range(500):
            key = f"key_{i}"
            value = f"value_{i}" * 100  # Larger values
            
            start_time = time.time()
            cache.put(key, value)
            duration = time.time() - start_time
            benchmark.add_result("cache_put", duration)
        
        # Test cache hits
        for i in range(1000):
            key = f"key_{i % 500}"  # Ensure hits
            
            start_time = time.time()
            result = cache.get(key)
            duration = time.time() - start_time
            benchmark.add_result("cache_get", duration, success=(result is not None))
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] > 0.95  # At least 95% success rate
        assert stats["operations_per_second"] > 1000  # At least 1000 ops/sec
    
    def test_cache_concurrent_access(self):
        """Test cache performance with concurrent access."""
        benchmark = PerformanceBenchmark("Cache Concurrent Access")
        cache = AdaptiveCacheManager(max_size=1000)
        
        # Pre-populate cache
        for i in range(200):
            cache.put(f"key_{i}", f"value_{i}")
        
        def cache_worker(worker_id: int, num_operations: int):
            """Worker function for concurrent cache access."""
            results = []
            for i in range(num_operations):
                key = f"key_{(worker_id * 1000 + i) % 200}"
                
                start_time = time.time()
                result = cache.get(key)
                duration = time.time() - start_time
                
                results.append({
                    "operation": "concurrent_get",
                    "duration": duration,
                    "success": result is not None,
                    "worker_id": worker_id
                })
            return results
        
        benchmark.start_benchmark()
        
        # Run concurrent workers
        num_workers = 10
        operations_per_worker = 100
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(cache_worker, worker_id, operations_per_worker)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                worker_results = future.result()
                for result in worker_results:
                    benchmark.add_result(**result)
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] > 0.9  # At least 90% success rate under concurrency
        assert stats["operations_per_second"] > 500  # At least 500 ops/sec
    
    def test_cache_memory_efficiency(self):
        """Test cache memory efficiency and eviction performance."""
        benchmark = PerformanceBenchmark("Cache Memory Efficiency")
        cache = AdaptiveCacheManager(max_size=100)  # Small cache to trigger evictions
        
        benchmark.start_benchmark()
        
        # Fill cache beyond capacity to test eviction
        for i in range(500):  # 5x cache capacity
            key = f"key_{i}"
            value = f"value_{i}" * 50  # Moderately sized values
            
            start_time = time.time()
            cache.put(key, value)
            duration = time.time() - start_time
            benchmark.add_result("cache_put_with_eviction", duration)
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Verify cache size is maintained
        cache_stats = cache.get_stats()
        assert cache_stats["size"] <= 100  # Should not exceed max size
        
        # Check performance under eviction pressure
        stats = benchmark.get_statistics()
        assert stats["avg_duration"] < 0.01  # Should be fast even with evictions


class TestLoadBalancerPerformance:
    """Test load balancer performance and efficiency."""
    
    def test_worker_selection_performance(self):
        """Test worker selection performance under high load."""
        benchmark = PerformanceBenchmark("Load Balancer Worker Selection")
        
        # Create performance engine with load balancer
        engine = HyperscalePerformanceEngine()
        load_balancer = engine.load_balancer
        
        # Add more workers for testing
        for i in range(10):
            worker_id = f"test_worker_{i}"
            load_balancer.register_worker(worker_id, {
                "capacity": 100,
                "type": "compilation",
                "region": f"region_{i % 3}"
            })
        
        benchmark.start_benchmark()
        
        # Test worker selection performance
        for i in range(1000):
            start_time = time.time()
            selected_worker = load_balancer.select_worker({
                "request_size": i % 1000,
                "priority": "normal"
            })
            duration = time.time() - start_time
            
            benchmark.add_result(
                "worker_selection",
                duration,
                success=(selected_worker is not None)
            )
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] == 1.0  # Should always select a worker
        assert stats["avg_duration"] < 0.001  # Should be very fast (<1ms)
        assert stats["operations_per_second"] > 5000  # High throughput
    
    def test_load_balancer_with_worker_updates(self):
        """Test load balancer performance with frequent worker updates."""
        benchmark = PerformanceBenchmark("Load Balancer with Updates")
        
        engine = HyperscalePerformanceEngine()
        load_balancer = engine.load_balancer
        
        # Add workers
        for i in range(5):
            worker_id = f"update_worker_{i}"
            load_balancer.register_worker(worker_id, {"capacity": 100})
        
        benchmark.start_benchmark()
        
        # Simulate mixed load: selections and updates
        for i in range(500):
            # Worker selection
            start_time = time.time()
            selected_worker = load_balancer.select_worker()
            selection_duration = time.time() - start_time
            benchmark.add_result("selection_with_updates", selection_duration)
            
            # Worker metric update
            if selected_worker:
                start_time = time.time()
                load_balancer.update_worker_metrics(selected_worker, {
                    "response_time": 50 + (i % 100),
                    "error_occurred": i % 20 == 0,  # 5% error rate
                    "request_completed": True
                })
                update_duration = time.time() - start_time
                benchmark.add_result("metric_update", update_duration)
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] > 0.95
        assert stats["avg_duration"] < 0.005  # Should handle updates efficiently


class TestCompilationPerformance:
    """Test compilation performance optimization."""
    
    def test_compilation_optimization_throughput(self):
        """Test compilation optimization throughput."""
        benchmark = PerformanceBenchmark("Compilation Optimization")
        
        engine = HyperscalePerformanceEngine()
        
        # Test different compilation scenarios
        test_scenarios = [
            {"model_type": "spikeformer", "model_size": 500000, "complexity": "low"},
            {"model_type": "dsformer", "model_size": 1000000, "complexity": "medium"},
            {"model_type": "spikeformer", "model_size": 2000000, "complexity": "high"},
        ]
        
        benchmark.start_benchmark()
        
        for scenario in test_scenarios:
            for i in range(20):  # 20 iterations per scenario
                request = {
                    "model_type": scenario["model_type"],
                    "model_size": scenario["model_size"],
                    "optimization_level": 2,
                    "target": "simulation",
                    "request_id": f"{scenario['complexity']}_{i}"
                }
                
                start_time = time.time()
                result = engine.optimize_compilation(request)
                duration = time.time() - start_time
                
                benchmark.add_result(
                    f"compilation_{scenario['complexity']}",
                    duration,
                    success=result["final_result"]["success"],
                    model_size=scenario["model_size"],
                    complexity=scenario["complexity"]
                )
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] > 0.9
        assert stats["avg_duration"] < 5.0  # Should complete within 5 seconds
    
    def test_parallel_compilation_performance(self):
        """Test parallel compilation performance."""
        benchmark = PerformanceBenchmark("Parallel Compilation")
        
        engine = HyperscalePerformanceEngine()
        
        def compile_worker(worker_id: int, num_compilations: int):
            """Worker for parallel compilation testing."""
            results = []
            for i in range(num_compilations):
                request = {
                    "model_type": "spikeformer",
                    "model_size": 1000000,
                    "optimization_level": 2,
                    "target": "simulation",
                    "worker_id": worker_id,
                    "request_id": i
                }
                
                start_time = time.time()
                result = engine.optimize_compilation(request)
                duration = time.time() - start_time
                
                results.append({
                    "operation": "parallel_compilation",
                    "duration": duration,
                    "success": result["final_result"]["success"],
                    "worker_id": worker_id
                })
            return results
        
        benchmark.start_benchmark()
        
        # Run parallel compilations
        num_workers = 4
        compilations_per_worker = 5
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(compile_worker, worker_id, compilations_per_worker)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                worker_results = future.result()
                for result in worker_results:
                    benchmark.add_result(**result)
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] > 0.9
        # Parallel execution should provide better throughput
        assert stats["operations_per_second"] > 2  # At least 2 compilations/sec


class TestSecurityPerformance:
    """Test security system performance impact."""
    
    def test_input_validation_performance(self):
        """Test input validation performance."""
        benchmark = PerformanceBenchmark("Input Validation")
        
        validator = InputValidator()
        
        # Create test inputs of varying complexity
        test_inputs = [
            {"model_type": "simple", "params": {"layers": 12}},  # Simple
            {"model_type": "complex", "params": {"layers": 24, "hidden_size": 1024, 
             "attention_heads": 16, "vocab_size": 50000}},  # Complex
            {"model_type": "large", "data": ["item_" + str(i) for i in range(1000)]},  # Large
        ]
        
        benchmark.start_benchmark()
        
        # Test validation performance
        for test_input in test_inputs:
            for i in range(100):  # 100 iterations per input type
                start_time = time.time()
                is_valid, errors = validator.validate_model_input(test_input)
                duration = time.time() - start_time
                
                benchmark.add_result(
                    "input_validation",
                    duration,
                    success=True,  # Always succeeds (validation completes)
                    input_type=test_input["model_type"],
                    is_valid=is_valid
                )
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["avg_duration"] < 0.01  # Should be fast (<10ms)
        assert stats["operations_per_second"] > 1000  # High throughput
    
    def test_security_system_overhead(self):
        """Test overall security system performance overhead."""
        benchmark = PerformanceBenchmark("Security System Overhead")
        
        security_system = ComprehensiveSecuritySystem()
        security_system.enable_security()
        
        try:
            test_request = {
                "model_data": {
                    "model_type": "spikeformer",
                    "parameters": {"layers": 12, "hidden_size": 768}
                },
                "config": {
                    "target": "simulation",
                    "optimization_level": 2,
                    "secure_mode": True
                }
            }
            
            benchmark.start_benchmark()
            
            # Test security validation performance
            for i in range(100):
                start_time = time.time()
                allowed, result = security_system.secure_compilation_request(test_request)
                duration = time.time() - start_time
                
                benchmark.add_result(
                    "security_validation",
                    duration,
                    success=allowed,
                    threats_detected=len(result.get("threats_detected", [])),
                    validation_errors=len(result.get("validation_errors", []))
                )
            
            benchmark.end_benchmark()
            benchmark.print_results()
            
            # Assertions
            stats = benchmark.get_statistics()
            assert stats["success_rate"] > 0.95  # Most requests should be allowed
            assert stats["avg_duration"] < 0.1  # Security overhead should be reasonable
            
        finally:
            security_system.disable_security()


class TestResiliencePerformance:
    """Test resilience system performance impact."""
    
    def test_circuit_breaker_performance(self):
        """Test circuit breaker performance overhead."""
        benchmark = PerformanceBenchmark("Circuit Breaker")
        
        orchestrator = ResilienceOrchestrator()
        circuit_breaker = orchestrator.create_circuit_breaker("test_circuit")
        
        def fast_operation():
            return "success"
        
        benchmark.start_benchmark()
        
        # Test circuit breaker overhead
        for i in range(1000):
            start_time = time.time()
            result = circuit_breaker.call(fast_operation)
            duration = time.time() - start_time
            
            benchmark.add_result(
                "circuit_breaker_call",
                duration,
                success=(result == "success")
            )
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration"] < 0.001  # Very low overhead
        assert stats["operations_per_second"] > 5000
    
    def test_resilient_execution_performance(self):
        """Test resilient execution performance with retries."""
        benchmark = PerformanceBenchmark("Resilient Execution")
        
        orchestrator = ResilienceOrchestrator()
        
        # Function that fails occasionally
        failure_count = 0
        def unreliable_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count % 10 == 0:  # Fail every 10th call
                raise Exception("Simulated failure")
            return "success"
        
        benchmark.start_benchmark()
        
        # Test resilient execution
        for i in range(100):
            start_time = time.time()
            try:
                result = orchestrator.execute_with_resilience(
                    "test_operation",
                    unreliable_operation,
                    enable_retry=True,
                    enable_degradation=True
                )
                success = True
            except Exception:
                result = None
                success = False
            
            duration = time.time() - start_time
            benchmark.add_result(
                "resilient_execution",
                duration,
                success=success,
                result_type="success" if result == "success" else "degraded" if result else "failed"
            )
        
        benchmark.end_benchmark()
        benchmark.print_results()
        
        # Assertions
        stats = benchmark.get_statistics()
        assert stats["success_rate"] > 0.8  # Should handle most failures


class TestSystemStressTest:
    """Comprehensive system stress tests."""
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        benchmark = PerformanceBenchmark("Memory Usage Stress Test")
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = HyperscalePerformanceEngine()
        engine.start_performance_monitoring()
        
        try:
            benchmark.start_benchmark()
            
            # Sustained load test
            for batch in range(10):  # 10 batches
                batch_start = time.time()
                
                # Process batch of requests
                for i in range(20):  # 20 requests per batch
                    request = {
                        "model_type": "spikeformer",
                        "model_size": 1000000 + (i * 10000),
                        "optimization_level": 2,
                        "target": "simulation",
                        "batch_id": batch,
                        "request_id": i
                    }
                    
                    result = engine.optimize_compilation(request)
                    
                batch_duration = time.time() - batch_start
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                benchmark.add_result(
                    "batch_processing",
                    batch_duration,
                    success=True,
                    batch_id=batch,
                    memory_usage_mb=current_memory,
                    memory_growth_mb=current_memory - initial_memory
                )
                
                # Brief pause between batches
                time.sleep(0.1)
            
            benchmark.end_benchmark()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory
            
            benchmark.print_results()
            print(f"\nMemory Analysis:")
            print(f"  Initial Memory: {initial_memory:.1f} MB")
            print(f"  Final Memory: {final_memory:.1f} MB")
            print(f"  Memory Growth: {memory_growth:.1f} MB")
            
            # Assertions
            assert memory_growth < 100  # Should not grow more than 100MB
            
        finally:
            engine.stop_performance_monitoring()
    
    def test_concurrent_system_load(self):
        """Test system performance under concurrent load from multiple components."""
        benchmark = PerformanceBenchmark("Concurrent System Load")
        
        # Initialize all major systems
        performance_engine = HyperscalePerformanceEngine()
        security_system = ComprehensiveSecuritySystem()
        resilience_orchestrator = ResilienceOrchestrator()
        research_orchestrator = ResearchOrchestrator()
        
        security_system.enable_security()
        performance_engine.start_performance_monitoring()
        
        try:
            def performance_worker():
                """Worker that stresses performance engine."""
                results = []
                for i in range(10):
                    start_time = time.time()
                    request = {
                        "model_type": "spikeformer",
                        "model_size": 500000 + (i * 50000),
                        "optimization_level": 2,
                        "target": "simulation"
                    }
                    result = performance_engine.optimize_compilation(request)
                    duration = time.time() - start_time
                    results.append(("performance", duration, result["final_result"]["success"]))
                return results
            
            def security_worker():
                """Worker that stresses security system."""
                results = []
                for i in range(20):
                    start_time = time.time()
                    request = {
                        "model_data": {"model_type": "spikeformer", "params": {"size": i}},
                        "config": {"target": "simulation", "secure_mode": True}
                    }
                    allowed, _ = security_system.secure_compilation_request(request)
                    duration = time.time() - start_time
                    results.append(("security", duration, allowed))
                return results
            
            def resilience_worker():
                """Worker that stresses resilience system."""
                results = []
                for i in range(15):
                    start_time = time.time()
                    def test_op():
                        time.sleep(0.01)  # Simulate work
                        return "success"
                    
                    result = resilience_orchestrator.execute_with_resilience(
                        "test_op", test_op, enable_retry=True
                    )
                    duration = time.time() - start_time
                    results.append(("resilience", duration, result == "success"))
                return results
            
            benchmark.start_benchmark()
            
            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                
                # Submit multiple workers of each type
                for _ in range(2):  # 2 of each worker type
                    futures.append(executor.submit(performance_worker))
                    futures.append(executor.submit(security_worker))
                    futures.append(executor.submit(resilience_worker))
                
                # Collect all results
                for future in as_completed(futures):
                    worker_results = future.result()
                    for operation_type, duration, success in worker_results:
                        benchmark.add_result(
                            f"concurrent_{operation_type}",
                            duration,
                            success=success,
                            worker_type=operation_type
                        )
            
            benchmark.end_benchmark()
            benchmark.print_results()
            
            # Analyze results by operation type
            operation_types = set(r["operation"].split("_", 1)[1] for r in benchmark.results)
            for op_type in operation_types:
                type_results = [r for r in benchmark.results if op_type in r["operation"]]
                type_durations = [r["duration"] for r in type_results if r["success"]]
                if type_durations:
                    print(f"\n{op_type.capitalize()} under concurrent load:")
                    print(f"  Operations: {len(type_results)}")
                    print(f"  Avg Duration: {statistics.mean(type_durations):.3f}s")
                    print(f"  Success Rate: {sum(r['success'] for r in type_results) / len(type_results):.1%}")
            
            # Overall assertions
            stats = benchmark.get_statistics()
            assert stats["success_rate"] > 0.85  # System should remain stable under load
            
        finally:
            security_system.disable_security()
            performance_engine.stop_performance_monitoring()


def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("\n" + "="*60)
    print("SPIKE-TRANSFORMER-COMPILER PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # Cache Performance
    print("\n🗄️  CACHE PERFORMANCE TESTS")
    cache_tests = TestCachePerformance()
    cache_tests.test_cache_throughput()
    cache_tests.test_cache_concurrent_access()
    cache_tests.test_cache_memory_efficiency()
    
    # Load Balancer Performance
    print("\n⚖️  LOAD BALANCER PERFORMANCE TESTS")
    lb_tests = TestLoadBalancerPerformance()
    lb_tests.test_worker_selection_performance()
    lb_tests.test_load_balancer_with_worker_updates()
    
    # Compilation Performance
    print("\n🔄 COMPILATION PERFORMANCE TESTS")
    comp_tests = TestCompilationPerformance()
    comp_tests.test_compilation_optimization_throughput()
    comp_tests.test_parallel_compilation_performance()
    
    # Security Performance
    print("\n🔒 SECURITY PERFORMANCE TESTS")
    sec_tests = TestSecurityPerformance()
    sec_tests.test_input_validation_performance()
    sec_tests.test_security_system_overhead()
    
    # Resilience Performance
    print("\n🛡️  RESILIENCE PERFORMANCE TESTS")
    res_tests = TestResiliencePerformance()
    res_tests.test_circuit_breaker_performance()
    res_tests.test_resilient_execution_performance()
    
    # System Stress Tests
    print("\n🔥 SYSTEM STRESS TESTS")
    stress_tests = TestSystemStressTest()
    stress_tests.test_memory_usage_under_load()
    stress_tests.test_concurrent_system_load()
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run all benchmarks when executed directly
    run_all_benchmarks()
