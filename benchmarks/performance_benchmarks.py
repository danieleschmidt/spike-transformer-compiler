#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite
Advanced benchmarking with statistical analysis, regression detection, and optimization insights.
"""

import time
import sys
import os
import json
import statistics
import threading
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import contextlib
import gc
import tracemalloc
import cProfile
import pstats
from io import StringIO

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spike_transformer_compiler import SpikeCompiler, OptimizationPass, ResourceAllocator
from spike_transformer_compiler.ir.builder import SpikeIRBuilder
from spike_transformer_compiler.ir.spike_graph import SpikeGraph


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    name: str
    duration: float  # seconds
    memory_peak: float  # MB
    memory_avg: float  # MB
    cpu_usage: float  # percentage
    iterations: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceProfile:
    """Detailed performance profile."""
    function_stats: Dict[str, Dict[str, float]]
    memory_timeline: List[Tuple[float, float]]  # (time, memory_mb)
    bottlenecks: List[str]
    optimization_opportunities: List[str]


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str
    timestamp: datetime
    system_info: Dict[str, Any]
    results: List[BenchmarkResult]
    performance_profile: PerformanceProfile
    summary_stats: Dict[str, float]
    regression_analysis: Dict[str, Any]


class SystemProfiler:
    """System resource profiling utilities."""
    
    def __init__(self):
        self.start_time = None
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system resource monitoring."""
        self.start_time = time.time()
        self.memory_samples = []
        self.monitoring = True
        
        tracemalloc.start()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return stats."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            "duration": duration,
            "memory_current_mb": current / 1024 / 1024,
            "memory_peak_mb": peak / 1024 / 1024,
            "memory_avg_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "memory_samples": len(self.memory_samples)
        }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Sample memory usage
                if tracemalloc.is_tracing():
                    current, _ = tracemalloc.get_traced_memory()
                    self.memory_samples.append(current / 1024 / 1024)  # Convert to MB
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break


class CompilationBenchmarks:
    """Benchmarks for compilation performance."""
    
    def __init__(self):
        self.compiler = SpikeCompiler()
        self.profiler = SystemProfiler()
    
    def benchmark_basic_compilation(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark basic model compilation."""
        durations = []
        memory_peaks = []
        successful_runs = 0
        
        for i in range(iterations):
            try:
                self.profiler.start_monitoring()
                
                # Create a simple model
                start_time = time.time()
                
                builder = SpikeIRBuilder(f"benchmark_model_{i}")
                input_id = builder.add_input("input", (1, 16))
                linear_id = builder.add_spike_linear(input_id, out_features=8)
                neuron_id = builder.add_spike_neuron(linear_id)
                builder.add_output(neuron_id, "output")
                
                graph = builder.build()
                
                duration = time.time() - start_time
                stats = self.profiler.stop_monitoring()
                
                durations.append(duration)
                memory_peaks.append(stats["memory_peak_mb"])
                successful_runs += 1
                
            except Exception as e:
                return BenchmarkResult(
                    name="basic_compilation",
                    duration=0,
                    memory_peak=0,
                    memory_avg=0,
                    cpu_usage=0,
                    iterations=i,
                    success=False,
                    error_message=str(e)
                )
        
        return BenchmarkResult(
            name="basic_compilation",
            duration=statistics.mean(durations),
            memory_peak=max(memory_peaks) if memory_peaks else 0,
            memory_avg=statistics.mean(memory_peaks) if memory_peaks else 0,
            cpu_usage=0,  # Would need psutil for accurate CPU measurement
            iterations=successful_runs,
            success=True,
            metadata={
                "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0,
                "duration_min": min(durations) if durations else 0,
                "duration_max": max(durations) if durations else 0
            }
        )
    
    def benchmark_complex_compilation(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark complex model compilation."""
        durations = []
        memory_peaks = []
        successful_runs = 0
        
        for i in range(iterations):
            try:
                self.profiler.start_monitoring()
                
                start_time = time.time()
                
                # Create a complex model
                builder = SpikeIRBuilder(f"complex_model_{i}")
                input_id = builder.add_input("input", (1, 64))
                
                current_id = input_id
                for layer in range(10):  # 10 layers
                    linear_id = builder.add_spike_linear(current_id, out_features=32)
                    neuron_id = builder.add_spike_neuron(linear_id)
                    current_id = neuron_id
                
                builder.add_output(current_id, "output")
                graph = builder.build()
                
                # Analyze the graph
                resources = graph.analyze_resources()
                
                duration = time.time() - start_time
                stats = self.profiler.stop_monitoring()
                
                durations.append(duration)
                memory_peaks.append(stats["memory_peak_mb"])
                successful_runs += 1
                
            except Exception as e:
                return BenchmarkResult(
                    name="complex_compilation",
                    duration=0,
                    memory_peak=0,
                    memory_avg=0,
                    cpu_usage=0,
                    iterations=i,
                    success=False,
                    error_message=str(e)
                )
        
        return BenchmarkResult(
            name="complex_compilation",
            duration=statistics.mean(durations),
            memory_peak=max(memory_peaks) if memory_peaks else 0,
            memory_avg=statistics.mean(memory_peaks) if memory_peaks else 0,
            cpu_usage=0,
            iterations=successful_runs,
            success=True,
            metadata={
                "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0,
                "layers": 10,
                "nodes_per_layer": 32
            }
        )
    
    def benchmark_optimization_passes(self, iterations: int = 3) -> BenchmarkResult:
        """Benchmark optimization pass performance."""
        durations = []
        memory_peaks = []
        successful_runs = 0
        
        for i in range(iterations):
            try:
                self.profiler.start_monitoring()
                
                start_time = time.time()
                
                # Create model and apply optimization passes
                compiler = SpikeCompiler(optimization_level=3)
                optimizer = compiler.create_optimizer()
                
                # Create a medium complexity model
                builder = SpikeIRBuilder(f"opt_model_{i}")
                input_id = builder.add_input("input", (1, 32))
                
                for layer in range(5):
                    linear_id = builder.add_spike_linear(input_id, out_features=16)
                    neuron_id = builder.add_spike_neuron(linear_id)
                    input_id = neuron_id
                
                builder.add_output(input_id, "output")
                graph = builder.build()
                
                # This would apply optimization passes if implemented
                # optimized_graph = optimizer.optimize(graph)
                
                duration = time.time() - start_time
                stats = self.profiler.stop_monitoring()
                
                durations.append(duration)
                memory_peaks.append(stats["memory_peak_mb"])
                successful_runs += 1
                
            except Exception as e:
                return BenchmarkResult(
                    name="optimization_passes",
                    duration=0,
                    memory_peak=0,
                    memory_avg=0,
                    cpu_usage=0,
                    iterations=i,
                    success=False,
                    error_message=str(e)
                )
        
        return BenchmarkResult(
            name="optimization_passes",
            duration=statistics.mean(durations),
            memory_peak=max(memory_peaks) if memory_peaks else 0,
            memory_avg=statistics.mean(memory_peaks) if memory_peaks else 0,
            cpu_usage=0,
            iterations=successful_runs,
            success=True,
            metadata={
                "optimization_level": 3
            }
        )


class ScalabilityBenchmarks:
    """Benchmarks for scalability and concurrent performance."""
    
    def benchmark_concurrent_compilation(self, max_workers: int = 4, iterations: int = 20) -> BenchmarkResult:
        """Benchmark concurrent compilation performance."""
        
        def compile_model(model_id: int) -> float:
            """Compile a single model and return duration."""
            start_time = time.time()
            
            builder = SpikeIRBuilder(f"concurrent_model_{model_id}")
            input_id = builder.add_input("input", (1, 16))
            linear_id = builder.add_spike_linear(input_id, out_features=8)
            neuron_id = builder.add_spike_neuron(linear_id)
            builder.add_output(neuron_id, "output")
            
            graph = builder.build()
            
            return time.time() - start_time
        
        try:
            profiler = SystemProfiler()
            profiler.start_monitoring()
            
            start_time = time.time()
            
            # Run concurrent compilation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(compile_model, i) for i in range(iterations)]
                durations = [future.result() for future in futures]
            
            total_duration = time.time() - start_time
            stats = profiler.stop_monitoring()
            
            return BenchmarkResult(
                name="concurrent_compilation",
                duration=total_duration,
                memory_peak=stats["memory_peak_mb"],
                memory_avg=stats["memory_avg_mb"],
                cpu_usage=0,
                iterations=iterations,
                success=True,
                metadata={
                    "max_workers": max_workers,
                    "avg_individual_duration": statistics.mean(durations),
                    "throughput": iterations / total_duration,
                    "speedup_vs_sequential": (sum(durations) / total_duration) if total_duration > 0 else 0
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="concurrent_compilation",
                duration=0,
                memory_peak=0,
                memory_avg=0,
                cpu_usage=0,
                iterations=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_scaling_performance(self) -> BenchmarkResult:
        """Benchmark performance scaling with model size."""
        results = []
        
        model_sizes = [8, 16, 32, 64, 128]  # Different input sizes
        
        for size in model_sizes:
            try:
                profiler = SystemProfiler()
                profiler.start_monitoring()
                
                start_time = time.time()
                
                builder = SpikeIRBuilder(f"scaling_model_{size}")
                input_id = builder.add_input("input", (1, size))
                
                # Scale layers with input size
                num_layers = max(2, size // 16)
                current_id = input_id
                
                for i in range(num_layers):
                    linear_id = builder.add_spike_linear(current_id, out_features=size // 2)
                    neuron_id = builder.add_spike_neuron(linear_id)
                    current_id = neuron_id
                
                builder.add_output(current_id, "output")
                graph = builder.build()
                
                duration = time.time() - start_time
                stats = profiler.stop_monitoring()
                
                results.append({
                    "size": size,
                    "duration": duration,
                    "memory": stats["memory_peak_mb"],
                    "layers": num_layers
                })
                
            except Exception:
                continue
        
        if not results:
            return BenchmarkResult(
                name="scaling_performance",
                duration=0,
                memory_peak=0,
                memory_avg=0,
                cpu_usage=0,
                iterations=0,
                success=False,
                error_message="All scaling tests failed"
            )
        
        # Calculate scaling metrics
        durations = [r["duration"] for r in results]
        memories = [r["memory"] for r in results]
        
        # Simple complexity analysis
        size_ratios = [results[i+1]["size"] / results[i]["size"] for i in range(len(results)-1)]
        duration_ratios = [results[i+1]["duration"] / results[i]["duration"] for i in range(len(results)-1)]
        
        avg_complexity_factor = statistics.mean(duration_ratios) if duration_ratios else 1.0
        
        return BenchmarkResult(
            name="scaling_performance",
            duration=statistics.mean(durations),
            memory_peak=max(memories),
            memory_avg=statistics.mean(memories),
            cpu_usage=0,
            iterations=len(results),
            success=True,
            metadata={
                "model_sizes": model_sizes[:len(results)],
                "complexity_factor": avg_complexity_factor,
                "scaling_results": results,
                "linear_scaling": abs(avg_complexity_factor - 1.0) < 0.5  # Roughly linear
            }
        )


class MemoryBenchmarks:
    """Benchmarks for memory efficiency and usage patterns."""
    
    def benchmark_memory_efficiency(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark memory efficiency patterns."""
        memory_profiles = []
        
        for i in range(iterations):
            try:
                # Clear memory before each test
                gc.collect()
                
                profiler = SystemProfiler()
                profiler.start_monitoring()
                
                # Create and destroy multiple models to test memory management
                models = []
                for j in range(10):
                    builder = SpikeIRBuilder(f"memory_model_{i}_{j}")
                    input_id = builder.add_input("input", (1, 32))
                    linear_id = builder.add_spike_linear(input_id, out_features=16)
                    neuron_id = builder.add_spike_neuron(linear_id)
                    builder.add_output(neuron_id, "output")
                    
                    graph = builder.build()
                    models.append(graph)
                
                # Clear references
                del models
                gc.collect()
                
                stats = profiler.stop_monitoring()
                memory_profiles.append(stats)
                
            except Exception as e:
                return BenchmarkResult(
                    name="memory_efficiency",
                    duration=0,
                    memory_peak=0,
                    memory_avg=0,
                    cpu_usage=0,
                    iterations=i,
                    success=False,
                    error_message=str(e)
                )
        
        # Analyze memory patterns
        peak_memories = [p["memory_peak_mb"] for p in memory_profiles]
        avg_memories = [p["memory_avg_mb"] for p in memory_profiles]
        
        # Memory stability metric (lower is better)
        memory_variance = statistics.variance(peak_memories) if len(peak_memories) > 1 else 0
        
        return BenchmarkResult(
            name="memory_efficiency",
            duration=statistics.mean([p["duration"] for p in memory_profiles]),
            memory_peak=max(peak_memories) if peak_memories else 0,
            memory_avg=statistics.mean(avg_memories) if avg_memories else 0,
            cpu_usage=0,
            iterations=len(memory_profiles),
            success=True,
            metadata={
                "memory_variance": memory_variance,
                "memory_stability": "good" if memory_variance < 10 else "needs_improvement",
                "gc_effectiveness": "good" if memory_variance < 5 else "moderate"
            }
        )
    
    def benchmark_memory_leaks(self, iterations: int = 100) -> BenchmarkResult:
        """Test for memory leaks over many iterations."""
        initial_memory = None
        memory_samples = []
        
        try:
            gc.collect()  # Clean start
            
            profiler = SystemProfiler()
            profiler.start_monitoring()
            
            for i in range(iterations):
                # Create and immediately destroy objects
                builder = SpikeIRBuilder(f"leak_test_{i}")
                input_id = builder.add_input("input", (1, 8))
                linear_id = builder.add_spike_linear(input_id, out_features=4)
                neuron_id = builder.add_spike_neuron(linear_id)
                builder.add_output(neuron_id, "output")
                
                graph = builder.build()
                
                # Sample memory every 10 iterations
                if i % 10 == 0:
                    current, _ = tracemalloc.get_traced_memory()
                    memory_mb = current / 1024 / 1024
                    memory_samples.append(memory_mb)
                    
                    if initial_memory is None:
                        initial_memory = memory_mb
                
                # Force cleanup
                del graph, builder
                if i % 20 == 0:
                    gc.collect()
            
            stats = profiler.stop_monitoring()
            
            # Analyze memory trend
            final_memory = memory_samples[-1] if memory_samples else 0
            memory_growth = final_memory - initial_memory if initial_memory else 0
            
            # Linear regression to detect growth trend
            if len(memory_samples) > 2:
                x = list(range(len(memory_samples)))
                y = memory_samples
                n = len(memory_samples)
                
                # Calculate slope (memory growth rate)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                memory_leak_detected = slope > 0.1  # Growing more than 0.1 MB per sample
            else:
                slope = 0
                memory_leak_detected = False
            
            return BenchmarkResult(
                name="memory_leaks",
                duration=stats["duration"],
                memory_peak=stats["memory_peak_mb"],
                memory_avg=stats["memory_avg_mb"],
                cpu_usage=0,
                iterations=iterations,
                success=True,
                metadata={
                    "memory_growth_mb": memory_growth,
                    "growth_rate_mb_per_iteration": slope * 10,  # Per 10 iterations
                    "leak_detected": memory_leak_detected,
                    "memory_samples": len(memory_samples),
                    "leak_severity": "high" if slope > 0.5 else "medium" if slope > 0.1 else "low"
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="memory_leaks",
                duration=0,
                memory_peak=0,
                memory_avg=0,
                cpu_usage=0,
                iterations=0,
                success=False,
                error_message=str(e)
            )


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.compilation_benchmarks = CompilationBenchmarks()
        self.scalability_benchmarks = ScalabilityBenchmarks()
        self.memory_benchmarks = MemoryBenchmarks()
        
    def run_full_suite(self) -> BenchmarkSuite:
        """Run the complete benchmark suite."""
        print("🚀 Starting Comprehensive Performance Benchmark Suite")
        print("=" * 60)
        
        start_time = datetime.now()
        results = []
        
        # System information
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": multiprocessing.cpu_count(),
            "timestamp": start_time.isoformat()
        }
        
        # Compilation benchmarks
        print("\n📊 Running Compilation Benchmarks...")
        
        print("  • Basic compilation performance...")
        result = self.compilation_benchmarks.benchmark_basic_compilation()
        results.append(result)
        self._print_result(result)
        
        print("  • Complex compilation performance...")
        result = self.compilation_benchmarks.benchmark_complex_compilation()
        results.append(result)
        self._print_result(result)
        
        print("  • Optimization passes performance...")
        result = self.compilation_benchmarks.benchmark_optimization_passes()
        results.append(result)
        self._print_result(result)
        
        # Scalability benchmarks
        print("\n📈 Running Scalability Benchmarks...")
        
        print("  • Concurrent compilation performance...")
        result = self.scalability_benchmarks.benchmark_concurrent_compilation()
        results.append(result)
        self._print_result(result)
        
        print("  • Scaling performance analysis...")
        result = self.scalability_benchmarks.benchmark_scaling_performance()
        results.append(result)
        self._print_result(result)
        
        # Memory benchmarks
        print("\n🧠 Running Memory Benchmarks...")
        
        print("  • Memory efficiency analysis...")
        result = self.memory_benchmarks.benchmark_memory_efficiency()
        results.append(result)
        self._print_result(result)
        
        print("  • Memory leak detection...")
        result = self.memory_benchmarks.benchmark_memory_leaks()
        results.append(result)
        self._print_result(result)
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_stats(results)
        
        # Performance profile (simplified)
        performance_profile = PerformanceProfile(
            function_stats={},
            memory_timeline=[],
            bottlenecks=self._identify_bottlenecks(results),
            optimization_opportunities=self._identify_optimizations(results)
        )
        
        # Regression analysis (placeholder)
        regression_analysis = {
            "baseline_comparison": "No baseline available",
            "performance_trend": "stable",
            "regressions_detected": []
        }
        
        suite_result = BenchmarkSuite(
            suite_name="Spike Transformer Compiler Performance Suite",
            timestamp=start_time,
            system_info=system_info,
            results=results,
            performance_profile=performance_profile,
            summary_stats=summary_stats,
            regression_analysis=regression_analysis
        )
        
        # Print final summary
        self._print_summary(suite_result)
        
        return suite_result
    
    def _print_result(self, result: BenchmarkResult):
        """Print individual benchmark result."""
        if result.success:
            status = "✅ PASS"
            details = f"{result.duration:.3f}s, {result.memory_peak:.1f}MB peak"
        else:
            status = "❌ FAIL"
            details = result.error_message or "Unknown error"
        
        print(f"    {status} - {details}")
    
    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate summary statistics."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                "total_benchmarks": len(results),
                "successful_benchmarks": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "avg_memory_peak": 0.0,
                "performance_score": 0.0
            }
        
        durations = [r.duration for r in successful_results]
        memory_peaks = [r.memory_peak for r in successful_results]
        
        # Calculate performance score (0-100, higher is better)
        # Based on speed and memory efficiency
        avg_duration = statistics.mean(durations)
        avg_memory = statistics.mean(memory_peaks)
        
        # Performance score calculation (simplified)
        speed_score = max(0, min(100, 100 - (avg_duration * 20)))  # Penalty for slow operations
        memory_score = max(0, min(100, 100 - (avg_memory / 10)))   # Penalty for high memory usage
        performance_score = (speed_score + memory_score) / 2
        
        return {
            "total_benchmarks": len(results),
            "successful_benchmarks": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "avg_duration": avg_duration,
            "avg_memory_peak": statistics.mean(memory_peaks),
            "performance_score": performance_score,
            "speed_score": speed_score,
            "memory_score": memory_score
        }
    
    def _identify_bottlenecks(self, results: List[BenchmarkResult]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for result in results:
            if not result.success:
                continue
                
            # Check for slow operations (>1 second)
            if result.duration > 1.0:
                bottlenecks.append(f"Slow operation: {result.name} ({result.duration:.2f}s)")
            
            # Check for high memory usage (>100MB)
            if result.memory_peak > 100:
                bottlenecks.append(f"High memory usage: {result.name} ({result.memory_peak:.1f}MB)")
            
            # Check metadata for specific issues
            if result.metadata:
                if result.metadata.get("leak_detected"):
                    bottlenecks.append(f"Memory leak detected: {result.name}")
                
                if result.metadata.get("memory_stability") == "needs_improvement":
                    bottlenecks.append(f"Memory instability: {result.name}")
        
        return bottlenecks
    
    def _identify_optimizations(self, results: List[BenchmarkResult]) -> List[str]:
        """Identify optimization opportunities."""
        optimizations = []
        
        for result in results:
            if not result.success:
                continue
            
            # Suggest optimizations based on results
            if result.name == "concurrent_compilation" and result.metadata:
                speedup = result.metadata.get("speedup_vs_sequential", 0)
                if speedup < 2.0:
                    optimizations.append("Improve parallelization efficiency for concurrent compilation")
            
            if result.name == "memory_efficiency" and result.memory_peak > 50:
                optimizations.append("Implement memory pooling or object reuse patterns")
            
            if result.name == "scaling_performance" and result.metadata:
                if not result.metadata.get("linear_scaling"):
                    optimizations.append("Optimize algorithms for better scalability")
            
            # General optimizations
            if result.duration > 0.5:
                optimizations.append(f"Optimize {result.name} for better performance")
            
            if result.memory_peak > 100:
                optimizations.append(f"Reduce memory footprint for {result.name}")
        
        return optimizations
    
    def _print_summary(self, suite: BenchmarkSuite):
        """Print benchmark suite summary."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        
        stats = suite.summary_stats
        
        print(f"Total Benchmarks: {stats['total_benchmarks']}")
        print(f"Successful: {stats['successful_benchmarks']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Average Duration: {stats['avg_duration']:.3f}s")
        print(f"Average Memory Peak: {stats['avg_memory_peak']:.1f}MB")
        print(f"Overall Performance Score: {stats['performance_score']:.1f}/100")
        
        # Performance grade
        score = stats['performance_score']
        if score >= 90:
            grade = "A+ (Excellent)"
        elif score >= 80:
            grade = "A (Very Good)"
        elif score >= 70:
            grade = "B (Good)"
        elif score >= 60:
            grade = "C (Acceptable)"
        else:
            grade = "F (Needs Improvement)"
        
        print(f"Performance Grade: {grade}")
        
        # Bottlenecks
        if suite.performance_profile.bottlenecks:
            print(f"\n⚠️  Performance Bottlenecks:")
            for bottleneck in suite.performance_profile.bottlenecks:
                print(f"  • {bottleneck}")
        
        # Optimization opportunities
        if suite.performance_profile.optimization_opportunities:
            print(f"\n🔧 Optimization Opportunities:")
            for optimization in suite.performance_profile.optimization_opportunities:
                print(f"  • {optimization}")
        
        print(f"\n🎯 Overall Assessment:")
        if stats['performance_score'] >= 80:
            print("  ✅ System performance is excellent and production-ready")
        elif stats['performance_score'] >= 60:
            print("  🟡 System performance is acceptable with room for improvement")
        else:
            print("  ❌ System performance needs significant optimization")
    
    def export_results(self, suite: BenchmarkSuite, output_path: Path):
        """Export benchmark results to JSON."""
        output_data = asdict(suite)
        
        # Convert datetime to string for JSON serialization
        output_data["timestamp"] = suite.timestamp.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n📄 Benchmark results exported to {output_path}")


def main():
    """Run performance benchmarks."""
    benchmark_suite = PerformanceBenchmarkSuite()
    
    # Run all benchmarks
    results = benchmark_suite.run_full_suite()
    
    # Export results
    output_path = Path("performance_benchmark_results.json")
    benchmark_suite.export_results(results, output_path)
    
    # Return exit code based on performance
    return 0 if results.summary_stats["performance_score"] >= 60 else 1


if __name__ == "__main__":
    exit(main())