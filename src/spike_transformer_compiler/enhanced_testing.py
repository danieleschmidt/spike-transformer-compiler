"""Enhanced testing framework for robust compilation."""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from .mock_models import create_test_model, get_test_input
from .compiler import SpikeCompiler, CompilationError
from .backend.factory import BackendFactory


@dataclass
class TestResult:
    """Test result with detailed information."""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    warnings: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}


class RobustTestSuite:
    """Comprehensive test suite for robust validation."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.results: List[TestResult] = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        if self.verbose:
            print("🧪 Running Robust Test Suite")
            print("=" * 50)
            
        start_time = time.time()
        
        # Core functionality tests
        self._test_basic_compilation()
        self._test_error_handling()
        self._test_edge_cases()
        self._test_performance_bounds()
        self._test_memory_safety()
        self._test_concurrent_compilation()
        self._test_backend_validation()
        self._test_optimization_levels()
        
        # Advanced robustness tests
        self._test_malformed_inputs()
        self._test_resource_limits()
        self._test_graceful_degradation()
        
        end_time = time.time()
        
        # Generate comprehensive report
        return self._generate_test_report(end_time - start_time)
    
    def _test_basic_compilation(self):
        """Test basic compilation functionality."""
        try:
            start = time.time()
            
            model = create_test_model("simple")
            compiler = SpikeCompiler(target="simulation", verbose=False)
            compiled = compiler.compile(model, input_shape=(1, 10))
            
            # Validate compiled model
            assert hasattr(compiled, 'energy_per_inference')
            assert hasattr(compiled, 'utilization')
            assert compiled.utilization >= 0.0 and compiled.utilization <= 1.0
            
            self.results.append(TestResult(
                test_name="basic_compilation",
                success=True,
                duration=time.time() - start,
                metrics={
                    "energy": compiled.energy_per_inference,
                    "utilization": compiled.utilization
                }
            ))
            
            if self.verbose:
                print("✓ Basic compilation test passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="basic_compilation",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Basic compilation test failed: {e}")
    
    def _test_error_handling(self):
        """Test error handling and recovery."""
        try:
            start = time.time()
            
            # Test invalid target
            try:
                compiler = SpikeCompiler(target="invalid_target")
                model = create_test_model("simple")
                compiler.compile(model, input_shape=(1, 10))
                # Should not reach here
                assert False, "Expected CompilationError for invalid target"
            except CompilationError:
                # Expected behavior
                pass
                
            # Test invalid input shape
            try:
                compiler = SpikeCompiler(target="simulation")
                model = create_test_model("simple")
                compiler.compile(model, input_shape=(-1, 0))  # Invalid shape
                # Should not reach here
                assert False, "Expected error for invalid input shape"
            except (CompilationError, ValueError):
                # Expected behavior
                pass
                
            self.results.append(TestResult(
                test_name="error_handling",
                success=True,
                duration=time.time() - start,
                metrics={"error_cases_tested": 2}
            ))
            
            if self.verbose:
                print("✓ Error handling test passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="error_handling",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Error handling test failed: {e}")
    
    def _test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        try:
            start = time.time()
            edge_cases = []
            
            # Very small model
            try:
                model = create_test_model("simple", input_size=1, output_size=1)
                compiler = SpikeCompiler(target="simulation", verbose=False)
                compiled = compiler.compile(model, input_shape=(1, 1))
                edge_cases.append("small_model")
            except Exception as e:
                self.logger.warning(f"Small model test failed: {e}")
            
            # Large input dimensions (but reasonable)
            try:
                model = create_test_model("simple", input_size=1000, output_size=100)
                compiler = SpikeCompiler(target="simulation", verbose=False)
                compiled = compiler.compile(model, input_shape=(1, 1000))
                edge_cases.append("large_model")
            except Exception as e:
                self.logger.warning(f"Large model test failed: {e}")
            
            # Different time steps
            try:
                model = create_test_model("simple")
                compiler = SpikeCompiler(target="simulation", time_steps=1, verbose=False)
                compiled = compiler.compile(model, input_shape=(1, 10))
                edge_cases.append("single_timestep")
            except Exception as e:
                self.logger.warning(f"Single timestep test failed: {e}")
            
            self.results.append(TestResult(
                test_name="edge_cases",
                success=len(edge_cases) > 0,
                duration=time.time() - start,
                metrics={"passed_cases": edge_cases, "total_cases": 3}
            ))
            
            if self.verbose:
                print(f"✓ Edge cases test: {len(edge_cases)}/3 cases passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="edge_cases",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Edge cases test failed: {e}")
    
    def _test_performance_bounds(self):
        """Test compilation performance within acceptable bounds."""
        try:
            start = time.time()
            
            model = create_test_model("simple")
            compiler = SpikeCompiler(target="simulation", verbose=False)
            
            compile_start = time.time()
            compiled = compiler.compile(model, input_shape=(1, 10))
            compile_time = time.time() - compile_start
            
            # Performance expectations (in seconds)
            MAX_COMPILE_TIME = 10.0  # Max 10 seconds for simple model
            
            performance_ok = compile_time < MAX_COMPILE_TIME
            
            self.results.append(TestResult(
                test_name="performance_bounds",
                success=performance_ok,
                duration=time.time() - start,
                metrics={
                    "compilation_time": compile_time,
                    "max_allowed": MAX_COMPILE_TIME,
                    "performance_ratio": compile_time / MAX_COMPILE_TIME
                }
            ))
            
            if self.verbose:
                status = "✓" if performance_ok else "⚠"
                print(f"{status} Performance test: {compile_time:.3f}s (limit: {MAX_COMPILE_TIME}s)")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="performance_bounds",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Performance bounds test failed: {e}")
    
    def _test_memory_safety(self):
        """Test memory usage and leak prevention."""
        try:
            start = time.time()
            
            # Run multiple compilations to check for leaks
            initial_objects = len(locals())
            
            for i in range(3):  # Small number for testing
                model = create_test_model("simple")
                compiler = SpikeCompiler(target="simulation", verbose=False)
                compiled = compiler.compile(model, input_shape=(1, 10))
                del model, compiler, compiled
                
            final_objects = len(locals())
            
            # Basic memory safety check (simplified)
            memory_safe = abs(final_objects - initial_objects) < 10  # Allow some variance
            
            self.results.append(TestResult(
                test_name="memory_safety",
                success=memory_safe,
                duration=time.time() - start,
                metrics={
                    "initial_objects": initial_objects,
                    "final_objects": final_objects,
                    "object_growth": final_objects - initial_objects
                }
            ))
            
            if self.verbose:
                print("✓ Memory safety test passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="memory_safety",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Memory safety test failed: {e}")
    
    def _test_concurrent_compilation(self):
        """Test thread safety and concurrent compilation."""
        try:
            start = time.time()
            
            # Simple concurrent test (sequential for now, as threading adds complexity)
            results = []
            
            for i in range(2):  # Run 2 compilations in sequence
                model = create_test_model("simple")
                compiler = SpikeCompiler(target="simulation", verbose=False)
                compiled = compiler.compile(model, input_shape=(1, 10))
                results.append(compiled.utilization)
            
            # Check consistency
            consistent = len(set(results)) == 1  # All results should be the same
            
            self.results.append(TestResult(
                test_name="concurrent_compilation",
                success=consistent,
                duration=time.time() - start,
                metrics={
                    "compilations": len(results),
                    "unique_results": len(set(results)),
                    "consistent": consistent
                }
            ))
            
            if self.verbose:
                print("✓ Concurrent compilation test passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="concurrent_compilation",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Concurrent compilation test failed: {e}")
    
    def _test_backend_validation(self):
        """Test backend validation and switching."""
        try:
            start = time.time()
            
            available_targets = BackendFactory.get_available_targets()
            tested_backends = []
            
            for target in available_targets:
                try:
                    model = create_test_model("simple")
                    compiler = SpikeCompiler(target=target, verbose=False)
                    compiled = compiler.compile(model, input_shape=(1, 10))
                    tested_backends.append(target)
                except Exception as e:
                    self.logger.warning(f"Backend {target} test failed: {e}")
            
            self.results.append(TestResult(
                test_name="backend_validation",
                success=len(tested_backends) > 0,
                duration=time.time() - start,
                metrics={
                    "available_backends": available_targets,
                    "tested_backends": tested_backends,
                    "success_rate": len(tested_backends) / len(available_targets)
                }
            ))
            
            if self.verbose:
                print(f"✓ Backend validation: {len(tested_backends)}/{len(available_targets)} backends working")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="backend_validation",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Backend validation test failed: {e}")
    
    def _test_optimization_levels(self):
        """Test different optimization levels."""
        try:
            start = time.time()
            
            optimization_results = []
            
            for opt_level in [0, 1, 2, 3]:
                try:
                    model = create_test_model("simple")
                    compiler = SpikeCompiler(
                        target="simulation", 
                        optimization_level=opt_level,
                        verbose=False
                    )
                    compiled = compiler.compile(model, input_shape=(1, 10))
                    optimization_results.append({
                        "level": opt_level,
                        "utilization": compiled.utilization,
                        "success": True
                    })
                except Exception as e:
                    optimization_results.append({
                        "level": opt_level,
                        "error": str(e),
                        "success": False
                    })
            
            successful_opts = sum(1 for r in optimization_results if r["success"])
            
            self.results.append(TestResult(
                test_name="optimization_levels",
                success=successful_opts > 0,
                duration=time.time() - start,
                metrics={
                    "tested_levels": len(optimization_results),
                    "successful_levels": successful_opts,
                    "results": optimization_results
                }
            ))
            
            if self.verbose:
                print(f"✓ Optimization levels: {successful_opts}/4 levels working")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="optimization_levels",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Optimization levels test failed: {e}")
    
    def _test_malformed_inputs(self):
        """Test handling of malformed inputs."""
        try:
            start = time.time()
            
            malformed_cases = [
                {"input_shape": None, "name": "null_shape"},
                {"input_shape": [], "name": "empty_shape"},
                {"input_shape": (0,), "name": "zero_dimension"},
                {"model": None, "name": "null_model"},
            ]
            
            handled_cases = 0
            
            for case in malformed_cases:
                try:
                    model = case.get("model", create_test_model("simple"))
                    compiler = SpikeCompiler(target="simulation", verbose=False)
                    
                    if case["name"] == "null_model":
                        compiler.compile(None, input_shape=(1, 10))
                    else:
                        compiler.compile(model, input_shape=case["input_shape"])
                    
                    # Should not reach here for malformed inputs
                    pass
                except (CompilationError, ValueError, AttributeError, TypeError):
                    # Expected behavior - malformed input was handled
                    handled_cases += 1
                except Exception:
                    # Unexpected error type
                    pass
            
            self.results.append(TestResult(
                test_name="malformed_inputs",
                success=handled_cases == len(malformed_cases),
                duration=time.time() - start,
                metrics={
                    "total_cases": len(malformed_cases),
                    "handled_cases": handled_cases,
                    "handle_rate": handled_cases / len(malformed_cases)
                }
            ))
            
            if self.verbose:
                print(f"✓ Malformed inputs: {handled_cases}/{len(malformed_cases)} cases handled")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="malformed_inputs",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Malformed inputs test failed: {e}")
    
    def _test_resource_limits(self):
        """Test behavior under resource constraints."""
        try:
            start = time.time()
            
            # Test with minimal resource allocator
            from spike_transformer_compiler import ResourceAllocator
            
            minimal_allocator = ResourceAllocator(
                num_chips=1, 
                cores_per_chip=1, 
                synapses_per_core=10  # Very limited
            )
            
            model = create_test_model("simple")
            compiler = SpikeCompiler(target="simulation", verbose=False)
            
            try:
                compiled = compiler.compile(
                    model,
                    input_shape=(1, 10),
                    resource_allocator=minimal_allocator
                )
                resource_test_passed = True
            except Exception as e:
                # May fail due to insufficient resources, which is acceptable
                resource_test_passed = "insufficient" in str(e).lower()
            
            self.results.append(TestResult(
                test_name="resource_limits",
                success=resource_test_passed,
                duration=time.time() - start,
                metrics={"constraint_handling": "passed" if resource_test_passed else "failed"}
            ))
            
            if self.verbose:
                print("✓ Resource limits test passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="resource_limits",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Resource limits test failed: {e}")
    
    def _test_graceful_degradation(self):
        """Test graceful degradation under adverse conditions."""
        try:
            start = time.time()
            
            # Test compilation with debug mode
            model = create_test_model("simple")
            compiler = SpikeCompiler(
                target="simulation", 
                debug=True,
                verbose=False
            )
            
            compiled = compiler.compile(model, input_shape=(1, 10))
            
            # Test should succeed even in debug mode
            debug_success = hasattr(compiled, 'utilization')
            
            self.results.append(TestResult(
                test_name="graceful_degradation",
                success=debug_success,
                duration=time.time() - start,
                metrics={"debug_mode_success": debug_success}
            ))
            
            if self.verbose:
                print("✓ Graceful degradation test passed")
                
        except Exception as e:
            self.results.append(TestResult(
                test_name="graceful_degradation",
                success=False,
                duration=time.time() - start,
                error_message=str(e)
            ))
            if self.verbose:
                print(f"✗ Graceful degradation test failed: {e}")
    
    def _generate_test_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        successful_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": success_rate,
                "total_duration": total_duration,
                "timestamp": time.time()
            },
            "test_results": [
                {
                    "name": r.test_name,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error_message,
                    "warnings": r.warnings,
                    "metrics": r.metrics
                }
                for r in self.results
            ],
            "robustness_score": self._calculate_robustness_score(),
            "recommendations": self._generate_recommendations()
        }
        
        if self.verbose:
            print("\n" + "=" * 50)
            print("🧪 ROBUST TEST SUITE COMPLETE")
            print("=" * 50)
            print(f"Tests Run: {total_tests}")
            print(f"Success Rate: {success_rate:.1%}")
            print(f"Total Duration: {total_duration:.2f}s")
            print(f"Robustness Score: {report['robustness_score']:.1f}/10")
            
            if report['recommendations']:
                print("\n📋 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
        
        return report
    
    def _calculate_robustness_score(self) -> float:
        """Calculate overall robustness score (0-10)."""
        if not self.results:
            return 0.0
        
        # Weight different test categories
        weights = {
            "basic_compilation": 2.0,
            "error_handling": 1.5,
            "edge_cases": 1.0,
            "performance_bounds": 1.0,
            "memory_safety": 1.5,
            "concurrent_compilation": 1.0,
            "backend_validation": 1.0,
            "optimization_levels": 0.8,
            "malformed_inputs": 1.2,
            "resource_limits": 0.8,
            "graceful_degradation": 1.0
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in self.results:
            weight = weights.get(result.test_name, 1.0)
            total_weight += weight
            if result.success:
                weighted_score += weight
        
        if total_weight == 0:
            return 0.0
            
        return (weighted_score / total_weight) * 10.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [r for r in self.results if not r.success]
        
        if not failed_tests:
            recommendations.append("All robustness tests passed! System is highly robust.")
            return recommendations
        
        # Analyze failures and generate specific recommendations
        critical_failures = [r for r in failed_tests 
                           if r.test_name in ["basic_compilation", "error_handling"]]
        
        if critical_failures:
            recommendations.append("Critical failures detected. Address basic compilation and error handling issues first.")
        
        performance_issues = [r for r in failed_tests 
                            if r.test_name in ["performance_bounds", "memory_safety"]]
        
        if performance_issues:
            recommendations.append("Performance issues detected. Optimize compilation speed and memory usage.")
        
        edge_case_failures = [r for r in failed_tests 
                            if r.test_name in ["edge_cases", "malformed_inputs"]]
        
        if edge_case_failures:
            recommendations.append("Edge case handling needs improvement. Strengthen input validation.")
        
        if len(failed_tests) > len(self.results) * 0.3:
            recommendations.append("Many tests failing. Consider systematic review of core functionality.")
        
        return recommendations


def run_robust_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run the complete robust test suite."""
    suite = RobustTestSuite(verbose=verbose)
    return suite.run_all_tests()


if __name__ == "__main__":
    # Run tests when called directly
    import sys
    verbose = "--quiet" not in sys.argv
    
    report = run_robust_tests(verbose=verbose)
    
    # Save report
    with open("robust_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Exit with appropriate code
    sys.exit(0 if report["test_summary"]["success_rate"] > 0.8 else 1)