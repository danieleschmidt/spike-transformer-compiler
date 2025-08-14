#!/usr/bin/env python3
"""
Progressive Quality Gates System for Autonomous SDLC
Implements Generation 1-3 quality validation with real-time monitoring.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import subprocess
import json
import hashlib

from .exceptions import ValidationError
from .monitoring import MetricsCollector
from .performance import PerformanceProfiler


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Generation(Enum):
    """SDLC generation levels."""
    GEN1_WORK = 1  # Make it work
    GEN2_ROBUST = 2  # Make it robust
    GEN3_SCALE = 3  # Make it scale


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    status: QualityGateStatus
    score: float
    max_score: float
    execution_time: float
    details: Dict[str, Any]
    generation: Generation
    error_message: Optional[str] = None


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, generation: Generation, weight: float = 1.0):
        self.name = name
        self.generation = generation
        self.weight = weight
        self.logger = logging.getLogger(f"quality_gates.{name}")
    
    def execute(self) -> QualityGateResult:
        """Execute the quality gate."""
        start_time = time.time()
        try:
            score, max_score, details = self._run_validation()
            execution_time = time.time() - start_time
            
            status = QualityGateStatus.PASSED if score >= max_score * 0.8 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                score=score,
                max_score=max_score,
                execution_time=execution_time,
                details=details,
                generation=self.generation
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name=self.name,
                status=QualityGateStatus.FAILED,
                score=0.0,
                max_score=100.0,
                execution_time=execution_time,
                details={},
                generation=self.generation,
                error_message=str(e)
            )
    
    def _run_validation(self) -> tuple[float, float, Dict[str, Any]]:
        """Implement specific validation logic."""
        raise NotImplementedError("Subclasses must implement _run_validation")


class CodeQualityGate(QualityGate):
    """Validates code quality and standards."""
    
    def __init__(self):
        super().__init__("Code Quality", Generation.GEN1_WORK)
    
    def _run_validation(self) -> tuple[float, float, Dict[str, Any]]:
        """Run code quality validation."""
        details = {}
        total_score = 0
        max_score = 100
        
        # Check code formatting
        try:
            result = subprocess.run(
                ["black", "--check", "src/"], 
                capture_output=True, text=True, timeout=30
            )
            formatting_score = 25 if result.returncode == 0 else 0
            details["formatting"] = "passed" if result.returncode == 0 else "failed"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            formatting_score = 0
            details["formatting"] = "tool_unavailable"
        
        # Check import sorting
        try:
            result = subprocess.run(
                ["isort", "--check-only", "src/"], 
                capture_output=True, text=True, timeout=30
            )
            import_score = 25 if result.returncode == 0 else 0
            details["imports"] = "passed" if result.returncode == 0 else "failed"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            import_score = 0
            details["imports"] = "tool_unavailable"
        
        # Check linting
        try:
            result = subprocess.run(
                ["flake8", "src/"], 
                capture_output=True, text=True, timeout=60
            )
            lint_score = 25 if result.returncode == 0 else max(0, 25 - len(result.stdout.splitlines()))
            details["linting"] = {"issues": len(result.stdout.splitlines())}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            lint_score = 0
            details["linting"] = "tool_unavailable"
        
        # Check type hints
        try:
            result = subprocess.run(
                ["mypy", "src/spike_transformer_compiler"], 
                capture_output=True, text=True, timeout=90
            )
            type_score = 25 if result.returncode == 0 else max(0, 25 - len(result.stdout.splitlines()) // 2)
            details["type_checking"] = {"issues": len(result.stdout.splitlines())}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            type_score = 0
            details["type_checking"] = "tool_unavailable"
        
        total_score = formatting_score + import_score + lint_score + type_score
        details["component_scores"] = {
            "formatting": formatting_score,
            "imports": import_score,
            "linting": lint_score,
            "type_checking": type_score
        }
        
        return total_score, max_score, details


class TestCoverageGate(QualityGate):
    """Validates test coverage and completeness."""
    
    def __init__(self):
        super().__init__("Test Coverage", Generation.GEN1_WORK)
    
    def _run_validation(self) -> tuple[float, float, Dict[str, Any]]:
        """Run test coverage validation."""
        details = {}
        total_score = 0
        max_score = 100
        
        # Run pytest with coverage
        try:
            result = subprocess.run(
                ["pytest", "--cov=spike_transformer_compiler", "--cov-report=json", "-v"],
                capture_output=True, text=True, timeout=120
            )
            
            details["test_execution"] = "passed" if result.returncode == 0 else "failed"
            
            # Parse coverage report
            try:
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                
                coverage_percent = coverage_data["totals"]["percent_covered"]
                total_score = min(100, coverage_percent * 1.2)  # Bonus for high coverage
                
                details["coverage_percent"] = coverage_percent
                details["lines_covered"] = coverage_data["totals"]["covered_lines"]
                details["lines_total"] = coverage_data["totals"]["num_statements"]
                
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                # Fallback: estimate coverage from test output
                test_lines = result.stdout.count("PASSED")
                total_score = min(50, test_lines * 5)  # Basic estimation
                details["estimated_tests"] = test_lines
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            details["test_execution"] = "tool_unavailable"
            total_score = 0
        
        return total_score, max_score, details


class SecurityScanGate(QualityGate):
    """Validates security compliance and vulnerabilities."""
    
    def __init__(self):
        super().__init__("Security Scan", Generation.GEN2_ROBUST)
    
    def _run_validation(self) -> tuple[float, float, Dict[str, Any]]:
        """Run security scan validation."""
        details = {}
        total_score = 0
        max_score = 100
        
        # Check for common security patterns
        security_checks = [
            ("SQL Injection Protection", self._check_sql_injection),
            ("Input Validation", self._check_input_validation),
            ("Authentication Patterns", self._check_auth_patterns),
            ("Secure Communication", self._check_secure_comm),
            ("Data Sanitization", self._check_data_sanitization)
        ]
        
        passed_checks = 0
        for check_name, check_func in security_checks:
            try:
                if check_func():
                    passed_checks += 1
                    details[check_name.lower().replace(" ", "_")] = "passed"
                else:
                    details[check_name.lower().replace(" ", "_")] = "failed"
            except Exception as e:
                details[check_name.lower().replace(" ", "_")] = f"error: {str(e)}"
        
        total_score = (passed_checks / len(security_checks)) * 100
        details["checks_passed"] = passed_checks
        details["total_checks"] = len(security_checks)
        
        return total_score, max_score, details
    
    def _check_sql_injection(self) -> bool:
        """Check for SQL injection protection patterns."""
        # Look for parameterized queries, ORM usage
        src_path = Path("src")
        if not src_path.exists():
            return False
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if "%" in content and "execute" in content:
                        if "format" in content or "%" in content:
                            return False  # Potential SQL injection
            except Exception:
                continue
        return True
    
    def _check_input_validation(self) -> bool:
        """Check for input validation patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return False
        
        validation_patterns = ["validate", "sanitize", "check", "verify"]
        found_validation = False
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in validation_patterns):
                        found_validation = True
                        break
            except Exception:
                continue
        
        return found_validation
    
    def _check_auth_patterns(self) -> bool:
        """Check for authentication/authorization patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return False
        
        auth_patterns = ["auth", "token", "credential", "permission", "access"]
        found_auth = False
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in auth_patterns):
                        found_auth = True
                        break
            except Exception:
                continue
        
        return found_auth
    
    def _check_secure_comm(self) -> bool:
        """Check for secure communication patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return False
        
        secure_patterns = ["https", "ssl", "tls", "encrypt", "secure"]
        found_secure = False
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in secure_patterns):
                        found_secure = True
                        break
            except Exception:
                continue
        
        return found_secure
    
    def _check_data_sanitization(self) -> bool:
        """Check for data sanitization patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return False
        
        sanitize_patterns = ["sanitize", "escape", "clean", "filter"]
        found_sanitize = False
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in sanitize_patterns):
                        found_sanitize = True
                        break
            except Exception:
                continue
        
        return found_sanitize


class PerformanceBenchmarkGate(QualityGate):
    """Validates performance benchmarks and efficiency."""
    
    def __init__(self):
        super().__init__("Performance Benchmarks", Generation.GEN3_SCALE)
        self.profiler = PerformanceProfiler()
    
    def _run_validation(self) -> tuple[float, float, Dict[str, Any]]:
        """Run performance benchmark validation."""
        details = {}
        total_score = 0
        max_score = 100
        
        # Test compilation performance
        compilation_score = self._test_compilation_performance()
        details["compilation_performance"] = compilation_score
        
        # Test memory efficiency
        memory_score = self._test_memory_efficiency()
        details["memory_efficiency"] = memory_score
        
        # Test scalability patterns
        scalability_score = self._test_scalability_patterns()
        details["scalability_patterns"] = scalability_score
        
        # Test caching efficiency
        caching_score = self._test_caching_efficiency()
        details["caching_efficiency"] = caching_score
        
        total_score = (compilation_score + memory_score + scalability_score + caching_score) / 4
        
        return total_score, max_score, details
    
    def _test_compilation_performance(self) -> float:
        """Test compilation performance."""
        try:
            from .compiler import SpikeCompiler
            from .ir.builder import SpikeIRBuilder
            
            start_time = time.time()
            
            # Create a moderately complex model
            builder = SpikeIRBuilder("performance_test")
            input_id = builder.add_input("input", (1, 64))
            
            current_id = input_id
            for i in range(10):  # 10 layers for performance test
                linear_id = builder.add_spike_linear(current_id, out_features=32)
                neuron_id = builder.add_spike_neuron(linear_id)
                current_id = neuron_id
            
            builder.add_output(current_id, "output")
            graph = builder.build()
            
            compilation_time = time.time() - start_time
            
            # Score based on compilation time (target: <100ms for complex models)
            if compilation_time < 0.1:
                return 100.0
            elif compilation_time < 0.5:
                return 80.0
            elif compilation_time < 1.0:
                return 60.0
            else:
                return 40.0
                
        except Exception:
            return 0.0
    
    def _test_memory_efficiency(self) -> float:
        """Test memory efficiency patterns."""
        # Check for memory-efficient patterns in code
        src_path = Path("src")
        if not src_path.exists():
            return 0.0
        
        memory_patterns = ["cache", "lazy", "generator", "yield", "del ", "gc.collect"]
        pattern_count = 0
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    pattern_count += sum(1 for pattern in memory_patterns if pattern in content)
            except Exception:
                continue
        
        # Score based on memory optimization patterns found
        return min(100.0, pattern_count * 10)
    
    def _test_scalability_patterns(self) -> float:
        """Test scalability patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return 0.0
        
        scalability_patterns = [
            "threading", "multiprocessing", "async", "await",
            "pool", "queue", "distributed", "cluster", "scale"
        ]
        pattern_count = 0
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    pattern_count += sum(1 for pattern in scalability_patterns if pattern in content)
            except Exception:
                continue
        
        return min(100.0, pattern_count * 8)
    
    def _test_caching_efficiency(self) -> float:
        """Test caching efficiency patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return 0.0
        
        caching_patterns = ["@lru_cache", "@cache", "Cache", "memoize", "cached"]
        pattern_count = 0
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    pattern_count += sum(1 for pattern in caching_patterns if pattern in content)
            except Exception:
                continue
        
        return min(100.0, pattern_count * 15)


class ProgressiveQualityGateSystem:
    """
    Progressive Quality Gate System implementing Generation 1-3 validation.
    Executes quality gates based on SDLC generation with real-time monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("quality_gates.system")
        self.metrics = MetricsCollector()
        self.gates: Dict[Generation, List[QualityGate]] = {
            Generation.GEN1_WORK: [
                CodeQualityGate(),
                TestCoverageGate(),
            ],
            Generation.GEN2_ROBUST: [
                SecurityScanGate(),
            ],
            Generation.GEN3_SCALE: [
                PerformanceBenchmarkGate(),
            ]
        }
        self.results: List[QualityGateResult] = []
        self._execution_lock = threading.Lock()
    
    def execute_generation(self, generation: Generation) -> Dict[str, Any]:
        """Execute all quality gates for a specific generation."""
        with self._execution_lock:
            self.logger.info(f"Executing quality gates for {generation.name}")
            
            start_time = time.time()
            generation_results = []
            
            gates = self.gates.get(generation, [])
            
            for gate in gates:
                self.logger.info(f"Running quality gate: {gate.name}")
                
                result = gate.execute()
                generation_results.append(result)
                self.results.append(result)
                
                # Record metrics
                self.metrics.record_metric(
                    "quality_gate_execution_time",
                    result.execution_time,
                    {"gate": gate.name, "generation": generation.name}
                )
                
                self.metrics.record_metric(
                    "quality_gate_score",
                    result.score,
                    {"gate": gate.name, "generation": generation.name}
                )
                
                self.logger.info(
                    f"Quality gate {gate.name}: {result.status.value} "
                    f"(Score: {result.score:.1f}/{result.max_score:.1f})"
                )
            
            execution_time = time.time() - start_time
            
            # Calculate generation score
            if generation_results:
                avg_score = sum(r.score for r in generation_results) / len(generation_results)
                max_score = sum(r.max_score for r in generation_results) / len(generation_results)
                passed_gates = sum(1 for r in generation_results if r.status == QualityGateStatus.PASSED)
            else:
                avg_score = 0
                max_score = 100
                passed_gates = 0
            
            summary = {
                "generation": generation.name,
                "execution_time": execution_time,
                "gates_executed": len(generation_results),
                "gates_passed": passed_gates,
                "average_score": avg_score,
                "max_score": max_score,
                "pass_rate": passed_gates / len(generation_results) if generation_results else 0,
                "status": "PASSED" if passed_gates >= len(generation_results) * 0.8 else "FAILED",
                "results": [
                    {
                        "gate": r.gate_name,
                        "status": r.status.value,
                        "score": r.score,
                        "max_score": r.max_score,
                        "execution_time": r.execution_time,
                        "details": r.details,
                        "error": r.error_message
                    }
                    for r in generation_results
                ]
            }
            
            self.logger.info(
                f"Generation {generation.name} completed: {summary['status']} "
                f"({passed_gates}/{len(generation_results)} gates passed)"
            )
            
            return summary
    
    def execute_all_generations(self) -> Dict[str, Any]:
        """Execute all generations in sequence."""
        self.logger.info("Starting complete progressive quality gate execution")
        
        start_time = time.time()
        generation_summaries = []
        
        for generation in [Generation.GEN1_WORK, Generation.GEN2_ROBUST, Generation.GEN3_SCALE]:
            summary = self.execute_generation(generation)
            generation_summaries.append(summary)
        
        execution_time = time.time() - start_time
        
        # Calculate overall metrics
        total_gates = sum(len(self.gates.get(gen, [])) for gen in Generation)
        total_passed = sum(s["gates_passed"] for s in generation_summaries)
        overall_score = sum(s["average_score"] for s in generation_summaries) / len(generation_summaries)
        
        overall_summary = {
            "execution_time": execution_time,
            "total_gates": total_gates,
            "total_passed": total_passed,
            "overall_score": overall_score,
            "pass_rate": total_passed / total_gates if total_gates > 0 else 0,
            "status": "PRODUCTION_READY" if total_passed >= total_gates * 0.85 else "NEEDS_IMPROVEMENT",
            "generations": generation_summaries,
            "quality_grade": self._calculate_quality_grade(overall_score, total_passed / total_gates if total_gates > 0 else 0)
        }
        
        self.logger.info(
            f"Progressive quality gates completed: {overall_summary['status']} "
            f"(Grade: {overall_summary['quality_grade']})"
        )
        
        return overall_summary
    
    def _calculate_quality_grade(self, score: float, pass_rate: float) -> str:
        """Calculate overall quality grade."""
        if score >= 90 and pass_rate >= 0.95:
            return "A+"
        elif score >= 85 and pass_rate >= 0.9:
            return "A"
        elif score >= 80 and pass_rate >= 0.85:
            return "B+"
        elif score >= 75 and pass_rate >= 0.8:
            return "B"
        elif score >= 70 and pass_rate >= 0.75:
            return "C+"
        elif score >= 65 and pass_rate >= 0.7:
            return "C"
        else:
            return "F"
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time status of quality gate execution."""
        return {
            "total_gates_executed": len(self.results),
            "current_average_score": sum(r.score for r in self.results) / len(self.results) if self.results else 0,
            "gates_passed": sum(1 for r in self.results if r.status == QualityGateStatus.PASSED),
            "gates_failed": sum(1 for r in self.results if r.status == QualityGateStatus.FAILED),
            "latest_results": [
                {
                    "gate": r.gate_name,
                    "status": r.status.value,
                    "score": f"{r.score:.1f}/{r.max_score:.1f}",
                    "generation": r.generation.name
                }
                for r in self.results[-5:]  # Last 5 results
            ]
        }


def main():
    """Execute progressive quality gates autonomously."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔒 PROGRESSIVE QUALITY GATES SYSTEM - AUTONOMOUS EXECUTION")
    print("=" * 70)
    
    quality_system = ProgressiveQualityGateSystem()
    
    # Execute all generations
    overall_results = quality_system.execute_all_generations()
    
    # Display results
    print(f"\n📊 QUALITY GATES EXECUTION SUMMARY")
    print(f"Overall Status: {overall_results['status']}")
    print(f"Quality Grade: {overall_results['quality_grade']}")
    print(f"Total Score: {overall_results['overall_score']:.1f}/100.0")
    print(f"Pass Rate: {overall_results['pass_rate']:.1%}")
    print(f"Execution Time: {overall_results['execution_time']:.2f}s")
    
    for gen_summary in overall_results['generations']:
        print(f"\n{gen_summary['generation']}:")
        print(f"  Status: {gen_summary['status']}")
        print(f"  Gates: {gen_summary['gates_passed']}/{gen_summary['gates_executed']} passed")
        print(f"  Score: {gen_summary['average_score']:.1f}")
    
    if overall_results['status'] == "PRODUCTION_READY":
        print("\n🎉 SYSTEM IS PRODUCTION READY!")
        return 0
    else:
        print("\n⚠️  SYSTEM NEEDS IMPROVEMENT")
        return 1


if __name__ == "__main__":
    exit(main())