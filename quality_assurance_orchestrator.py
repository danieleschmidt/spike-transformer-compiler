#!/usr/bin/env python3
"""Quality Assurance Orchestrator: Comprehensive QA automation.

This module orchestrates comprehensive quality assurance including automated testing,
security scanning, performance benchmarking, compliance validation, and deployment
readiness assessment for the Spike-Transformer-Compiler system.
"""

import os
import sys
import time
import json
import subprocess
import threading
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging


class QualityGateStatus(Enum):
    """Quality gate status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    RUNNING = "running"


class TestSuite(Enum):
    """Test suite types."""
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    PERFORMANCE_TESTS = "performance_tests"
    SECURITY_TESTS = "security_tests"
    COMPLIANCE_TESTS = "compliance_tests"
    SMOKE_TESTS = "smoke_tests"


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    status: QualityGateStatus
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp: float
    overall_status: QualityGateStatus
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    skipped_gates: int
    execution_time: float
    gate_results: List[QualityGateResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class TestRunner:
    """Advanced test runner with comprehensive reporting."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {}
        self.coverage_threshold = 85.0
        
    def run_unit_tests(self) -> QualityGateResult:
        """Run unit tests with coverage analysis."""
        start_time = time.time()
        
        try:
            # Check if pytest is available
            if not self._check_pytest_available():
                return QualityGateResult(
                    gate_name="unit_tests",
                    status=QualityGateStatus.SKIPPED,
                    execution_time=0,
                    error_message="pytest not available"
                )
            
            # Run pytest with coverage
            test_dir = self.project_root / "tests"
            if not test_dir.exists():
                return QualityGateResult(
                    gate_name="unit_tests",
                    status=QualityGateStatus.SKIPPED,
                    execution_time=time.time() - start_time,
                    error_message="No tests directory found"
                )
            
            # Prepare pytest command
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_dir),
                "-v",
                "--tb=short",
                "--maxfail=10",
                "--disable-warnings"
            ]
            
            # Try to add coverage if available
            try:
                subprocess.check_call([sys.executable, "-c", "import pytest_cov"], 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cmd.extend([
                    "--cov=spike_transformer_compiler",
                    "--cov-report=term-missing",
                    "--cov-report=json"
                ])
                coverage_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                coverage_available = False
            
            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")
            
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            execution_time = time.time() - start_time
            
            # Extract test metrics from output
            test_count, passed_count, failed_count = self._parse_pytest_output(result.stdout)
            
            # Check coverage if available
            coverage_percentage = 0.0
            if coverage_available:
                coverage_percentage = self._extract_coverage_percentage()
            
            # Determine status
            if result.returncode == 0:
                if coverage_available and coverage_percentage < self.coverage_threshold:
                    status = QualityGateStatus.WARNING
                    warnings = [f"Coverage {coverage_percentage:.1f}% below threshold {self.coverage_threshold}%"]
                else:
                    status = QualityGateStatus.PASSED
                    warnings = []
            else:
                status = QualityGateStatus.FAILED
                warnings = []
            
            return QualityGateResult(
                gate_name="unit_tests",
                status=status,
                execution_time=execution_time,
                details={
                    "total_tests": test_count,
                    "passed_tests": passed_count,
                    "failed_tests": failed_count,
                    "stdout": result.stdout[-1000:],  # Last 1000 chars
                    "stderr": result.stderr[-1000:] if result.stderr else ""
                },
                warnings=warnings,
                metrics={
                    "test_count": test_count,
                    "success_rate": (passed_count / max(1, test_count)) * 100,
                    "coverage_percentage": coverage_percentage
                },
                error_message=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="unit_tests",
                status=QualityGateStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message="Tests timed out after 5 minutes"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="unit_tests",
                status=QualityGateStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_performance_tests(self) -> QualityGateResult:
        """Run performance benchmarks."""
        start_time = time.time()
        
        try:
            # Check if performance test file exists
            perf_test_file = self.project_root / "tests" / "test_performance_benchmarks.py"
            if not perf_test_file.exists():
                return QualityGateResult(
                    gate_name="performance_tests",
                    status=QualityGateStatus.SKIPPED,
                    execution_time=time.time() - start_time,
                    error_message="Performance test file not found"
                )
            
            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")
            
            # Run performance tests
            cmd = [sys.executable, str(perf_test_file)]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                env=env,
                timeout=600  # 10 minute timeout for performance tests
            )
            
            execution_time = time.time() - start_time
            
            # Parse performance metrics from output
            performance_metrics = self._parse_performance_output(result.stdout)
            
            # Determine status based on performance thresholds
            status = QualityGateStatus.PASSED
            warnings = []
            
            # Check performance thresholds
            if "avg_latency" in performance_metrics and performance_metrics["avg_latency"] > 200:
                warnings.append(f"Average latency {performance_metrics['avg_latency']:.1f}ms exceeds 200ms threshold")
                status = QualityGateStatus.WARNING
            
            if "cache_hit_rate" in performance_metrics and performance_metrics["cache_hit_rate"] < 0.7:
                warnings.append(f"Cache hit rate {performance_metrics['cache_hit_rate']:.1%} below 70% threshold")
                status = QualityGateStatus.WARNING
            
            if result.returncode != 0:
                status = QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="performance_tests",
                status=status,
                execution_time=execution_time,
                details={
                    "stdout": result.stdout[-2000:],  # Last 2000 chars
                    "stderr": result.stderr[-1000:] if result.stderr else "",
                    "return_code": result.returncode
                },
                warnings=warnings,
                metrics=performance_metrics,
                error_message=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="performance_tests",
                status=QualityGateStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message="Performance tests timed out after 10 minutes"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_tests",
                status=QualityGateStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_security_tests(self) -> QualityGateResult:
        """Run security validation tests."""
        start_time = time.time()
        
        try:
            # Basic security checks
            security_issues = []
            
            # Check for hardcoded secrets
            security_issues.extend(self._scan_for_secrets())
            
            # Check for dangerous imports
            security_issues.extend(self._scan_for_dangerous_imports())
            
            # Check file permissions
            security_issues.extend(self._check_file_permissions())
            
            # Check for SQL injection patterns
            security_issues.extend(self._scan_for_sql_injection())
            
            execution_time = time.time() - start_time
            
            # Determine status
            critical_issues = [issue for issue in security_issues if issue.get("severity") == "critical"]
            high_issues = [issue for issue in security_issues if issue.get("severity") == "high"]
            
            if critical_issues:
                status = QualityGateStatus.FAILED
                error_message = f"Found {len(critical_issues)} critical security issues"
            elif high_issues:
                status = QualityGateStatus.WARNING
                error_message = None
            else:
                status = QualityGateStatus.PASSED
                error_message = None
            
            warnings = [f"{issue['severity'].upper()}: {issue['message']}" for issue in security_issues[:5]]
            
            return QualityGateResult(
                gate_name="security_tests",
                status=status,
                execution_time=execution_time,
                details={
                    "security_issues": security_issues,
                    "scanned_files": self._count_python_files()
                },
                warnings=warnings,
                metrics={
                    "total_issues": len(security_issues),
                    "critical_issues": len(critical_issues),
                    "high_issues": len(high_issues)
                },
                error_message=error_message
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="security_tests",
                status=QualityGateStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _check_pytest_available(self) -> bool:
        """Check if pytest is available."""
        try:
            subprocess.check_call(
                [sys.executable, "-c", "import pytest"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int]:
        """Parse pytest output to extract test counts."""
        import re
        
        # Look for patterns like "5 passed, 2 failed" or "10 passed"
        pattern = r"(\d+)\s+(passed|failed|error|skipped)"
        matches = re.findall(pattern, output)
        
        passed = 0
        failed = 0
        total = 0
        
        for count, status in matches:
            count = int(count)
            total += count
            if status == "passed":
                passed += count
            elif status in ["failed", "error"]:
                failed += count
        
        return total, passed, failed
    
    def _extract_coverage_percentage(self) -> float:
        """Extract coverage percentage from coverage report."""
        try:
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0.0)
        except Exception:
            pass
        return 0.0
    
    def _parse_performance_output(self, output: str) -> Dict[str, float]:
        """Parse performance test output for metrics."""
        metrics = {}
        
        # Look for performance metrics in output
        import re
        
        # Extract latency metrics
        latency_match = re.search(r"Average Duration: ([\d\.]+)s", output)
        if latency_match:
            metrics["avg_latency"] = float(latency_match.group(1)) * 1000  # Convert to ms
        
        # Extract throughput metrics
        throughput_match = re.search(r"Operations/Second: ([\d\.]+)", output)
        if throughput_match:
            metrics["throughput"] = float(throughput_match.group(1))
        
        # Extract cache hit rate
        cache_match = re.search(r"Cache Hit Rate: ([\d\.]+)", output)
        if cache_match:
            metrics["cache_hit_rate"] = float(cache_match.group(1))
        
        # Extract success rate
        success_match = re.search(r"Success Rate: ([\d\.]+)%", output)
        if success_match:
            metrics["success_rate"] = float(success_match.group(1))
        
        return metrics
    
    def _scan_for_secrets(self) -> List[Dict[str, Any]]:
        """Scan for potential hardcoded secrets."""
        issues = []
        secret_patterns = [
            (r"password\s*=\s*['\"][^'\"]{3,}['\"]?", "Potential hardcoded password"),
            (r"api_key\s*=\s*['\"][^'\"]{10,}['\"]?", "Potential hardcoded API key"),
            (r"secret\s*=\s*['\"][^'\"]{8,}['\"]?", "Potential hardcoded secret"),
            (r"token\s*=\s*['\"][^'\"]{15,}['\"]?", "Potential hardcoded token"),
        ]
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, message in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "message": message,
                            "severity": "high"
                        })
            except Exception:
                continue
        
        return issues
    
    def _scan_for_dangerous_imports(self) -> List[Dict[str, Any]]:
        """Scan for potentially dangerous imports."""
        issues = []
        dangerous_imports = [
            "subprocess",
            "os.system",
            "eval",
            "exec",
            "__import__",
            "compile"
        ]
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for dangerous in dangerous_imports:
                    if dangerous in content:
                        # Check if it's in a comment or string
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if dangerous in line and not line.strip().startswith('#'):
                                issues.append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "message": f"Potentially dangerous import/usage: {dangerous}",
                                    "severity": "medium"
                                })
                                break
            except Exception:
                continue
        
        return issues
    
    def _check_file_permissions(self) -> List[Dict[str, Any]]:
        """Check for overly permissive file permissions."""
        issues = []
        
        for py_file in self._get_python_files():
            try:
                mode = py_file.stat().st_mode
                # Check if file is world-writable
                if mode & 0o002:
                    issues.append({
                        "file": str(py_file.relative_to(self.project_root)),
                        "message": "File is world-writable",
                        "severity": "medium"
                    })
            except Exception:
                continue
        
        return issues
    
    def _scan_for_sql_injection(self) -> List[Dict[str, Any]]:
        """Scan for potential SQL injection vulnerabilities."""
        issues = []
        sql_patterns = [
            r"SELECT\s+.*\+.*FROM",
            r"INSERT\s+.*\+.*INTO",
            r"UPDATE\s+.*\+.*SET",
            r"DELETE\s+.*\+.*FROM",
        ]
        
        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in sql_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "message": "Potential SQL injection vulnerability",
                            "severity": "high"
                        })
                        break
            except Exception:
                continue
        
        return issues
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))
        return python_files
    
    def _count_python_files(self) -> int:
        """Count Python files in the project."""
        return len(self._get_python_files())


class CodeQualityAnalyzer:
    """Code quality analysis and metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def run_code_quality_analysis(self) -> QualityGateResult:
        """Run comprehensive code quality analysis."""
        start_time = time.time()
        
        try:
            metrics = {}
            warnings = []
            issues = []
            
            # Calculate code metrics
            metrics.update(self._calculate_code_metrics())
            
            # Check code complexity
            complexity_issues = self._check_code_complexity()
            issues.extend(complexity_issues)
            
            # Check documentation coverage
            doc_coverage = self._check_documentation_coverage()
            metrics["doc_coverage"] = doc_coverage
            
            if doc_coverage < 70:
                warnings.append(f"Documentation coverage {doc_coverage:.1f}% below 70% threshold")
            
            # Check code style (basic)
            style_issues = self._check_code_style()
            issues.extend(style_issues)
            
            # Determine overall status
            critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
            high_issues = [issue for issue in issues if issue.get("severity") == "high"]
            
            if critical_issues:
                status = QualityGateStatus.FAILED
                error_message = f"Found {len(critical_issues)} critical code quality issues"
            elif high_issues or warnings:
                status = QualityGateStatus.WARNING
                error_message = None
            else:
                status = QualityGateStatus.PASSED
                error_message = None
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                gate_name="code_quality",
                status=status,
                execution_time=execution_time,
                details={
                    "issues": issues[:10],  # Limit to first 10 issues
                    "total_files_analyzed": metrics.get("total_files", 0)
                },
                warnings=warnings,
                metrics=metrics,
                error_message=error_message
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="code_quality",
                status=QualityGateStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _calculate_code_metrics(self) -> Dict[str, float]:
        """Calculate basic code metrics."""
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "total_blank_lines": 0
        }
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                metrics["total_files"] += 1
                metrics["total_lines"] += len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        metrics["total_blank_lines"] += 1
                    elif stripped.startswith('#'):
                        metrics["total_comment_lines"] += 1
                    else:
                        metrics["total_code_lines"] += 1
                        
            except Exception:
                continue
        
        # Calculate ratios
        if metrics["total_lines"] > 0:
            metrics["comment_ratio"] = (metrics["total_comment_lines"] / metrics["total_lines"]) * 100
        else:
            metrics["comment_ratio"] = 0
        
        return metrics
    
    def _check_code_complexity(self) -> List[Dict[str, Any]]:
        """Check for overly complex functions."""
        issues = []
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple complexity check: count nested levels
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Count indentation level
                    indent_level = (len(line) - len(line.lstrip())) // 4
                    
                    if indent_level > 6:  # More than 6 levels of nesting
                        issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "message": f"High nesting level ({indent_level}) detected",
                            "severity": "medium"
                        })
                        
            except Exception:
                continue
        
        return issues
    
    def _check_documentation_coverage(self) -> float:
        """Check documentation coverage."""
        total_functions = 0
        documented_functions = 0
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple function detection
                import re
                function_matches = re.finditer(r'^\s*def\s+\w+', content, re.MULTILINE)
                
                for match in function_matches:
                    total_functions += 1
                    
                    # Check if next non-empty line starts with triple quotes
                    lines = content[match.end():].split('\n')
                    for line in lines[:5]:  # Check next 5 lines
                        stripped = line.strip()
                        if stripped and (stripped.startswith('"""') or stripped.startswith("'''")):
                            documented_functions += 1
                            break
                        elif stripped and not stripped.startswith('#'):
                            break
                            
            except Exception:
                continue
        
        if total_functions > 0:
            return (documented_functions / total_functions) * 100
        return 100.0
    
    def _check_code_style(self) -> List[Dict[str, Any]]:
        """Basic code style checks."""
        issues = []
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    # Check line length
                    if len(line.rstrip()) > 100:  # More than 100 characters
                        issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "message": f"Line too long ({len(line.rstrip())} characters)",
                            "severity": "low"
                        })
                    
                    # Check for trailing whitespace
                    if line.rstrip() != line.rstrip('\n').rstrip():
                        issues.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "message": "Trailing whitespace detected",
                            "severity": "low"
                        })
                        
            except Exception:
                continue
        
        return issues


class QualityAssuranceOrchestrator:
    """Main orchestrator for quality assurance processes."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.test_runner = TestRunner(self.project_root)
        self.code_analyzer = CodeQualityAnalyzer(self.project_root)
        self.quality_gates = [
            "unit_tests",
            "performance_tests",
            "security_tests",
            "code_quality"
        ]
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_all_quality_gates(self) -> QualityReport:
        """Run all quality gates and generate comprehensive report."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("SPIKE-TRANSFORMER-COMPILER QUALITY ASSURANCE")
        print("="*80)
        
        gate_results = []
        
        # Run each quality gate
        for gate_name in self.quality_gates:
            print(f"\n🔄 Running {gate_name.replace('_', ' ').title()}...")
            
            gate_result = self._run_quality_gate(gate_name)
            gate_results.append(gate_result)
            
            # Print immediate result
            status_emoji = {
                QualityGateStatus.PASSED: "✅",
                QualityGateStatus.WARNING: "⚠️",
                QualityGateStatus.FAILED: "❌",
                QualityGateStatus.SKIPPED: "⏭️"
            }
            
            print(f"   {status_emoji[gate_result.status]} {gate_result.status.value.upper()} in {gate_result.execution_time:.2f}s")
            
            if gate_result.warnings:
                for warning in gate_result.warnings[:3]:  # Show first 3 warnings
                    print(f"     ⚠️  {warning}")
            
            if gate_result.error_message:
                print(f"     ❌ {gate_result.error_message}")
        
        # Calculate overall metrics
        total_gates = len(gate_results)
        passed_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.PASSED)
        failed_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.FAILED)
        warning_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.WARNING)
        skipped_gates = sum(1 for r in gate_results if r.status == QualityGateStatus.SKIPPED)
        
        # Determine overall status
        if failed_gates > 0:
            overall_status = QualityGateStatus.FAILED
        elif warning_gates > 0:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.PASSED
        
        execution_time = time.time() - start_time
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results)
        
        # Create comprehensive report
        report = QualityReport(
            timestamp=time.time(),
            overall_status=overall_status,
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            warning_gates=warning_gates,
            skipped_gates=skipped_gates,
            execution_time=execution_time,
            gate_results=gate_results,
            summary=self._generate_summary(gate_results),
            recommendations=recommendations
        )
        
        # Print comprehensive summary
        self._print_quality_report(report)
        
        return report
    
    def _run_quality_gate(self, gate_name: str) -> QualityGateResult:
        """Run a specific quality gate."""
        try:
            if gate_name == "unit_tests":
                return self.test_runner.run_unit_tests()
            elif gate_name == "performance_tests":
                return self.test_runner.run_performance_tests()
            elif gate_name == "security_tests":
                return self.test_runner.run_security_tests()
            elif gate_name == "code_quality":
                return self.code_analyzer.run_code_quality_analysis()
            else:
                return QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.SKIPPED,
                    execution_time=0,
                    error_message=f"Unknown quality gate: {gate_name}"
                )
        except Exception as e:
            self.logger.error(f"Error running quality gate {gate_name}: {e}")
            return QualityGateResult(
                gate_name=gate_name,
                status=QualityGateStatus.FAILED,
                execution_time=0,
                error_message=str(e)
            )
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        for result in gate_results:
            if result.status == QualityGateStatus.FAILED:
                if result.gate_name == "unit_tests":
                    recommendations.append("Fix failing unit tests before deployment")
                elif result.gate_name == "performance_tests":
                    recommendations.append("Optimize performance bottlenecks identified in tests")
                elif result.gate_name == "security_tests":
                    recommendations.append("Address critical security vulnerabilities immediately")
                elif result.gate_name == "code_quality":
                    recommendations.append("Refactor code to improve quality metrics")
            
            elif result.status == QualityGateStatus.WARNING:
                if result.gate_name == "unit_tests":
                    if "coverage" in str(result.warnings):
                        recommendations.append("Increase test coverage to meet quality standards")
                elif result.gate_name == "performance_tests":
                    recommendations.append("Consider performance optimizations for better user experience")
                elif result.gate_name == "security_tests":
                    recommendations.append("Review and address security warnings")
                elif result.gate_name == "code_quality":
                    recommendations.append("Improve documentation and code style consistency")
        
        # Add general recommendations
        failed_count = sum(1 for r in gate_results if r.status == QualityGateStatus.FAILED)
        if failed_count == 0:
            recommendations.append("All critical quality gates passed - ready for deployment consideration")
        else:
            recommendations.append(f"Address {failed_count} failing quality gates before deployment")
        
        return recommendations
    
    def _generate_summary(self, gate_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate summary metrics from quality gate results."""
        summary = {
            "quality_score": 0.0,
            "total_execution_time": sum(r.execution_time for r in gate_results),
            "gate_status_distribution": {},
            "key_metrics": {}
        }
        
        # Calculate quality score (0-100)
        total_gates = len(gate_results)
        if total_gates > 0:
            passed_weight = 100
            warning_weight = 70
            failed_weight = 0
            skipped_weight = 50
            
            weighted_score = 0
            for result in gate_results:
                if result.status == QualityGateStatus.PASSED:
                    weighted_score += passed_weight
                elif result.status == QualityGateStatus.WARNING:
                    weighted_score += warning_weight
                elif result.status == QualityGateStatus.FAILED:
                    weighted_score += failed_weight
                elif result.status == QualityGateStatus.SKIPPED:
                    weighted_score += skipped_weight
            
            summary["quality_score"] = weighted_score / total_gates
        
        # Status distribution
        for result in gate_results:
            status = result.status.value
            summary["gate_status_distribution"][status] = summary["gate_status_distribution"].get(status, 0) + 1
        
        # Collect key metrics
        for result in gate_results:
            if result.metrics:
                for metric_name, metric_value in result.metrics.items():
                    summary["key_metrics"][f"{result.gate_name}_{metric_name}"] = metric_value
        
        return summary
    
    def _print_quality_report(self, report: QualityReport) -> None:
        """Print comprehensive quality report."""
        print("\n" + "="*80)
        print("QUALITY ASSURANCE REPORT")
        print("="*80)
        
        # Overall status
        status_emoji = {
            QualityGateStatus.PASSED: "✅",
            QualityGateStatus.WARNING: "⚠️",
            QualityGateStatus.FAILED: "❌"
        }
        
        print(f"\n{status_emoji[report.overall_status]} Overall Status: {report.overall_status.value.upper()}")
        print(f"📈 Quality Score: {report.summary['quality_score']:.1f}/100")
        print(f"⏱️  Total Execution Time: {report.execution_time:.2f}s")
        
        # Gate summary
        print(f"\n📊 Gate Summary:")
        print(f"   ✅ Passed: {report.passed_gates}")
        print(f"   ⚠️  Warnings: {report.warning_gates}")
        print(f"   ❌ Failed: {report.failed_gates}")
        print(f"   ⏭️  Skipped: {report.skipped_gates}")
        
        # Key metrics
        if report.summary["key_metrics"]:
            print(f"\n📊 Key Metrics:")
            for metric_name, metric_value in list(report.summary["key_metrics"].items())[:10]:
                if isinstance(metric_value, float):
                    print(f"   {metric_name}: {metric_value:.2f}")
                else:
                    print(f"   {metric_name}: {metric_value}")
        
        # Recommendations
        if report.recommendations:
            print(f"\n📝 Recommendations:")
            for i, recommendation in enumerate(report.recommendations[:5], 1):
                print(f"   {i}. {recommendation}")
        
        # Deployment readiness
        if report.overall_status == QualityGateStatus.PASSED:
            print(f"\n🚀 DEPLOYMENT READY: All quality gates passed successfully")
        elif report.overall_status == QualityGateStatus.WARNING:
            print(f"\n⚠️  DEPLOYMENT WITH CAUTION: Some quality gates have warnings")
        else:
            print(f"\n🚫 DEPLOYMENT BLOCKED: Critical quality gates failed")
        
        print("\n" + "="*80)
    
    def export_report(self, report: QualityReport, output_file: str = "quality_report.json") -> None:
        """Export quality report to JSON file."""
        report_data = {
            "timestamp": report.timestamp,
            "overall_status": report.overall_status.value,
            "summary": report.summary,
            "gate_results": [
                {
                    "gate_name": result.gate_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "metrics": result.metrics,
                    "warnings": result.warnings,
                    "error_message": result.error_message
                }
                for result in report.gate_results
            ],
            "recommendations": report.recommendations
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Quality report exported to {output_file}")


def main():
    """Main entry point for quality assurance orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Assurance Orchestrator")
    parser.add_argument("--project-root", help="Project root directory", default=".")
    parser.add_argument("--output", help="Output file for report", default="quality_report.json")
    parser.add_argument("--gates", nargs="*", help="Specific gates to run", 
                       choices=["unit_tests", "performance_tests", "security_tests", "code_quality"])
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = QualityAssuranceOrchestrator(args.project_root)
    
    # Override quality gates if specified
    if args.gates:
        orchestrator.quality_gates = args.gates
    
    # Run quality assurance
    try:
        report = orchestrator.run_all_quality_gates()
        
        # Export report
        orchestrator.export_report(report, args.output)
        
        # Exit with appropriate code
        if report.overall_status == QualityGateStatus.FAILED:
            sys.exit(1)
        elif report.overall_status == QualityGateStatus.WARNING:
            sys.exit(2)  # Warning exit code
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\n⚠️  Quality assurance interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error in quality assurance: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
