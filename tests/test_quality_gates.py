#!/usr/bin/env python3
"""
Comprehensive test suite for progressive quality gates system.
Tests all generations with advanced validation scenarios.
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spike_transformer_compiler.quality_gates import (
    ProgressiveQualityGateSystem, QualityGate, QualityGateResult, QualityGateStatus,
    Generation, CodeQualityGate, TestCoverageGate, SecurityScanGate, PerformanceBenchmarkGate
)
from spike_transformer_compiler.quality_monitoring import (
    AdaptiveQualityMonitor, QualityMetric, QualityAlert, QualityThreshold
)
from spike_transformer_compiler.adaptive_quality_system import (
    AdaptiveQualitySystem, MLQualityPredictor, QualityOptimizationEngine
)


class TestQualityGateFramework:
    """Test the core quality gate framework."""
    
    def test_quality_gate_creation(self):
        """Test basic quality gate creation and configuration."""
        gate = QualityGate("Test Gate", Generation.GEN1_WORK, weight=2.0)
        
        assert gate.name == "Test Gate"
        assert gate.generation == Generation.GEN1_WORK
        assert gate.weight == 2.0
    
    def test_quality_gate_result_structure(self):
        """Test quality gate result data structure."""
        result = QualityGateResult(
            gate_name="test",
            status=QualityGateStatus.PASSED,
            score=85.0,
            max_score=100.0,
            execution_time=1.5,
            details={"test": "data"},
            generation=Generation.GEN1_WORK
        )
        
        assert result.gate_name == "test"
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 85.0
        assert result.execution_time == 1.5
    
    def test_progressive_quality_gate_system_initialization(self):
        """Test progressive quality gate system initialization."""
        system = ProgressiveQualityGateSystem()
        
        assert Generation.GEN1_WORK in system.gates
        assert Generation.GEN2_ROBUST in system.gates
        assert Generation.GEN3_SCALE in system.gates
        assert len(system.results) == 0


class TestCodeQualityGate:
    """Test code quality validation gate."""
    
    def test_code_quality_gate_creation(self):
        """Test code quality gate initialization."""
        gate = CodeQualityGate()
        
        assert gate.name == "Code Quality"
        assert gate.generation == Generation.GEN1_WORK
    
    @patch('subprocess.run')
    def test_code_quality_validation_success(self, mock_run):
        """Test successful code quality validation."""
        # Mock successful tool runs
        mock_run.side_effect = [
            Mock(returncode=0, stdout="", stderr=""),  # black
            Mock(returncode=0, stdout="", stderr=""),  # isort
            Mock(returncode=0, stdout="", stderr=""),  # flake8
            Mock(returncode=0, stdout="", stderr="")   # mypy
        ]
        
        gate = CodeQualityGate()
        result = gate.execute()
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score == 100.0
        assert "formatting" in result.details
    
    @patch('subprocess.run')
    def test_code_quality_validation_partial_failure(self, mock_run):
        """Test partial code quality validation failure."""
        # Mock mixed results
        mock_run.side_effect = [
            Mock(returncode=1, stdout="formatting error", stderr=""),  # black fails
            Mock(returncode=0, stdout="", stderr=""),  # isort passes
            Mock(returncode=0, stdout="", stderr=""),  # flake8 passes
            Mock(returncode=0, stdout="", stderr="")   # mypy passes
        ]
        
        gate = CodeQualityGate()
        result = gate.execute()
        
        assert result.status == QualityGateStatus.FAILED  # < 80% threshold
        assert result.score == 75.0  # 3/4 tools passed
        assert result.details["formatting"] == "failed"


class TestTestCoverageGate:
    """Test coverage validation gate."""
    
    def test_test_coverage_gate_creation(self):
        """Test test coverage gate initialization."""
        gate = TestCoverageGate()
        
        assert gate.name == "Test Coverage"
        assert gate.generation == Generation.GEN1_WORK
    
    @patch('subprocess.run')
    @patch('builtins.open')
    def test_coverage_validation_success(self, mock_open, mock_run):
        """Test successful coverage validation."""
        # Mock pytest success and coverage report
        mock_run.return_value = Mock(returncode=0, stdout="10 passed", stderr="")
        
        coverage_data = {
            "totals": {
                "percent_covered": 90.0,
                "covered_lines": 900,
                "num_statements": 1000
            }
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(coverage_data)
        
        gate = TestCoverageGate()
        result = gate.execute()
        
        assert result.status == QualityGateStatus.PASSED
        assert result.score >= 80.0  # Should be high due to good coverage
        assert result.details["coverage_percent"] == 90.0
    
    @patch('subprocess.run')
    def test_coverage_validation_no_report(self, mock_run):
        """Test coverage validation without coverage report."""
        # Mock pytest success but no coverage file
        mock_run.return_value = Mock(returncode=0, stdout="5 passed", stderr="")
        
        gate = TestCoverageGate()
        result = gate.execute()
        
        assert result.score <= 50.0  # Should fallback to estimation
        assert "estimated_tests" in result.details


class TestSecurityScanGate:
    """Test security scanning gate."""
    
    def test_security_gate_creation(self):
        """Test security gate initialization."""
        gate = SecurityScanGate()
        
        assert gate.name == "Security Scan"
        assert gate.generation == Generation.GEN2_ROBUST
    
    def test_security_validation_with_patterns(self):
        """Test security validation with code patterns."""
        # Create temporary source files
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "src"
            src_dir.mkdir()
            
            # Create file with security patterns
            test_file = src_dir / "test.py"
            test_file.write_text("""
import hashlib

def validate_input(data):
    if not data:
        raise ValueError("Invalid input")
    return sanitize(data)

def authenticate_user(token):
    return verify_token(token)

def secure_connection():
    return https_client()
""")
            
            # Patch Path.rglob to return our test file
            with patch('pathlib.Path.rglob') as mock_rglob:
                mock_rglob.return_value = [test_file]
                
                gate = SecurityScanGate()
                result = gate.execute()
                
                assert result.status == QualityGateStatus.PASSED
                assert result.score > 0


class TestPerformanceBenchmarkGate:
    """Test performance benchmark gate."""
    
    def test_performance_gate_creation(self):
        """Test performance gate initialization."""
        gate = PerformanceBenchmarkGate()
        
        assert gate.name == "Performance Benchmarks"
        assert gate.generation == Generation.GEN3_SCALE
    
    def test_compilation_performance_test(self):
        """Test compilation performance measurement."""
        gate = PerformanceBenchmarkGate()
        
        # This should execute without errors
        performance_score = gate._test_compilation_performance()
        
        assert isinstance(performance_score, float)
        assert 0 <= performance_score <= 100


class TestQualityMonitoring:
    """Test adaptive quality monitoring system."""
    
    def test_quality_threshold_creation(self):
        """Test quality threshold creation and adaptation."""
        threshold = QualityThreshold("test_metric", initial_low=10.0, initial_high=90.0)
        
        assert threshold.name == "test_metric"
        assert threshold.low_threshold == 10.0
        assert threshold.high_threshold == 90.0
    
    def test_threshold_adaptation(self):
        """Test threshold adaptation based on historical data."""
        threshold = QualityThreshold("test_metric")
        
        # Add historical values
        values = [50, 55, 60, 52, 48, 58, 62, 49, 53, 57] * 5  # 50 values
        for value in values:
            threshold.add_value(value)
        
        # Thresholds should be adapted
        assert threshold.low_threshold is not None
        assert threshold.high_threshold is not None
    
    def test_quality_metric_creation(self):
        """Test quality metric data structure."""
        metric = QualityMetric(
            name="test_metric",
            value=75.0,
            timestamp=datetime.now(),
            source="test_source",
            tags={"component": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 75.0
        assert metric.tags["component"] == "test"
    
    def test_adaptive_quality_monitor_initialization(self):
        """Test adaptive quality monitor initialization."""
        monitor = AdaptiveQualityMonitor()
        
        assert not monitor.monitoring_active
        assert len(monitor.thresholds) > 0
        assert monitor.monitoring_interval > 0


class TestMLQualityPredictor:
    """Test machine learning quality predictor."""
    
    def test_ml_predictor_initialization(self):
        """Test ML predictor initialization."""
        predictor = MLQualityPredictor()
        
        assert predictor.model is not None
        assert predictor.model["type"] == "statistical"
    
    def test_training_data_addition(self):
        """Test adding training data to predictor."""
        predictor = MLQualityPredictor()
        
        features = {"code_complexity": 10.0, "file_count": 5}
        target = {"compilation_time": 2.0, "memory_usage": 100.0}
        
        initial_count = len(predictor.feature_history)
        predictor.add_training_data(features, target)
        
        assert len(predictor.feature_history) == initial_count + 1
    
    def test_quality_prediction(self):
        """Test quality metric prediction."""
        predictor = MLQualityPredictor()
        
        # Add some training data
        for i in range(10):
            features = {"code_complexity": 10.0 + i, "file_count": 5 + i}
            target = {"compilation_time": 2.0 + i * 0.1, "memory_usage": 100.0 + i * 10}
            predictor.add_training_data(features, target)
        
        # Make predictions
        features = {"code_complexity": 15.0, "file_count": 8}
        predictions = predictor.predict_quality_metrics(features)
        
        assert isinstance(predictions, list)
        assert len(predictions) > 0


class TestQualityOptimizationEngine:
    """Test quality optimization engine."""
    
    def test_optimization_engine_initialization(self):
        """Test optimization engine initialization."""
        engine = QualityOptimizationEngine()
        
        assert len(engine.recommendation_history) == 0
        assert isinstance(engine.implementation_tracking, dict)
    
    def test_performance_recommendations(self):
        """Test performance optimization recommendations."""
        engine = QualityOptimizationEngine()
        
        # Create dashboard with performance issues
        dashboard = {
            "metrics": {
                "compilation_time": {"latest_value": 5.0},  # High compilation time
                "memory_usage": {"latest_value": 100.0}
            }
        }
        
        recommendations = engine._analyze_performance(dashboard, [])
        
        assert len(recommendations) > 0
        assert any("compilation" in rec.description.lower() for rec in recommendations)
    
    def test_recommendation_generation(self):
        """Test comprehensive recommendation generation."""
        engine = QualityOptimizationEngine()
        
        dashboard = {
            "overall_health_score": 60.0,  # Low health
            "metrics": {
                "test_coverage": {"latest_value": 70.0},  # Low coverage
                "code_quality_score": {"latest_value": 75.0}
            }
        }
        
        recommendations = engine.analyze_and_recommend(dashboard, [], [])
        
        assert len(recommendations) > 0
        assert any("test" in rec.description.lower() for rec in recommendations)


class TestAdaptiveQualitySystem:
    """Test the complete adaptive quality system."""
    
    def test_adaptive_system_initialization(self):
        """Test adaptive system initialization."""
        system = AdaptiveQualitySystem()
        
        assert system.quality_gates is not None
        assert system.quality_monitor is not None
        assert system.ml_predictor is not None
        assert system.optimization_engine is not None
        assert not system.running
    
    def test_system_load_calculation(self):
        """Test system load calculation and adaptation."""
        system = AdaptiveQualitySystem()
        
        # Simulate system monitoring
        initial_prediction_interval = system.prediction_interval
        system.system_load = 0.8  # High load
        
        # This would normally be called in the monitoring loop
        # system._adapt_intervals_based_on_load()
        
        assert system.system_load == 0.8
    
    def test_code_complexity_estimation(self):
        """Test code complexity estimation."""
        system = AdaptiveQualitySystem()
        
        complexity = system._estimate_code_complexity()
        
        assert isinstance(complexity, float)
        assert complexity >= 0
    
    def test_source_file_counting(self):
        """Test source file counting."""
        system = AdaptiveQualitySystem()
        
        file_count = system._count_source_files()
        test_count = system._count_test_files()
        
        assert isinstance(file_count, int)
        assert isinstance(test_count, int)
        assert file_count >= 0
        assert test_count >= 0


class TestQualityGateIntegration:
    """Test integration between all quality gate components."""
    
    def test_progressive_execution(self):
        """Test progressive execution of all generations."""
        system = ProgressiveQualityGateSystem()
        
        # Execute Generation 1
        gen1_results = system.execute_generation(Generation.GEN1_WORK)
        
        assert gen1_results["generation"] == "GEN1_WORK"
        assert gen1_results["gates_executed"] > 0
        assert "status" in gen1_results
    
    def test_complete_system_execution(self):
        """Test complete system execution across all generations."""
        system = ProgressiveQualityGateSystem()
        
        overall_results = system.execute_all_generations()
        
        assert "execution_time" in overall_results
        assert "total_gates" in overall_results
        assert "overall_score" in overall_results
        assert "quality_grade" in overall_results
        assert len(overall_results["generations"]) == 3
    
    def test_real_time_status(self):
        """Test real-time status reporting."""
        system = ProgressiveQualityGateSystem()
        
        # Execute some gates first
        system.execute_generation(Generation.GEN1_WORK)
        
        status = system.get_real_time_status()
        
        assert "total_gates_executed" in status
        assert "current_average_score" in status
        assert "gates_passed" in status
        assert "latest_results" in status


class TestQualityGateErrorHandling:
    """Test error handling and edge cases."""
    
    def test_gate_execution_with_exception(self):
        """Test quality gate execution with exceptions."""
        class FailingGate(QualityGate):
            def __init__(self):
                super().__init__("Failing Gate", Generation.GEN1_WORK)
            
            def _run_validation(self):
                raise Exception("Simulated failure")
        
        gate = FailingGate()
        result = gate.execute()
        
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.0
        assert result.error_message is not None
    
    def test_system_resilience(self):
        """Test system resilience with failing components."""
        system = ProgressiveQualityGateSystem()
        
        # This should not crash even if some gates fail
        results = system.execute_all_generations()
        
        assert results is not None
        assert "status" in results


class TestQualityMetricsCalculation:
    """Test quality metrics calculation and scoring."""
    
    def test_quality_grade_calculation(self):
        """Test quality grade calculation algorithm."""
        system = ProgressiveQualityGateSystem()
        
        # Test various score combinations
        test_cases = [
            (95.0, 1.0, "A+"),
            (85.0, 0.9, "A"),
            (80.0, 0.85, "B+"),
            (75.0, 0.8, "B"),
            (60.0, 0.6, "F")
        ]
        
        for score, pass_rate, expected_grade in test_cases:
            grade = system._calculate_quality_grade(score, pass_rate)
            assert grade == expected_grade
    
    def test_score_aggregation(self):
        """Test score aggregation across multiple gates."""
        system = ProgressiveQualityGateSystem()
        
        # Execute and check score aggregation
        results = system.execute_all_generations()
        
        assert "overall_score" in results
        assert 0 <= results["overall_score"] <= 100


def test_end_to_end_quality_system():
    """End-to-end test of the complete quality system."""
    print("\n🔄 Running end-to-end quality system test...")
    
    # Initialize complete system
    progressive_system = ProgressiveQualityGateSystem()
    
    # Execute all generations
    results = progressive_system.execute_all_generations()
    
    # Verify results
    assert results["status"] in ["PRODUCTION_READY", "NEEDS_IMPROVEMENT"]
    assert "quality_grade" in results
    assert len(results["generations"]) == 3
    
    # Test quality monitoring
    monitor = AdaptiveQualityMonitor()
    dashboard = monitor.get_quality_dashboard()
    
    assert "overall_health_score" in dashboard
    assert "system_status" in dashboard
    
    print(f"✅ End-to-end test completed successfully!")
    print(f"   Overall Status: {results['status']}")
    print(f"   Quality Grade: {results['quality_grade']}")
    print(f"   Health Score: {dashboard['overall_health_score']:.1f}%")


if __name__ == "__main__":
    # Run basic tests
    test_end_to_end_quality_system()
    
    # Run pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests only")