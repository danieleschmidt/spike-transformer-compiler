"""Comprehensive test suite for autonomous systems.

This module provides comprehensive testing for all autonomous systems including
research orchestrator, enhancement engine, resilience system, security system,
performance engine, and global deployment orchestrator.
"""

import pytest
import time
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the systems to test
try:
    from spike_transformer_compiler.research_orchestrator import (
        ResearchOrchestrator, ResearchHypothesis, ExperimentalFramework
    )
    from spike_transformer_compiler.autonomous_enhancement_engine import (
        SelfImprovingCompiler, PatternLearningEngine, AdaptiveOptimizer
    )
    from spike_transformer_compiler.enhanced_resilience_system import (
        ResilienceOrchestrator, CircuitBreaker, RetryManager, HealthMonitor
    )
    from spike_transformer_compiler.comprehensive_security_system import (
        ComprehensiveSecuritySystem, InputValidator, ThreatDetectionEngine
    )
    from spike_transformer_compiler.hyperscale_performance_engine import (
        HyperscalePerformanceEngine, AdaptiveCacheManager, AutoScalingManager
    )
    from spike_transformer_compiler.global_deployment_orchestrator import (
        GlobalDeploymentOrchestrator, InternationalizationManager, ComplianceManager
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestResearchOrchestrator:
    """Test suite for Research Orchestrator system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = ResearchOrchestrator()
        self.sample_compilation_history = [
            {
                "target": "loihi3",
                "stage_times": {"optimization": 3.0, "backend_compilation": 2.0},
                "final_performance": {"throughput": 800, "energy_efficiency": 0.6},
                "model_stats": {"num_parameters": 1000000}
            },
            {
                "target": "simulation",
                "stage_times": {"optimization": 4.0, "backend_compilation": 1.5},
                "final_performance": {"throughput": 1200, "energy_efficiency": 0.4},
                "model_stats": {"num_parameters": 1500000}
            }
        ]
    
    def test_opportunity_discovery(self):
        """Test research opportunity discovery."""
        opportunities = self.orchestrator.discover_research_opportunities(
            self.sample_compilation_history
        )
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        for opportunity in opportunities:
            assert "type" in opportunity
            assert "focus" in opportunity
            assert "potential_impact" in opportunity
            assert "description" in opportunity
    
    def test_autonomous_research_execution(self):
        """Test autonomous research execution."""
        opportunities = self.orchestrator.discover_research_opportunities(
            self.sample_compilation_history
        )
        
        # Test with limited opportunities to speed up test
        limited_opportunities = opportunities[:1] if opportunities else []
        
        research_results = self.orchestrator.execute_autonomous_research(limited_opportunities)
        
        assert isinstance(research_results, list)
        
        for result in research_results:
            assert "opportunity" in result
            assert "hypothesis" in result
            assert "framework" in result
            assert "results" in result
            
            # Verify hypothesis structure
            hypothesis = result["hypothesis"]
            assert hasattr(hypothesis, "id")
            assert hasattr(hypothesis, "title")
            assert hasattr(hypothesis, "success_metrics")
    
    def test_research_report_generation(self):
        """Test research report generation."""
        # Create mock research results
        mock_hypothesis = ResearchHypothesis(
            id="test_001",
            title="Test Hypothesis",
            description="Test description",
            success_metrics=["metric1", "metric2"],
            status="validated"
        )
        
        mock_results = [{
            "hypothesis": mock_hypothesis,
            "results": {
                "conclusion": "Test conclusion",
                "reproducibility_score": 0.85,
                "statistical_analysis": {
                    "p_value": 0.02,
                    "significant_improvements": [{"improvement_percentage": 25.0}]
                }
            }
        }]
        
        report = self.orchestrator.generate_research_report(mock_results)
        
        assert "summary" in report
        assert "detailed_results" in report
        assert "recommendations" in report
        assert "future_research_directions" in report
        
        assert report["summary"]["total_projects"] == 1
        assert report["summary"]["validated_hypotheses"] == 1


class TestSelfImprovingCompiler:
    """Test suite for Self-Improving Compiler system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.compiler = SelfImprovingCompiler()
    
    def test_pattern_learning(self):
        """Test compilation pattern learning."""
        compilation_data = {
            "target": "loihi3",
            "optimization_sequence": ["dead_code_elimination", "spike_fusion"],
            "stage_times": {"optimization": 2.5, "backend_compilation": 1.8},
            "final_performance": {"throughput": 800, "energy_efficiency": 0.6, "utilization": 0.75},
            "model_stats": {"num_parameters": 1000000, "complexity_score": 0.7},
            "target_info": {"hardware_type": "loihi3", "memory_limit": 1000},
            "optimization_level": 3,
            "model_type": "spikeformer"
        }
        
        # Test learning from compilation
        self.compiler.learn_from_compilation(compilation_data)
        
        # Verify patterns were learned
        assert len(self.compiler.pattern_learner.learned_patterns) > 0
        
        # Test pattern recommendation
        model_features = {
            "model_size": 1000000,
            "target_hardware": "loihi3",
            "model_type": "spikeformer"
        }
        
        recommendations = self.compiler.pattern_learner.recommend_optimization_sequence(model_features)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_capability_evolution(self):
        """Test new capability evolution."""
        def test_capability():
            return "test_capability_result"
        
        capability_id = self.compiler.evolve_new_capability(
            "Test Capability",
            test_capability
        )
        
        assert isinstance(capability_id, str)
        assert len(capability_id) > 0
        
        # Check evolution log
        assert len(self.compiler.evolution_log) > 0
        latest_entry = self.compiler.evolution_log[-1]
        assert latest_entry["capability_id"] == capability_id
        assert latest_entry["description"] == "Test Capability"
    
    def test_autonomous_improvements(self):
        """Test autonomous improvement generation."""
        improvements = self.compiler.generate_autonomous_improvements()
        
        assert isinstance(improvements, list)
        
        for improvement in improvements:
            assert "type" in improvement
            if "expected_improvement" in improvement:
                assert isinstance(improvement["expected_improvement"], (int, float))


class TestResilienceOrchestrator:
    """Test suite for Resilience Orchestrator system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = ResilienceOrchestrator()
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and configuration."""
        circuit_breaker = self.orchestrator.create_circuit_breaker("test_circuit")
        
        assert circuit_breaker.name == "test_circuit"
        assert "test_circuit" in self.orchestrator.circuit_breakers
        
        # Test circuit breaker functionality
        def test_function():
            return "success"
        
        result = circuit_breaker.call(test_function)
        assert result == "success"
    
    def test_resilient_execution(self):
        """Test resilient execution with recovery."""
        def successful_operation():
            return "success"
        
        def failing_operation():
            raise Exception("Test failure")
        
        # Test successful operation
        result = self.orchestrator.execute_with_resilience(
            "test_operation",
            successful_operation,
            enable_retry=False
        )
        assert result == "success"
        
        # Test failing operation with degradation
        result = self.orchestrator.execute_with_resilience(
            "test_failing_operation",
            failing_operation,
            enable_retry=False,
            enable_degradation=True
        )
        
        # Should return degraded result instead of raising exception
        assert isinstance(result, dict)
        assert result.get("status") == "degraded"
    
    def test_health_monitoring(self):
        """Test health monitoring system."""
        # Register a test health check
        def test_health_check():
            return True
        
        self.orchestrator.health_monitor.register_health_check("test_service", test_health_check)
        
        # Start monitoring briefly
        self.orchestrator.start_resilience_monitoring()
        time.sleep(1)  # Allow one monitoring cycle
        self.orchestrator.stop_resilience_monitoring()
        
        # Check health summary
        summary = self.orchestrator.health_monitor.get_health_summary()
        assert "current_status" in summary
    
    def test_resilience_summary(self):
        """Test resilience summary generation."""
        summary = self.orchestrator.get_resilience_summary()
        
        assert "overall_resilience_score" in summary
        assert "operation_metrics" in summary
        assert "health_status" in summary
        assert isinstance(summary["overall_resilience_score"], float)
        assert 0.0 <= summary["overall_resilience_score"] <= 1.0


class TestComprehensiveSecuritySystem:
    """Test suite for Comprehensive Security System."""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_system = ComprehensiveSecuritySystem()
        self.security_system.enable_security()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.security_system.disable_security()
    
    def test_input_validation(self):
        """Test input validation system."""
        validator = InputValidator()
        
        # Test valid input
        valid_data = {
            "model_type": "spikeformer",
            "parameters": {"layers": 12, "hidden_size": 768}
        }
        
        is_valid, errors = validator.validate_model_input(valid_data)
        assert is_valid == True
        assert len(errors) == 0
        
        # Test invalid input with malicious content
        malicious_data = {
            "model_type": "spikeformer",
            "code": "import os; os.system('rm -rf /')"
        }
        
        is_valid, errors = validator.validate_model_input(malicious_data)
        assert is_valid == False
        assert len(errors) > 0
    
    def test_threat_detection(self):
        """Test threat detection engine."""
        detector = ThreatDetectionEngine()
        
        # Test benign request
        benign_request = {
            "model_data": {"model_type": "spikeformer"},
            "config": {"target": "simulation"}
        }
        
        threat_level, threats = detector.analyze_compilation_request(benign_request)
        assert threat_level.value in ["low", "medium", "high", "critical"]
        
        # Test malicious request
        malicious_request = {
            "source_code": "import subprocess; subprocess.call(['rm', '-rf', '/'])",
            "config": {"allow_unsafe_operations": True}
        }
        
        threat_level, threats = detector.analyze_compilation_request(malicious_request)
        assert len(threats) > 0
    
    def test_secure_compilation_request(self):
        """Test secure compilation request processing."""
        # Test valid request
        valid_request = {
            "model_data": {
                "model_type": "spikeformer",
                "parameters": {"layers": 12}
            },
            "config": {
                "target": "simulation",
                "secure_mode": True
            }
        }
        
        allowed, result = self.security_system.secure_compilation_request(valid_request)
        assert allowed == True
        assert "request_id" in result
        
        # Test malicious request
        malicious_request = {
            "model_data": {
                "source_code": "__import__('os').system('evil_command')"
            },
            "config": {"target": "simulation"}
        }
        
        allowed, result = self.security_system.secure_compilation_request(malicious_request)
        assert allowed == False
        assert len(result.get("threats_detected", [])) > 0
    
    def test_artifact_security(self):
        """Test compilation artifact security."""
        test_artifact = b"test_compiled_model_data"
        test_metadata = {"artifact_id": "test_001", "encrypt": True}
        
        # Test artifact securing
        secured_artifact, secured_metadata = self.security_system.secure_compilation_artifact(
            test_artifact, test_metadata
        )
        
        assert "integrity_hash" in secured_metadata
        assert "signature" in secured_metadata
        assert secured_metadata.get("encrypted") == True
        
        # Test artifact verification
        verified, decrypted_artifact = self.security_system.verify_compilation_artifact(
            secured_artifact, secured_metadata
        )
        
        assert verified == True
        assert decrypted_artifact == test_artifact


class TestHyperscalePerformanceEngine:
    """Test suite for Hyperscale Performance Engine."""
    
    def setup_method(self):
        """Set up test environment."""
        self.engine = HyperscalePerformanceEngine()
    
    def test_cache_management(self):
        """Test adaptive cache management."""
        cache = AdaptiveCacheManager(max_size=10)
        
        # Test cache operations
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test cache statistics
        stats = cache.get_stats()
        assert "hit_rate" in stats
        assert "size" in stats
        assert stats["size"] == 1
    
    def test_load_balancing(self):
        """Test intelligent load balancing."""
        # Workers are initialized in setup
        selected_worker = self.engine.load_balancer.select_worker()
        assert selected_worker is not None
        assert selected_worker.startswith("worker_")
        
        # Test worker metrics update
        self.engine.load_balancer.update_worker_metrics(selected_worker, {
            "response_time": 100.0,
            "error_occurred": False,
            "request_completed": True
        })
        
        stats = self.engine.load_balancer.get_worker_stats()
        assert stats["total_workers"] > 0
        assert stats["healthy_workers"] > 0
    
    def test_auto_scaling(self):
        """Test auto-scaling functionality."""
        from spike_transformer_compiler.hyperscale_performance_engine import PerformanceMetrics
        
        # Create test metrics that should trigger scaling
        high_cpu_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_utilization=90.0,  # Above threshold
            memory_utilization=50.0,
            throughput=100.0,
            latency=50.0,
            queue_length=5,
            error_rate=0.0
        )
        
        initial_instances = self.engine.auto_scaler.current_instances
        
        # Update metrics (this should trigger scaling logic)
        self.engine.auto_scaler.update_metrics(high_cpu_metrics)
        
        # Get scaling stats
        stats = self.engine.auto_scaler.get_scaling_stats()
        assert "current_instances" in stats
        assert "policies_count" in stats
    
    def test_compilation_optimization(self):
        """Test compilation optimization."""
        from spike_transformer_compiler.hyperscale_performance_engine import OptimizationStrategy
        
        test_request = {
            "model_type": "spikeformer",
            "model_size": 1000000,
            "optimization_level": 2,
            "target": "simulation"
        }
        
        result = self.engine.optimize_compilation(
            test_request,
            optimization_strategies=[
                OptimizationStrategy.CACHING,
                OptimizationStrategy.LOAD_BALANCING
            ]
        )
        
        assert "optimizations_applied" in result
        assert "final_result" in result
        assert result["final_result"]["success"] == True
    
    def test_hyperscale_summary(self):
        """Test hyperscale performance summary."""
        summary = self.engine.get_hyperscale_summary()
        
        assert "performance_engine_status" in summary
        assert "cache_stats" in summary
        assert "load_balancer_stats" in summary
        assert "auto_scaling_stats" in summary
        assert "recent_performance" in summary


class TestGlobalDeploymentOrchestrator:
    """Test suite for Global Deployment Orchestrator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = GlobalDeploymentOrchestrator()
    
    def test_internationalization(self):
        """Test internationalization functionality."""
        i18n = InternationalizationManager()
        
        # Test localization
        english_message = i18n.localize_message("compilation_completed", "en")
        spanish_message = i18n.localize_message("compilation_completed", "es")
        
        assert english_message != spanish_message
        assert "compilation" in english_message.lower() or "completed" in english_message.lower()
        
        # Test number formatting
        english_number = i18n.format_number(1234.56, "en")
        german_number = i18n.format_number(1234.56, "de")
        
        assert english_number != german_number
    
    def test_compliance_management(self):
        """Test compliance management."""
        from spike_transformer_compiler.global_deployment_orchestrator import Region, ComplianceFramework
        
        compliance_manager = ComplianceManager()
        
        # Test GDPR requirements
        gdpr_requirements = compliance_manager.get_requirements_for_region(Region.EU_WEST)
        assert len(gdpr_requirements) > 0
        
        # Test compliance validation
        test_config = {
            "encryption_enabled": True,
            "audit_logging_enabled": True,
            "data_retention_policy": "7_years"
        }
        
        validation_result = compliance_manager.validate_compliance(test_config, Region.EU_WEST)
        assert "overall_compliance" in validation_result
        assert "compliance_score" in validation_result
    
    def test_deployment_planning(self):
        """Test deployment planning."""
        from spike_transformer_compiler.global_deployment_orchestrator import Region
        
        deployment_config = {
            "performance_requirements": {
                "max_latency_ms": 100,
                "min_throughput_ops_sec": 1000
            }
        }
        
        # Test planning with limited regions for faster testing
        test_regions = [Region.US_EAST, Region.EU_WEST]
        
        results = self.orchestrator.deploy_globally(
            deployment_config=deployment_config,
            target_regions=test_regions
        )
        
        assert "deployment_id" in results
        assert "deployment_results" in results
        assert "compliance_status" in results
        assert "performance_metrics" in results


class TestIntegration:
    """Integration tests for combined system functionality."""
    
    def test_full_system_integration(self):
        """Test integration between multiple systems."""
        # Initialize all systems
        research_orchestrator = ResearchOrchestrator()
        security_system = ComprehensiveSecuritySystem()
        performance_engine = HyperscalePerformanceEngine()
        
        # Enable security
        security_system.enable_security()
        
        try:
            # Test secure compilation with performance optimization
            compilation_request = {
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
            
            # Security validation
            allowed, security_result = security_system.secure_compilation_request(compilation_request)
            assert allowed == True
            
            # Performance optimization
            if allowed:
                optimization_result = performance_engine.optimize_compilation(compilation_request)
                assert optimization_result["final_result"]["success"] == True
            
            # Research opportunity discovery
            sample_history = [{
                "target": "simulation",
                "stage_times": {"optimization": 1.0, "backend_compilation": 0.5},
                "final_performance": {"throughput": 1000, "energy_efficiency": 0.8},
                "model_stats": {"num_parameters": 768000}
            }]
            
            opportunities = research_orchestrator.discover_research_opportunities(sample_history)
            assert isinstance(opportunities, list)
            
        finally:
            security_system.disable_security()
    
    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        performance_engine = HyperscalePerformanceEngine()
        
        # Start performance monitoring
        performance_engine.start_performance_monitoring()
        
        try:
            # Simulate multiple concurrent requests
            requests = []
            for i in range(5):  # Limited for test speed
                request = {
                    "model_type": "spikeformer",
                    "model_size": 500000 + i * 100000,
                    "optimization_level": 2,
                    "target": "simulation"
                }
                requests.append(request)
            
            # Process requests
            results = []
            for request in requests:
                result = performance_engine.optimize_compilation(request)
                results.append(result)
                assert result["final_result"]["success"] == True
            
            # Allow monitoring to collect data
            time.sleep(2)
            
            # Check performance summary
            summary = performance_engine.get_hyperscale_summary()
            assert summary["performance_engine_status"] == "active"
            
        finally:
            performance_engine.stop_performance_monitoring()
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery across systems."""
        resilience_orchestrator = ResilienceOrchestrator()
        
        # Test with a function that sometimes fails
        failure_count = 0
        
        def unreliable_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 times
                raise Exception("Simulated failure")
            return "success_after_retries"
        
        # Execute with resilience
        result = resilience_orchestrator.execute_with_resilience(
            "test_unreliable_operation",
            unreliable_operation,
            enable_retry=True,
            enable_degradation=True
        )
        
        # Should either succeed or return degraded result
        assert result is not None
        
        # Check resilience summary
        summary = resilience_orchestrator.get_resilience_summary()
        assert summary["operation_metrics"]["total_operations"] > 0


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
