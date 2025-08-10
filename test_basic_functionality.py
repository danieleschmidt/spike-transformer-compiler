#!/usr/bin/env python3
"""Basic functionality test without external dependencies."""

import sys
import os
import time
import traceback
from typing import Any, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports without PyTorch dependency."""
    try:
        # Test basic data structures 
        from spike_transformer_compiler.ir.types import SpikeTensor, SpikeType
        print("✓ IR types import successfully")
        
        # Test basic graph components (will fail due to torch import)
        try:
            from spike_transformer_compiler.ir.spike_graph import NodeType
            print("✓ NodeType enum imports successfully") 
        except ImportError:
            print("✗ SpikeGraph imports failed (expected due to torch dependency)")
        
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_ir_types():
    """Test IR type system."""
    try:
        from spike_transformer_compiler.ir.types import SpikeTensor, SpikeType, MembraneState, SynapticWeights
        
        # Test SpikeTensor creation
        tensor = SpikeTensor(shape=(1, 3, 224, 224), spike_type=SpikeType.BINARY)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.spike_type == SpikeType.BINARY
        print("✓ SpikeTensor creation works")
        
        # Test memory estimation
        memory_est = tensor.estimate_memory()
        assert memory_est > 0
        print(f"✓ Memory estimation works: {memory_est} bytes")
        
        # Test MembraneState
        membrane = MembraneState(shape=(100,), tau_mem=10.0)
        assert membrane.shape == (100,)
        assert membrane.tau_mem == 10.0
        print("✓ MembraneState creation works")
        
        # Test SynapticWeights
        weights = SynapticWeights(shape=(784, 100), dtype="float32", sparsity=0.9)
        assert weights.shape == (784, 100)
        assert weights.sparsity == 0.9
        print("✓ SynapticWeights creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ IR types test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system."""
    try:
        from spike_transformer_compiler.config import CompilerConfig, get_compiler_config
        
        # Test default config
        config = get_compiler_config()
        assert hasattr(config, 'cache_directory')
        assert hasattr(config, 'optimization_level')
        print("✓ Configuration system works")
        
        # Test config creation
        custom_config = CompilerConfig(
            optimization_level=3,
            cache_directory="/tmp/spike_cache"
        )
        assert custom_config.optimization_level == 3
        print("✓ Custom configuration creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_exceptions():
    """Test exception system."""
    try:
        from spike_transformer_compiler.exceptions import (
            CompilationError, ValidationError, ValidationUtils, ErrorContext
        )
        
        # Test exception creation
        error = CompilationError("Test error", error_code="TEST_ERROR")
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        print("✓ Exception creation works")
        
        # Test validation utils
        try:
            ValidationUtils.validate_optimization_level(5)  # Invalid level
            print("✗ Validation should have failed")
            return False
        except ValidationError:
            print("✓ Validation correctly rejects invalid optimization level")
        
        # Test valid optimization level
        ValidationUtils.validate_optimization_level(2)  # Valid level
        print("✓ Validation correctly accepts valid optimization level")
        
        return True
        
    except Exception as e:
        print(f"✗ Exception test failed: {e}")
        traceback.print_exc()
        return False

def test_security_framework():
    """Test security framework without model dependencies."""
    try:
        from spike_transformer_compiler.security import SecurityConfig, get_security_config
        
        # Test security config
        sec_config = get_security_config()
        assert hasattr(sec_config, 'max_model_size_mb')
        assert hasattr(sec_config, 'allowed_file_extensions')
        print("✓ Security configuration works")
        
        # Test input sanitizer
        from spike_transformer_compiler.security import InputSanitizer
        sanitizer = InputSanitizer(sec_config)
        
        # Test input shape sanitization
        sanitized_shape = sanitizer.sanitize_input_shape((1, 3, 224, 224))
        assert sanitized_shape == (1, 3, 224, 224)
        print("✓ Input shape sanitization works")
        
        # Test compilation target sanitization
        target = sanitizer.sanitize_compilation_target("simulation")
        assert target == "simulation"
        print("✓ Compilation target sanitization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Security framework test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """Test performance monitoring without heavy dependencies."""
    try:
        from spike_transformer_compiler.performance import PerformanceProfiler, ResourceMonitor
        
        # Test profiler creation
        profiler = PerformanceProfiler()
        profiler.start_compilation_profiling()
        
        # Simulate some work
        time.sleep(0.1)
        
        profiler.end_compilation_profiling()
        stats = profiler.get_compilation_stats()
        
        assert 'total_time' in stats
        print("✓ Performance profiler works")
        
        # Test resource monitor
        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()
        
        assert isinstance(usage, dict)
        print("✓ Resource monitor works")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_scaling_infrastructure():
    """Test auto-scaling infrastructure."""
    try:
        from spike_transformer_compiler.scaling.resource_pool import ResourceRequest
        from spike_transformer_compiler.scaling.auto_scaler import ScalingMetrics
        
        # Test resource request
        request = ResourceRequest(
            resource_type="compilation_worker",
            cpu_cores=4,
            memory_gb=8
        )
        assert request.cpu_cores == 4
        print("✓ Resource request creation works")
        
        # Test scaling metrics
        metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_utilization=50.0,
            memory_utilization=60.0,
            queue_length=5,
            active_tasks=3,
            completed_tasks_per_minute=10.0,
            average_task_duration=30.0
        )
        assert metrics.cpu_utilization == 50.0
        print("✓ Scaling metrics creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Scaling infrastructure test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all available tests."""
    tests = [
        ("Basic Imports", test_basic_imports),
        ("IR Types", test_ir_types),
        ("Configuration", test_configuration),  
        ("Exceptions", test_exceptions),
        ("Security Framework", test_security_framework),
        ("Performance Monitoring", test_performance_monitoring),
        ("Scaling Infrastructure", test_scaling_infrastructure),
    ]
    
    print("🧪 Running Spike-Transformer-Compiler Test Suite")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {failed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)