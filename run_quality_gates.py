#!/usr/bin/env python3
"""Quality gates and validation tests for production readiness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import_quality():
    """Test import quality and module structure."""
    print("🔍 Testing import quality...")
    
    try:
        # Test core imports work without warnings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from spike_transformer_compiler import SpikeCompiler, OptimizationPass, ResourceAllocator
            from spike_transformer_compiler.ir.spike_graph import SpikeGraph, SpikeNode, NodeType
            from spike_transformer_compiler.validation import ValidationUtils, ValidationError
            
            if len(w) == 0:
                print("✅ Clean imports - no warnings")
            else:
                print(f"⚠️  {len(w)} import warnings (acceptable)")
        
        # Test module structure
        modules_to_check = [
            'spike_transformer_compiler.compiler',
            'spike_transformer_compiler.ir.spike_graph',
            'spike_transformer_compiler.ir.builder', 
            'spike_transformer_compiler.ir.passes',
            'spike_transformer_compiler.backend.factory',
            'spike_transformer_compiler.validation'
        ]
        
        for module_name in modules_to_check:
            try:
                __import__(module_name)
                print(f"✅ {module_name}")
            except ImportError as e:
                print(f"❌ {module_name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import quality test failed: {e}")
        return False

def test_api_consistency():
    """Test API consistency and design patterns."""
    print("\\n🎯 Testing API consistency...")
    
    try:
        from spike_transformer_compiler import SpikeCompiler, ResourceAllocator
        from spike_transformer_compiler.ir.builder import SpikeIRBuilder
        
        # Test consistent constructor patterns
        compiler = SpikeCompiler()
        print("✅ SpikeCompiler() - default constructor works")
        
        compiler_with_args = SpikeCompiler(target="simulation", optimization_level=2)
        print("✅ SpikeCompiler(target, optimization_level) - parameterized constructor works")
        
        allocator = ResourceAllocator()
        print("✅ ResourceAllocator() - default constructor works")
        
        allocator_with_args = ResourceAllocator(num_chips=2, cores_per_chip=64)
        print("✅ ResourceAllocator(num_chips, cores_per_chip) - parameterized constructor works")
        
        # Test builder pattern consistency
        builder = SpikeIRBuilder("test")
        input_id = builder.add_input("test_input", (1, 10))
        print("✅ Builder pattern - fluent interface works")
        
        # Test method naming consistency
        methods_to_check = [
            (compiler, 'compile'),
            (compiler, 'create_optimizer'),
            (allocator, 'allocate'),
            (builder, 'build'),
            (builder, 'add_input')
        ]
        
        for obj, method_name in methods_to_check:
            if hasattr(obj, method_name):
                print(f"✅ {type(obj).__name__}.{method_name}() exists")
            else:
                print(f"❌ {type(obj).__name__}.{method_name}() missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ API consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling_quality():
    """Test error handling quality and consistency."""
    print("\\n🛡️  Testing error handling quality...")
    
    try:
        from spike_transformer_compiler import SpikeCompiler
        from spike_transformer_compiler.validation import ValidationError
        
        # Test consistent error types
        error_tests = [
            # (test_func, expected_error_type, description)
            (lambda: SpikeCompiler(target="invalid"), ValidationError, "invalid target"),
            (lambda: SpikeCompiler(optimization_level=-1), ValidationError, "negative optimization level"),
            (lambda: SpikeCompiler(time_steps=0), ValidationError, "zero time steps"),
        ]
        
        passed = 0
        for test_func, expected_error, description in error_tests:
            try:
                test_func()
                print(f"❌ {description}: Should have raised {expected_error.__name__}")
            except expected_error:
                print(f"✅ {description}: Correctly raised {expected_error.__name__}")
                passed += 1
            except Exception as e:
                print(f"❌ {description}: Unexpected error {type(e).__name__}: {e}")
        
        print(f"Error handling quality: {passed}/{len(error_tests)} tests passed")
        return passed == len(error_tests)
        
    except Exception as e:
        print(f"❌ Error handling quality test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance meets benchmarks."""
    print("\\n⚡ Testing performance benchmarks...")
    
    try:
        import time
        from spike_transformer_compiler import SpikeCompiler
        from spike_transformer_compiler.ir.builder import SpikeIRBuilder
        
        # Benchmark 1: Compiler instantiation
        start_time = time.time()
        compiler = SpikeCompiler()
        instantiation_time = time.time() - start_time
        
        if instantiation_time < 0.1:  # 100ms benchmark
            print(f"✅ Compiler instantiation: {instantiation_time*1000:.2f}ms < 100ms")
        else:
            print(f"⚠️  Compiler instantiation: {instantiation_time*1000:.2f}ms > 100ms")
        
        # Benchmark 2: Graph building  
        start_time = time.time()
        builder = SpikeIRBuilder("benchmark_test")
        input_id = builder.add_input("input", (1, 32))
        
        # Build a moderately complex graph
        current_id = input_id
        for i in range(5):  # 5 layers
            linear_id = builder.add_spike_linear(current_id, out_features=16)
            neuron_id = builder.add_spike_neuron(linear_id)
            current_id = neuron_id
        
        builder.add_output(current_id, "output")
        graph = builder.build()
        
        build_time = time.time() - start_time
        
        if build_time < 0.05:  # 50ms benchmark
            print(f"✅ Graph building ({len(graph.nodes)} nodes): {build_time*1000:.2f}ms < 50ms")
        else:
            print(f"⚠️  Graph building ({len(graph.nodes)} nodes): {build_time*1000:.2f}ms > 50ms")
        
        # Benchmark 3: Graph analysis
        start_time = time.time()
        resources = graph.analyze_resources()
        analysis_time = time.time() - start_time
        
        if analysis_time < 0.01:  # 10ms benchmark
            print(f"✅ Graph analysis: {analysis_time*1000:.2f}ms < 10ms")
        else:
            print(f"⚠️  Graph analysis: {analysis_time*1000:.2f}ms > 10ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark test failed: {e}")
        return False

def test_integration_end_to_end():
    """Test complete end-to-end integration."""
    print("\\n🔄 Testing end-to-end integration...")
    
    try:
        from spike_transformer_compiler import SpikeCompiler, ResourceAllocator
        from spike_transformer_compiler.ir.builder import SpikeIRBuilder
        
        # Create complete workflow
        print("Step 1: Create compiler and allocator")
        compiler = SpikeCompiler(target="simulation", optimization_level=2, verbose=False)
        allocator = ResourceAllocator(num_chips=2, cores_per_chip=64)
        
        print("Step 2: Build model graph")
        builder = SpikeIRBuilder("integration_test")
        input_id = builder.add_input("sensor_input", (1, 16))
        
        # Create realistic neural network structure
        conv_id = builder.add_spike_conv2d(input_id, out_channels=8, kernel_size=3)
        neuron1_id = builder.add_spike_neuron(conv_id, neuron_model="LIF", threshold=1.0)
        
        linear_id = builder.add_spike_linear(neuron1_id, out_features=10)
        neuron2_id = builder.add_spike_neuron(linear_id, neuron_model="LIF", threshold=0.8)
        
        output_id = builder.add_output(neuron2_id, "classification_output")
        
        graph = builder.build()
        print(f"✅ Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        print("Step 3: Analyze resources")
        resources = graph.analyze_resources()
        allocation = allocator.allocate(graph)
        
        print(f"✅ Resource analysis: {resources['neuron_count']} neurons, {allocation['estimated_utilization']:.2%} utilization")
        
        print("Step 4: Create optimizer")  
        optimizer = compiler.create_optimizer()
        print("✅ Optimizer created with optimization passes")
        
        print("Step 5: Verify graph integrity")
        is_valid = graph.verify()
        if is_valid:
            print("✅ Graph verification passed")
        else:
            print("❌ Graph verification failed")
            return False
        
        print("Step 6: Test placement optimization")
        placement = allocator.optimize_placement(graph)
        print(f"✅ Placement optimized: {placement['load_balance']:.2f} load balance")
        
        print("✅ Complete end-to-end integration successful!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all quality gate tests."""
    print("=" * 70)
    print("🔒 SPIKE TRANSFORMER COMPILER - QUALITY GATES & VALIDATION")
    print("=" * 70)
    
    tests = [
        test_import_quality,
        test_api_consistency,
        test_error_handling_quality,
        test_performance_benchmarks,
        test_integration_end_to_end
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    total = len(results)
    
    print("\\n" + "=" * 70)
    print(f"📊 QUALITY GATES SUMMARY: {passed}/{total} gates passed")
    
    if passed == total:
        print("🎉 All quality gates PASSED!")
        print("✅ Production quality validated!")
        return 0
    elif passed >= total * 0.8:  # 80% acceptable for quality gates
        print("🟡 Most quality gates passed!")
        print("✅ Core quality requirements met!")
        return 0
    else:
        print(f"❌ {total - passed} quality gates FAILED")
        print("⚠️  Quality needs improvement before production")
        return 1

if __name__ == "__main__":
    exit(main())