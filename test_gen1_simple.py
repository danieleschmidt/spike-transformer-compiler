#!/usr/bin/env python3
"""Simplified basic test for Generation 1 functionality - no external dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from spike_transformer_compiler import SpikeCompiler, OptimizationPass, ResourceAllocator
        print("✅ Core imports successful")
        
        from spike_transformer_compiler.compiler import CompiledModel
        print("✅ Compiler classes imported")
        
        from spike_transformer_compiler.ir.spike_graph import SpikeGraph, SpikeNode, NodeType
        print("✅ IR classes imported")
        
        from spike_transformer_compiler.validation import ValidationUtils, ValidationError
        print("✅ Validation classes imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_instantiation():
    """Test basic object creation."""
    print("\\n🔧 Testing basic instantiation...")
    
    try:
        from spike_transformer_compiler import SpikeCompiler, OptimizationPass, ResourceAllocator
        
        # Test compiler creation
        compiler = SpikeCompiler(
            target="simulation",
            optimization_level=1,
            time_steps=4
        )
        print("✅ SpikeCompiler created successfully")
        
        # Test resource allocator
        allocator = ResourceAllocator(num_chips=1, cores_per_chip=64)
        print("✅ ResourceAllocator created successfully")
        
        # Test optimization passes
        passes = [
            OptimizationPass.SPIKE_COMPRESSION,
            OptimizationPass.WEIGHT_QUANTIZATION,
            OptimizationPass.NEURON_PRUNING,
            OptimizationPass.TEMPORAL_FUSION
        ]
        print(f"✅ {len(passes)} optimization passes available")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ir_components():
    """Test IR graph components."""
    print("\\n📊 Testing IR components...")
    
    try:
        from spike_transformer_compiler.ir.spike_graph import SpikeGraph, SpikeNode, NodeType, SpikeEdge
        from spike_transformer_compiler.ir.builder import SpikeIRBuilder
        from spike_transformer_compiler.ir.types import SpikeTensor, SpikeType
        
        # Test graph creation
        graph = SpikeGraph("test_graph")
        print("✅ SpikeGraph created")
        
        # Test node creation
        node = SpikeNode(
            id="test_node",
            node_type=NodeType.INPUT,
            operation="test_op",
            inputs=[],
            outputs=["test_out"],
            parameters={"test": "value"},
            metadata={}
        )
        graph.add_node(node)
        print("✅ SpikeNode added to graph")
        
        # Test builder
        builder = SpikeIRBuilder("test_build")
        print("✅ SpikeIRBuilder created")
        
        return True
        
    except Exception as e:
        print(f"❌ IR test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_validation():
    """Test validation utilities."""
    print("\\n🔍 Testing validation...")
    
    try:
        from spike_transformer_compiler.validation import ValidationUtils, ValidationError
        
        # Test valid inputs
        ValidationUtils.validate_target("simulation", ["simulation", "loihi3"])
        print("✅ Target validation works")
        
        ValidationUtils.validate_optimization_level(2)
        print("✅ Optimization level validation works")
        
        ValidationUtils.validate_time_steps(4)
        print("✅ Time steps validation works")
        
        ValidationUtils.validate_input_shape((1, 3, 32, 32))
        print("✅ Input shape validation works")
        
        # Test error cases
        try:
            ValidationUtils.validate_target("invalid", ["simulation"])
            print("❌ Should have raised ValidationError for invalid target")
            return False
        except ValidationError:
            print("✅ Correctly rejected invalid target")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("=" * 60)
    print("🚀 SPIKE TRANSFORMER COMPILER - GENERATION 1 SIMPLE TEST")
    print("=" * 60)
    
    test_functions = [
        test_imports,
        test_basic_instantiation, 
        test_ir_components,
        test_validation
    ]
    
    results = []
    for test_func in test_functions:
        results.append(test_func())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Generation 1 simple tests PASSED!")
        print("✅ Basic functionality (MAKE IT WORK) core is complete!")
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED")
        return 1

if __name__ == "__main__":
    exit(main())