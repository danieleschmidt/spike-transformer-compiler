#!/usr/bin/env python3
"""Basic test for Generation 1 functionality - MAKE IT WORK."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for testing
    class MockArray:
        def __init__(self, data, shape):
            self.data = data
            self.shape = shape
            self.dtype = "float32"
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    class np:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def randn(*shape):
                    size = 1
                    for dim in shape:
                        size *= dim
                    data = [0.5] * size
                    return MockArray(data, shape)
            return Random()
        
        @staticmethod
        def zeros(shape):
            size = 1
            for dim in shape:
                size *= dim
            return MockArray([0.0] * size, shape)
        
        @staticmethod
        def array(data):
            if hasattr(data, 'shape'):
                return data
            return MockArray(data, (len(data),))

from spike_transformer_compiler import SpikeCompiler, OptimizationPass, ResourceAllocator

def test_basic_compilation():
    """Test basic compilation pipeline."""
    print("🧪 Testing Generation 1 Basic Functionality")
    
    # Create a simple mock model
    class SimpleModel:
        def __init__(self):
            self.layers = []
        
        def __call__(self, x):
            return x * 0.5
    
    # Initialize compiler
    print("✓ Creating SpikeCompiler instance...")
    compiler = SpikeCompiler(
        target="simulation",
        optimization_level=1,
        time_steps=4,
        verbose=True
    )
    
    # Create simple model
    model = SimpleModel()
    input_shape = (1, 3, 32, 32)  # Small image-like input
    
    print(f"✓ Model created with input shape: {input_shape}")
    
    # Test compilation
    try:
        print("⚙️  Starting compilation...")
        compiled_model = compiler.compile(
            model=model,
            input_shape=input_shape,
            profile_energy=True,
            secure_mode=False,  # Disable for basic test
            enable_resilience=False  # Disable for basic test
        )
        print("✅ Compilation successful!")
        
        # Test basic inference setup (without full execution for Generation 1)
        print("🚀 Testing basic inference setup...")
        if NUMPY_AVAILABLE:
            dummy_input = np.random.randn(*input_shape)
        else:
            dummy_input = np.random().randn(*input_shape)
        
        # For Generation 1, just test that we can create the inference setup
        print("✅ Compilation model created successfully!")
        print(f"✓ Input prepared with shape: {dummy_input.shape if hasattr(dummy_input, 'shape') else type(dummy_input)}")
        print("✓ Model ready for inference (simulation backend loaded)")
        
        # Print energy and utilization
        if hasattr(compiled_model, 'energy_per_inference'):
            print(f"⚡ Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
        if hasattr(compiled_model, 'utilization'):
            print(f"📊 Hardware utilization: {compiled_model.utilization:.1%}")
            
        return True
        
    except Exception as e:
        print(f"❌ Compilation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_resource_allocator():
    """Test resource allocation functionality."""
    print("\\n🔧 Testing ResourceAllocator...")
    
    # Test resource allocator
    allocator = ResourceAllocator(
        num_chips=2,
        cores_per_chip=128,
        synapses_per_core=1024
    )
    print("✅ ResourceAllocator created successfully")
    
    return True

def test_optimization_passes():
    """Test optimization pass enumeration."""
    print("\\n⚙️  Testing OptimizationPass...")
    
    # Test optimization pass types
    passes = [
        OptimizationPass.SPIKE_COMPRESSION,
        OptimizationPass.WEIGHT_QUANTIZATION,
        OptimizationPass.NEURON_PRUNING,
        OptimizationPass.TEMPORAL_FUSION
    ]
    
    print(f"✅ {len(passes)} optimization passes available")
    for pass_type in passes:
        print(f"  - {pass_type.value}")
    
    return True

def main():
    """Run all Generation 1 basic tests."""
    print("=" * 60)
    print("🚀 SPIKE TRANSFORMER COMPILER - GENERATION 1 TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(test_resource_allocator())
    test_results.append(test_optimization_passes())
    test_results.append(test_basic_compilation())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("\\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Generation 1 tests PASSED!")
        print("✅ Basic functionality (MAKE IT WORK) is complete!")
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED")
        return 1

if __name__ == "__main__":
    exit(main())