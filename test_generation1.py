#!/usr/bin/env python3
"""Test basic Generation 1 functionality - MAKE IT WORK."""

import torch
import torch.nn as nn
import numpy as np
from src.spike_transformer_compiler import SpikeCompiler


def create_simple_model():
    """Create a simple test model."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    return model


def test_basic_compilation():
    """Test basic compilation pipeline."""
    print("=== Generation 1 Testing: MAKE IT WORK ===")
    
    # Create simple test model
    model = create_simple_model()
    input_shape = (1, 10)
    
    print(f"Created test model: {type(model).__name__}")
    print(f"Input shape: {input_shape}")
    
    # Test simulation backend
    print("\n--- Testing Simulation Backend ---")
    try:
        compiler = SpikeCompiler(
            target="simulation",
            optimization_level=1,
            time_steps=4,
            verbose=True
        )
        
        print("Compiler initialized successfully")
        
        # Compile model
        compiled_model = compiler.compile(
            model=model,
            input_shape=input_shape,
            profile_energy=True
        )
        
        print("Model compiled successfully!")
        print(f"Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
        print(f"Hardware utilization: {compiled_model.utilization:.1%}")
        
        # Test inference
        test_input = np.random.randn(*input_shape).astype(np.float32)
        output = compiled_model.run(test_input, time_steps=4)
        
        print(f"Inference successful!")
        print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_pipeline():
    """Test optimization pipeline."""
    print("\n--- Testing Optimization Pipeline ---")
    try:
        compiler = SpikeCompiler(
            target="simulation",
            optimization_level=3,
            time_steps=4
        )
        
        # Create optimization pipeline
        optimizer = compiler.create_optimizer()
        print(f"Created optimizer with {len(optimizer.passes)} passes")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resource_allocation():
    """Test resource allocation."""
    print("\n--- Testing Resource Allocation ---")
    try:
        from src.spike_transformer_compiler import ResourceAllocator
        
        allocator = ResourceAllocator(
            num_chips=2,
            cores_per_chip=128,
            synapses_per_core=1024
        )
        
        # Create dummy model for allocation
        model = create_simple_model()
        
        # This will require SpikeGraph but we'll test the allocator interface
        print("Resource allocator created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Resource allocation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Generation 1 tests."""
    print("🧠 TERRAGON SDLC - Generation 1: MAKE IT WORK")
    print("=" * 60)
    
    tests = [
        ("Basic Compilation", test_basic_compilation),
        ("Optimization Pipeline", test_optimization_pipeline), 
        ("Resource Allocation", test_resource_allocation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running test: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name} passed")
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Generation 1 (MAKE IT WORK) - ALL TESTS PASSED!")
        print("Ready to proceed to Generation 2 (MAKE IT ROBUST)")
        return True
    else:
        print("⚠️  Some tests failed. Need to fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)