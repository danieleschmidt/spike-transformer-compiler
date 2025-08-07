#!/usr/bin/env python3
"""Simple usage example for Spike-Transformer-Compiler."""

import torch
import torch.nn as nn
from spike_transformer_compiler import SpikeCompiler

def create_simple_cnn():
    """Create a simple CNN model for demonstration."""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def main():
    """Demonstrate basic compilation workflow."""
    print("=== Spike-Transformer-Compiler Simple Example ===\n")
    
    # Create a simple CNN model
    print("1. Creating simple CNN model...")
    model = create_simple_cnn()
    model.eval()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Initialize compiler
    print("\n2. Initializing compiler...")
    compiler = SpikeCompiler(
        target="simulation",  # Use simulation for this example
        optimization_level=2,
        time_steps=4,
        verbose=True
    )
    
    # Compile model
    print("\n3. Compiling model...")
    input_shape = (1, 1, 28, 28)  # MNIST-like input
    
    try:
        compiled_model = compiler.compile(
            model,
            input_shape=input_shape,
            profile_energy=True
        )
        
        print("✓ Compilation successful!")
        print(f"   Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
        print(f"   Hardware utilization: {compiled_model.utilization:.1%}")
        
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return
    
    # Test inference
    print("\n4. Testing inference...")
    test_input = torch.randn(1, 1, 28, 28)
    
    try:
        output = compiled_model.run(
            test_input,
            time_steps=4,
            return_spike_trains=False
        )
        
        print("✓ Inference successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Show debug trace if available
        if hasattr(compiled_model, 'debug_trace'):
            print("\n5. Debug trace:")
            compiled_model.debug_trace()
            
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()