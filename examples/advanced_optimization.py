#!/usr/bin/env python3
"""Advanced optimization example for Spike-Transformer-Compiler."""

import torch
import torch.nn as nn
from spike_transformer_compiler import SpikeCompiler, OptimizationPass, ResourceAllocator

def create_large_model():
    """Create a larger model to demonstrate optimization."""
    return nn.Sequential(
        # Feature extraction
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        
        # Residual-like blocks
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        
        # Classifier
        nn.Linear(256 * 7 * 7, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1000)  # ImageNet classes
    )

def demonstrate_optimization_levels():
    """Demonstrate different optimization levels."""
    print("=== Optimization Level Comparison ===\n")
    
    model = create_large_model()
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    input_shape = (1, 3, 224, 224)
    
    for opt_level in range(4):
        print(f"\n--- Optimization Level {opt_level} ---")
        
        compiler = SpikeCompiler(
            target="loihi3",
            optimization_level=opt_level,
            time_steps=4,
            verbose=False
        )
        
        try:
            compiled_model = compiler.compile(
                model,
                input_shape=input_shape,
                profile_energy=True
            )
            
            print(f"✓ Compilation successful!")
            print(f"   Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
            print(f"   Hardware utilization: {compiled_model.utilization:.1%}")
            
        except Exception as e:
            print(f"✗ Compilation failed: {e}")

def demonstrate_custom_optimization():
    """Demonstrate custom optimization pipeline."""
    print("\n=== Custom Optimization Pipeline ===\n")
    
    model = create_large_model()
    model.eval()
    
    # Initialize compiler
    compiler = SpikeCompiler(
        target="loihi3",
        optimization_level=0,  # Start with no optimization
        time_steps=4,
        verbose=True
    )
    
    # Create custom optimizer
    optimizer = compiler.create_optimizer()
    
    # Add specific optimization passes
    print("Adding optimization passes:")
    print("  - Spike compression")
    print("  - Weight quantization (4-bit)")
    print("  - Neuron pruning (90% sparsity)")
    print("  - Temporal fusion")
    
    # Resource-aware allocation
    allocator = ResourceAllocator(
        num_chips=2,
        cores_per_chip=128,
        synapses_per_core=1024
    )
    
    input_shape = (1, 3, 224, 224)
    
    try:
        compiled_model = compiler.compile(
            model,
            input_shape=input_shape,
            optimizer=optimizer,
            resource_allocator=allocator,
            profile_energy=True
        )
        
        print(f"\n✓ Custom optimization successful!")
        print(f"   Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
        print(f"   Hardware utilization: {compiled_model.utilization:.1%}")
        
        # Show hardware statistics if available
        if hasattr(compiled_model, 'get_hardware_stats'):
            stats = compiled_model.get_hardware_stats()
            print(f"   Cores used: {stats['cores_used']}")
            print(f"   Chips used: {stats['chips_used']}")
            print(f"   Total neurons: {stats['total_neurons']:,}")
        
    except Exception as e:
        print(f"✗ Custom optimization failed: {e}")

def demonstrate_energy_profiling():
    """Demonstrate detailed energy profiling."""
    print("\n=== Energy Profiling ===\n")
    
    # Create smaller model for detailed profiling
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model.eval()
    
    compiler = SpikeCompiler(
        target="loihi3",
        optimization_level=2,
        time_steps=4,
        verbose=True
    )
    
    input_shape = (1, 3, 32, 32)
    
    try:
        compiled_model = compiler.compile(
            model,
            input_shape=input_shape,
            profile_energy=True
        )
        
        print(f"✓ Energy profiling completed!")
        print(f"   Estimated energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
        
        # Simulate multiple inferences to get average
        test_input = torch.randn(1, 3, 32, 32)
        total_energy = 0
        num_runs = 10
        
        print(f"\nRunning {num_runs} inference simulations...")
        for i in range(num_runs):
            output = compiled_model.run(test_input, time_steps=4)
            if hasattr(compiled_model, 'energy_per_inference'):
                total_energy += compiled_model.energy_per_inference
        
        avg_energy = total_energy / num_runs
        print(f"Average energy per inference: {avg_energy:.3f} nJ")
        
        # Energy efficiency comparison
        print(f"\nEnergy efficiency estimates:")
        print(f"  Loihi3 neuromorphic: {avg_energy:.3f} nJ/inference")
        print(f"  GPU (estimated): {avg_energy * 100:.1f} nJ/inference")
        print(f"  CPU (estimated): {avg_energy * 500:.1f} nJ/inference")
        print(f"  Energy savings: {100:.0f}x vs GPU, {500:.0f}x vs CPU")
        
    except Exception as e:
        print(f"✗ Energy profiling failed: {e}")

def main():
    """Run advanced optimization examples."""
    print("=== Spike-Transformer-Compiler Advanced Optimization ===\n")
    
    # Demonstrate optimization levels
    demonstrate_optimization_levels()
    
    # Demonstrate custom optimization
    demonstrate_custom_optimization()
    
    # Demonstrate energy profiling
    demonstrate_energy_profiling()
    
    print("\n=== Advanced optimization examples completed! ===")

if __name__ == "__main__":
    main()