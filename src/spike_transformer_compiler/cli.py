"""Command-line interface for Spike-Transformer-Compiler."""

import click
import torch
import os
import sys
import json
from pathlib import Path
from typing import Optional
from . import __version__
from .compiler import SpikeCompiler, CompilationError
from .backend.factory import BackendFactory
from .frontend.pytorch_parser import PyTorchParser


@click.group()
@click.version_option(version=__version__)
def main():
    """Spike-Transformer-Compiler: Neuromorphic compilation for SpikeFormers."""
    pass


@main.command()
@click.argument("model_path")
@click.option("--target", default="loihi3", help="Target hardware platform")
@click.option("--output", "-o", help="Output directory for compiled model")
@click.option("--input-shape", required=True, help="Input shape as comma-separated values (e.g. 1,3,224,224)")
@click.option("--optimization-level", "-O", default=2, help="Optimization level (0-3)")
@click.option("--time-steps", default=4, help="Number of time steps")
@click.option("--chip-config", help="Hardware chip configuration")
@click.option("--profile-energy", is_flag=True, help="Enable energy profiling")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def compile(model_path, target, output, input_shape, optimization_level, time_steps, 
           chip_config, profile_energy, verbose, debug):
    """Compile a PyTorch model to neuromorphic hardware."""
    try:
        if verbose:
            click.echo(f"Spike-Transformer-Compiler v{__version__}")
            click.echo(f"Compiling {model_path} for {target} target...")
            
        # Parse input shape
        try:
            shape = tuple(map(int, input_shape.split(',')))
        except ValueError:
            click.echo("Error: Invalid input shape format. Use comma-separated integers (e.g. 1,3,224,224)", err=True)
            return 1
            
        # Load PyTorch model
        if not os.path.exists(model_path):
            click.echo(f"Error: Model file not found: {model_path}", err=True)
            return 1
            
        try:
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'eval'):
                model.eval()
        except Exception as e:
            click.echo(f"Error loading model: {e}", err=True)
            return 1
            
        # Initialize compiler
        compiler = SpikeCompiler(
            target=target,
            optimization_level=optimization_level,
            time_steps=time_steps,
            debug=debug,
            verbose=verbose
        )
        
        # Compile model
        compiled_model = compiler.compile(
            model,
            input_shape=shape,
            chip_config=chip_config,
            profile_energy=profile_energy
        )
        
        # Save compiled model
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model metadata
            metadata = {
                "model_path": model_path,
                "target": target,
                "input_shape": shape,
                "optimization_level": optimization_level,
                "time_steps": time_steps,
                "energy_per_inference": getattr(compiled_model, 'energy_per_inference', 0.0),
                "utilization": getattr(compiled_model, 'utilization', 0.0),
                "compiler_version": __version__
            }
            
            with open(output_dir / "compilation_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            click.echo(f"Compiled model saved to {output_dir}")
        
        # Show compilation results
        if verbose or not output:
            click.echo("\n=== Compilation Results ===")
            click.echo(f"Target: {target}")
            click.echo(f"Input shape: {shape}")
            click.echo(f"Time steps: {time_steps}")
            click.echo(f"Optimization level: {optimization_level}")
            
            if hasattr(compiled_model, 'energy_per_inference'):
                click.echo(f"Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
            if hasattr(compiled_model, 'utilization'):
                click.echo(f"Hardware utilization: {compiled_model.utilization:.1%}")
                
        click.echo("Compilation completed successfully!")
        return 0
        
    except CompilationError as e:
        click.echo(f"Compilation failed: {e}", err=True)
        return 1
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
@click.argument("model_path")
@click.option("--input-shape", required=True, help="Input shape as comma-separated values")
@click.option("--show-params", is_flag=True, help="Show model parameters")
@click.option("--show-complexity", is_flag=True, help="Show computational complexity")
@click.option("--detect-transformers", is_flag=True, help="Detect transformer blocks")
def analyze(model_path, input_shape, show_params, show_complexity, detect_transformers):
    """Analyze a PyTorch model for neuromorphic compilation."""
    try:
        # Parse input shape
        try:
            shape = tuple(map(int, input_shape.split(',')))
        except ValueError:
            click.echo("Error: Invalid input shape format. Use comma-separated integers", err=True)
            return 1
            
        # Load model
        if not os.path.exists(model_path):
            click.echo(f"Error: Model file not found: {model_path}", err=True)
            return 1
            
        model = torch.load(model_path, map_location='cpu')
        if hasattr(model, 'eval'):
            model.eval()
            
        parser = PyTorchParser()
        
        click.echo(f"=== Model Analysis: {model_path} ===")
        click.echo(f"Model type: {type(model).__name__}")
        
        # Basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        click.echo(f"Total parameters: {total_params:,}")
        click.echo(f"Trainable parameters: {trainable_params:,}")
        
        # Show detailed parameters
        if show_params:
            click.echo("\n=== Model Parameters ===")
            params = parser.extract_parameters(model)
            for name, info in params.items():
                if name != "total_parameters":
                    click.echo(f"{name}: {info['shape']} ({info['param_count']:,} params)")
                    
        # Show computational complexity
        if show_complexity:
            click.echo("\n=== Computational Complexity ===")
            try:
                complexity = parser.estimate_model_complexity(model, shape)
                click.echo(f"Convolution operations: {complexity['conv_operations']:,}")
                click.echo(f"Linear operations: {complexity['linear_operations']:,}")
                click.echo(f"Activation operations: {complexity['activation_operations']:,}")
                click.echo(f"Total operations: {complexity['total_operations']:,}")
            except Exception as e:
                click.echo(f"Could not estimate complexity: {e}")
                
        # Detect transformer blocks
        if detect_transformers:
            click.echo("\n=== Transformer Block Detection ===")
            try:
                blocks = parser.detect_transformer_blocks(model)
                if blocks:
                    click.echo(f"Found {len(blocks)} transformer-like blocks:")
                    for block in blocks:
                        click.echo(f"  - {block['name']}: attention={block['has_attention']}, "
                                 f"feedforward={block['has_feedforward']}, norm={block['has_norm']}")
                else:
                    click.echo("No transformer blocks detected")
            except Exception as e:
                click.echo(f"Could not detect transformer blocks: {e}")
                
        return 0
        
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)
        return 1


@main.command()
def info():
    """Show compiler and hardware information."""
    click.echo(f"Spike-Transformer-Compiler v{__version__}")
    click.echo(f"PyTorch version: {torch.__version__}")
    
    # Show available targets
    available_targets = BackendFactory.get_available_targets()
    click.echo(f"Available targets: {', '.join(available_targets)}")
    
    # Show supported layer types
    parser = PyTorchParser()
    supported_layers = list(parser.supported_layers.keys())
    click.echo(f"Supported PyTorch layers: {len(supported_layers)}")
    for layer_type in supported_layers:
        click.echo(f"  - {layer_type.__name__}")
    
    # Hardware-specific info
    click.echo("\nTarget Information:")
    click.echo("  loihi3: Intel Loihi 3 neuromorphic processor")
    click.echo("  simulation: Software simulation backend")


@main.command()
@click.argument("model_path")
@click.argument("input_data")
@click.option("--compiled-model", required=True, help="Path to compiled model metadata")
@click.option("--time-steps", default=4, help="Number of time steps")
@click.option("--return-spikes", is_flag=True, help="Return spike trains")
@click.option("--benchmark", is_flag=True, help="Run performance benchmark")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(model_path, input_data, compiled_model, time_steps, return_spikes, benchmark, verbose):
    """Run inference on a compiled neuromorphic model."""
    try:
        # Load compilation metadata
        if not os.path.exists(compiled_model):
            click.echo(f"Error: Compiled model metadata not found: {compiled_model}", err=True)
            return 1
            
        with open(compiled_model, 'r') as f:
            metadata = json.load(f)
            
        # Load input data
        if input_data.endswith('.pt') or input_data.endswith('.pth'):
            input_tensor = torch.load(input_data, map_location='cpu')
        elif input_data.endswith('.npy'):
            import numpy as np
            input_tensor = torch.from_numpy(np.load(input_data))
        else:
            click.echo("Error: Unsupported input data format. Use .pt, .pth, or .npy files", err=True)
            return 1
            
        # Recreate compiler with same settings
        compiler = SpikeCompiler(
            target=metadata['target'],
            optimization_level=metadata['optimization_level'],
            time_steps=metadata['time_steps'],
            verbose=verbose
        )
        
        # Load and recompile model (in a real system, we'd save/load compiled binaries)
        model = torch.load(model_path, map_location='cpu')
        compiled_model = compiler.compile(
            model,
            input_shape=tuple(metadata['input_shape']),
        )
        
        if verbose:
            click.echo(f"Running inference with {time_steps} time steps...")
            
        # Run inference
        import time
        if benchmark:
            # Warmup
            _ = compiled_model.run(input_tensor, time_steps=time_steps)
            
            # Benchmark
            start_time = time.time()
            result = compiled_model.run(
                input_tensor, 
                time_steps=time_steps,
                return_spike_trains=return_spikes
            )
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            click.echo(f"Inference time: {inference_time:.2f} ms")
        else:
            result = compiled_model.run(
                input_tensor, 
                time_steps=time_steps,
                return_spike_trains=return_spikes
            )
            
        if verbose:
            click.echo(f"Output shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            if hasattr(result, 'dtype'):
                click.echo(f"Output dtype: {result.dtype}")
                
        click.echo("Inference completed successfully!")
        return 0
        
    except Exception as e:
        click.echo(f"Inference failed: {e}", err=True)
        return 1


@main.command()
@click.option("--enable-research", is_flag=True, help="Enable research hypothesis-driven development")
@click.option("--global-deployment", is_flag=True, default=True, help="Enable global-first implementation")
@click.option("--adaptive-learning", is_flag=True, default=True, help="Enable self-improving patterns")
def autonomous(enable_research, global_deployment, adaptive_learning):
    """Execute autonomous SDLC cycle with progressive enhancement."""
    try:
        from .autonomous_executor import run_autonomous_execution
        
        click.echo("🚀 Starting Autonomous SDLC Execution")
        click.echo("=" * 50)
        click.echo(f"Research Mode: {'✓ ENABLED' if enable_research else '✗ DISABLED'}")
        click.echo(f"Global Deployment: {'✓ ENABLED' if global_deployment else '✗ DISABLED'}")
        click.echo(f"Adaptive Learning: {'✓ ENABLED' if adaptive_learning else '✗ DISABLED'}")
        click.echo("=" * 50)
        
        # Execute autonomous SDLC
        report = run_autonomous_execution(
            target_config={
                "enable_research": enable_research,
                "global_deployment": global_deployment,
                "adaptive_learning": adaptive_learning
            }
        )
        
        # Display results
        click.echo("\n🎯 AUTONOMOUS EXECUTION COMPLETE")
        click.echo("=" * 50)
        click.echo(f"Execution ID: {report['execution_summary']['execution_id']}")
        click.echo(f"Success: {'✓ YES' if report['execution_summary']['success'] else '✗ NO'}")
        click.echo(f"Duration: {report['execution_summary']['duration_seconds']:.2f} seconds")
        click.echo(f"Generations: {report['execution_summary']['generations_completed']}/3")
        click.echo(f"Quality Gates: {report['execution_summary']['quality_gates_passed']}")
        
        click.echo("\n📈 Progressive Enhancement:")
        for generation, status in report['progressive_enhancement'].items():
            click.echo(f"  {generation}: {status}")
        
        click.echo("\n🔧 Adaptive Patterns Applied:")
        for pattern in report['adaptive_patterns']:
            click.echo(f"  {pattern['name']} ({pattern['type']}): "
                     f"used {pattern['usage_count']} times, "
                     f"effectiveness {pattern['effectiveness']:.2f}")
        
        if report['global_deployment']['multi_region_ready']:
            click.echo("\n🌍 Global Deployment Ready:")
            click.echo(f"  I18n Support: {', '.join(report['global_deployment']['i18n_support'])}")
            click.echo(f"  Compliance: {', '.join(report['global_deployment']['compliance_validated'])}")
        
        if report['research_capabilities']['research_mode_enabled']:
            click.echo("\n🔬 Research Capabilities:")
            click.echo("  ✓ Hypothesis-driven development")
            click.echo("  ✓ Statistical validation framework")
            click.echo("  ✓ Reproducible experimental setup")
        
        click.echo("\n🚀 Production Readiness:")
        readiness = report['production_readiness']
        for component, ready in readiness.items():
            status = "✓ READY" if ready else "✗ NOT READY"
            click.echo(f"  {component.replace('_', ' ').title()}: {status}")
        
        click.echo("\n🎉 Autonomous SDLC execution completed successfully!")
        return 0
        
    except Exception as e:
        click.echo(f"Autonomous execution failed: {e}", err=True)
        return 1


@main.command()
def list_examples():
    """List available example models and configurations."""
    click.echo("Available example configurations:")
    
    examples = [
        {
            "name": "spikeformer_tiny",
            "description": "Tiny SpikeFormer for CIFAR-10",
            "input_shape": "1,3,32,32",
            "target": "loihi3"
        },
        {
            "name": "spikeformer_base", 
            "description": "Base SpikeFormer for ImageNet",
            "input_shape": "1,3,224,224",
            "target": "loihi3"
        },
        {
            "name": "simple_cnn",
            "description": "Simple CNN for demonstration",
            "input_shape": "1,1,28,28", 
            "target": "simulation"
        },
        {
            "name": "autonomous_demo",
            "description": "Autonomous compilation demonstration",
            "input_shape": "1,3,224,224",
            "target": "simulation"
        }
    ]
    
    for example in examples:
        click.echo(f"\n{example['name']}:")
        click.echo(f"  Description: {example['description']}")
        click.echo(f"  Input shape: {example['input_shape']}")
        click.echo(f"  Recommended target: {example['target']}")
        click.echo(f"  Usage: spike-compile <model.pth> --input-shape {example['input_shape']} --target {example['target']}")
    
    click.echo("\n🤖 Autonomous SDLC:")
    click.echo("  Description: Execute complete autonomous software development lifecycle")
    click.echo("  Usage: spike-compile autonomous [--enable-research] [--global-deployment] [--adaptive-learning]")


if __name__ == "__main__":
    main()