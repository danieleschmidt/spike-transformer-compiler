#!/usr/bin/env python3
"""CLI demonstration script for Spike-Transformer-Compiler."""

import torch
import torch.nn as nn
import tempfile
import os
import subprocess
import sys
from pathlib import Path

def create_demo_model():
    """Create a simple demo model."""
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def main():
    """Demonstrate CLI functionality."""
    print("=== Spike-Transformer-Compiler CLI Demo ===\n")
    
    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create and save demo model
        print("1. Creating demo model...")
        model = create_demo_model()
        model_path = temp_path / "demo_model.pth"
        torch.save(model, model_path)
        
        # Create demo input data
        input_data = torch.randn(1, 1, 28, 28)
        input_path = temp_path / "demo_input.pt"
        torch.save(input_data, input_path)
        
        print(f"   Model saved to: {model_path}")
        print(f"   Input data saved to: {input_path}")
        
        # Demonstrate CLI commands
        cli_commands = [
            # Show info
            {
                "name": "Show compiler info",
                "command": ["spike-compile", "info"]
            },
            
            # Analyze model
            {
                "name": "Analyze model",
                "command": [
                    "spike-compile", "analyze", str(model_path),
                    "--input-shape", "1,1,28,28",
                    "--show-params",
                    "--show-complexity"
                ]
            },
            
            # Compile model (simulation target)
            {
                "name": "Compile model for simulation",
                "command": [
                    "spike-compile", "compile", str(model_path),
                    "--input-shape", "1,1,28,28",
                    "--target", "simulation",
                    "--output", str(temp_path / "compiled_sim"),
                    "--optimization-level", "2",
                    "--time-steps", "4",
                    "--profile-energy",
                    "--verbose"
                ]
            },
            
            # Compile model (Loihi3 target)
            {
                "name": "Compile model for Loihi3", 
                "command": [
                    "spike-compile", "compile", str(model_path),
                    "--input-shape", "1,1,28,28",
                    "--target", "loihi3",
                    "--output", str(temp_path / "compiled_loihi3"),
                    "--optimization-level", "3",
                    "--time-steps", "8",
                    "--profile-energy",
                    "--verbose"
                ]
            },
            
            # List examples
            {
                "name": "List example configurations",
                "command": ["spike-compile", "list-examples"]
            },
        ]
        
        # Execute CLI commands
        for i, cmd_info in enumerate(cli_commands, 2):
            print(f"\n{i}. {cmd_info['name']}")
            print(f"   Command: {' '.join(cmd_info['command'])}")
            print("   Output:")
            
            try:
                # Add PYTHONPATH to include our project
                env = os.environ.copy()
                project_root = Path(__file__).parent.parent / "src"
                env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")
                
                result = subprocess.run(
                    cmd_info['command'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env
                )
                
                if result.returncode == 0:
                    # Indent output
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        print(f"   {line}")
                    print("   ✓ Command successful")
                else:
                    print(f"   ✗ Command failed (exit code {result.returncode})")
                    if result.stderr:
                        error_lines = result.stderr.strip().split('\n')
                        for line in error_lines:
                            print(f"   Error: {line}")
                            
            except subprocess.TimeoutExpired:
                print("   ✗ Command timed out")
            except FileNotFoundError:
                print("   ✗ spike-compile command not found")
                print("   Note: Make sure the package is installed and in PATH")
            except Exception as e:
                print(f"   ✗ Unexpected error: {e}")
        
        # Test inference if compilation was successful
        compiled_metadata_path = temp_path / "compiled_sim" / "compilation_metadata.json"
        if compiled_metadata_path.exists():
            print(f"\n{len(cli_commands) + 1}. Run inference on compiled model")
            
            run_command = [
                "spike-compile", "run", str(model_path), str(input_path),
                "--compiled-model", str(compiled_metadata_path),
                "--time-steps", "4",
                "--benchmark",
                "--verbose"
            ]
            
            print(f"   Command: {' '.join(run_command)}")
            print("   Output:")
            
            try:
                env = os.environ.copy()
                project_root = Path(__file__).parent.parent / "src"
                env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")
                
                result = subprocess.run(
                    run_command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env
                )
                
                if result.returncode == 0:
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines:
                        print(f"   {line}")
                    print("   ✓ Inference successful")
                else:
                    print(f"   ✗ Inference failed (exit code {result.returncode})")
                    if result.stderr:
                        error_lines = result.stderr.strip().split('\n')
                        for line in error_lines:
                            print(f"   Error: {line}")
                            
            except Exception as e:
                print(f"   ✗ Inference error: {e}")
        
        print("\n=== CLI Demo completed! ===")
        
        # Show summary
        print("\nTo use the CLI in your environment:")
        print("1. Install the package: pip install -e .")
        print("2. Use the commands shown above with your own models")
        print("3. Check 'spike-compile --help' for more options")

if __name__ == "__main__":
    main()