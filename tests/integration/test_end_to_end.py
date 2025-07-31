"""End-to-end integration tests for the compilation pipeline."""

import pytest
import torch
from spike_transformer_compiler import SpikeCompiler
from spike_transformer_compiler.optimization import OptimizationPass


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for the complete compilation pipeline."""
    
    def test_basic_compilation_pipeline(self, sample_model, temp_dir):
        """Test basic compilation pipeline from model to output."""
        compiler = SpikeCompiler(
            target="simulation",
            optimization_level=2,
            time_steps=4
        )
        
        # Test that compilation interface works (even if not implemented)
        with pytest.raises(NotImplementedError):
            compiled_model = compiler.compile(
                sample_model,
                input_shape=(1, 10),
                chip_config="single-chip"
            )
    
    def test_optimization_pipeline_integration(self, sample_model):
        """Test integration between compiler and optimization pipeline."""
        compiler = SpikeCompiler(target="simulation")
        optimizer = compiler.create_optimizer()
        
        # Test optimization pass integration
        with pytest.raises(NotImplementedError):
            optimizer.add_pass(OptimizationPass.SPIKE_COMPRESSION)
            optimizer.add_pass(OptimizationPass.WEIGHT_QUANTIZATION, bits=4)
            
            compiled_model = compiler.compile(
                sample_model,
                input_shape=(1, 10),
                optimizer=optimizer
            )
    
    def test_cli_integration(self, temp_dir):
        """Test CLI integration with core functionality."""
        from click.testing import CliRunner
        from spike_transformer_compiler.cli import main
        
        # Create a dummy model file
        dummy_model_path = temp_dir / "model.pth"
        torch.save(torch.nn.Linear(10, 1).state_dict(), dummy_model_path)
        
        runner = CliRunner()
        result = runner.invoke(main, [
            "compile",
            str(dummy_model_path),
            "--target", "simulation",
            "--verbose"
        ])
        
        assert result.exit_code == 0
        assert "Compiling" in result.output
    
    @pytest.mark.hardware
    def test_simulation_backend_integration(self, sample_model):
        """Test integration with simulation backend."""
        compiler = SpikeCompiler(target="simulation", debug=True)
        
        # Enable debug options
        compiler.set_debug_options(dump_ir=True, dump_passes=True)
        
        # Test compilation with debug enabled
        with pytest.raises(NotImplementedError):
            compiled_model = compiler.compile(
                sample_model,
                input_shape=(1, 10),
                profile_energy=True
            )
    
    def test_error_handling_integration(self, temp_dir):
        """Test error handling across the pipeline."""
        compiler = SpikeCompiler(target="invalid_target")
        
        # Test invalid model handling
        with pytest.raises((NotImplementedError, ValueError)):
            compiler.compile(None, (1, 10))
            
        # Test invalid input shape handling  
        with pytest.raises((NotImplementedError, ValueError)):
            compiler.compile(torch.nn.Linear(10, 1), None)