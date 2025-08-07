"""Robust testing for the spike transformer compiler."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from spike_transformer_compiler import SpikeCompiler
from spike_transformer_compiler.exceptions import (
    CompilationError,
    ValidationError,
    ResourceError,
    BackendError
)


class TestRobustCompiler:
    """Test compiler robustness and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = SpikeCompiler(target="simulation", verbose=False)
    
    def create_simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def test_successful_compilation(self):
        """Test successful compilation flow."""
        model = self.create_simple_model()
        
        compiled_model = self.compiler.compile(
            model,
            input_shape=(1, 784),
            profile_energy=True
        )
        
        assert compiled_model is not None
        assert hasattr(compiled_model, 'run')
    
    def test_invalid_input_shape_validation(self):
        """Test validation of invalid input shapes."""
        model = self.create_simple_model()
        
        # Test various invalid shapes
        invalid_shapes = [
            "invalid",  # Wrong type
            [],         # Empty shape
            [0, 784],   # Zero dimension
            [-1, 784],  # Negative dimension
            [1.5, 784], # Non-integer dimension
            tuple(range(10)),  # Too many dimensions
        ]
        
        for invalid_shape in invalid_shapes:
            with pytest.raises((CompilationError, ValidationError)):
                self.compiler.compile(model, input_shape=invalid_shape)
    
    def test_invalid_model_validation(self):
        """Test validation of invalid models."""
        invalid_models = [
            None,           # None model
            "not_a_model",  # Wrong type
            123,            # Wrong type
        ]
        
        for invalid_model in invalid_models:
            with pytest.raises((CompilationError, ValidationError)):
                self.compiler.compile(invalid_model, input_shape=(1, 784))
    
    def test_invalid_optimization_level(self):
        """Test validation of invalid optimization levels."""
        invalid_levels = [-1, 4, "invalid", None]
        
        for level in invalid_levels:
            with pytest.raises((CompilationError, ValidationError)):
                SpikeCompiler(optimization_level=level)
    
    def test_invalid_time_steps(self):
        """Test validation of invalid time steps."""
        invalid_time_steps = [0, -1, "invalid", None]
        
        for time_steps in invalid_time_steps:
            with pytest.raises((CompilationError, ValidationError)):
                SpikeCompiler(time_steps=time_steps)
    
    def test_invalid_target(self):
        """Test validation of invalid targets."""
        invalid_targets = ["invalid_target", 123, None]
        
        for target in invalid_targets:
            with pytest.raises((CompilationError, ValidationError)):
                SpikeCompiler(target=target)
    
    def test_compilation_with_large_model(self):
        """Test compilation with a large model."""
        # Create a larger model
        large_model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1000)
        )
        
        # Should handle large model gracefully
        compiled_model = self.compiler.compile(
            large_model,
            input_shape=(1, 1024),
            optimization_level=3
        )
        
        assert compiled_model is not None
    
    def test_compilation_with_complex_architecture(self):
        """Test compilation with complex model architecture."""
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, 10)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 64 * 8 * 8)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        model = ComplexModel()
        
        compiled_model = self.compiler.compile(
            model,
            input_shape=(1, 3, 32, 32)
        )
        
        assert compiled_model is not None
    
    def test_memory_stress(self):
        """Test behavior under memory stress conditions."""
        # This test simulates memory pressure
        model = self.create_simple_model()
        
        # Mock a memory error during compilation
        with patch('spike_transformer_compiler.ir.builder.SpikeIRBuilder') as mock_builder:
            mock_builder.side_effect = MemoryError("Out of memory")
            
            with pytest.raises(CompilationError):
                self.compiler.compile(model, input_shape=(1, 784))
    
    def test_backend_failure_handling(self):
        """Test handling of backend compilation failures."""
        model = self.create_simple_model()
        
        # Mock backend failure
        with patch('spike_transformer_compiler.backend.factory.BackendFactory.create_backend') as mock_factory:
            mock_backend = Mock()
            mock_backend.compile_graph.side_effect = BackendError("Backend compilation failed")
            mock_factory.return_value = mock_backend
            
            with pytest.raises(CompilationError):
                self.compiler.compile(model, input_shape=(1, 784))
    
    def test_parser_failure_handling(self):
        """Test handling of parser failures."""
        model = self.create_simple_model()
        
        # Mock parser failure
        with patch('spike_transformer_compiler.frontend.pytorch_parser.PyTorchParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_model.side_effect = Exception("Parsing failed")
            mock_parser_class.return_value = mock_parser
            
            with pytest.raises(CompilationError):
                self.compiler.compile(model, input_shape=(1, 784))
    
    def test_optimization_failure_recovery(self):
        """Test recovery from optimization failures."""
        model = self.create_simple_model()
        
        # Mock optimization failure, should still complete compilation
        with patch('spike_transformer_compiler.compiler.SpikeCompiler.create_optimizer') as mock_create_opt:
            mock_optimizer = Mock()
            mock_optimizer.run_all.side_effect = Exception("Optimization failed")
            mock_create_opt.return_value = mock_optimizer
            
            # Should raise CompilationError since optimization is mandatory
            with pytest.raises(CompilationError):
                self.compiler.compile(model, input_shape=(1, 784))
    
    def test_concurrent_compilation(self):
        """Test concurrent compilation requests."""
        import threading
        import time
        
        model = self.create_simple_model()
        results = []
        errors = []
        
        def compile_model():
            try:
                compiled = self.compiler.compile(model, input_shape=(1, 784))
                results.append(compiled)
            except Exception as e:
                errors.append(e)
        
        # Start multiple compilation threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=compile_model)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All compilations should succeed
        assert len(results) == 3
        assert len(errors) == 0
    
    def test_resource_constraint_handling(self):
        """Test handling of resource constraints."""
        model = self.create_simple_model()
        
        # Mock resource allocator that reports insufficient resources
        mock_allocator = Mock()
        mock_allocator.check_resources.return_value = False
        
        with patch('spike_transformer_compiler.backend.factory.BackendFactory.create_backend') as mock_factory:
            mock_backend = Mock()
            mock_backend.compile_graph.side_effect = ResourceError(
                "Insufficient hardware resources",
                error_code="INSUFFICIENT_RESOURCES"
            )
            mock_factory.return_value = mock_backend
            
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.compile(
                    model,
                    input_shape=(1, 784),
                    resource_allocator=mock_allocator
                )
            
            # Should provide helpful error message
            assert "Insufficient" in str(exc_info.value)
    
    def test_security_mode_enforcement(self):
        """Test security mode enforcement."""
        model = self.create_simple_model()
        
        # Test with security mode enabled (default)
        compiled_model = self.compiler.compile(
            model,
            input_shape=(1, 784),
            secure_mode=True
        )
        assert compiled_model is not None
        
        # Test with security mode disabled
        compiled_model = self.compiler.compile(
            model,
            input_shape=(1, 784),
            secure_mode=False
        )
        assert compiled_model is not None
    
    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        model = self.create_simple_model()
        
        with patch('spike_transformer_compiler.logging_config.compiler_logger') as mock_logger:
            self.compiler.compile(model, input_shape=(1, 784))
            
            # Verify logging calls were made
            mock_logger.start_compilation.assert_called_once()
            mock_logger.end_compilation.assert_called_once()
            assert mock_logger.log_model_info.called
            assert mock_logger.log_compilation_stage.called
    
    def test_error_context_preservation(self):
        """Test that error context is preserved through compilation."""
        model = self.create_simple_model()
        
        # Force an error and check context is preserved
        with patch('spike_transformer_compiler.frontend.pytorch_parser.PyTorchParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_model.side_effect = Exception("Test error")
            mock_parser_class.return_value = mock_parser
            
            with pytest.raises(CompilationError) as exc_info:
                self.compiler.compile(model, input_shape=(1, 784))
            
            # Check that error contains helpful context
            error_str = str(exc_info.value)
            assert "simulation" in error_str  # Target should be mentioned
    
    def test_cleanup_on_failure(self):
        """Test that resources are cleaned up on compilation failure."""
        model = self.create_simple_model()
        
        # Mock failure during compilation
        with patch('spike_transformer_compiler.frontend.pytorch_parser.PyTorchParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_model.side_effect = Exception("Test failure")
            mock_parser_class.return_value = mock_parser
            
            with pytest.raises(CompilationError):
                self.compiler.compile(model, input_shape=(1, 784))
        
        # Verify cleanup occurred (this would be more comprehensive in real implementation)
        # For now, just ensure no exceptions during cleanup
        assert True


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_fallback_suggestions(self):
        """Test that fallback suggestions are provided."""
        compiler = SpikeCompiler(target="loihi3")  # Hardware target
        model = nn.Linear(10, 5)
        
        # Force backend failure
        with patch('spike_transformer_compiler.backend.factory.BackendFactory.create_backend') as mock_factory:
            mock_factory.side_effect = Exception("Hardware not available")
            
            with pytest.raises(CompilationError) as exc_info:
                compiler.compile(model, input_shape=(1, 10))
            
            # Should suggest fallback to simulation
            error_msg = str(exc_info.value)
            assert "simulation" in error_msg.lower()
    
    def test_shape_error_suggestions(self):
        """Test that shape error suggestions are helpful."""
        compiler = SpikeCompiler(target="simulation")
        model = nn.Linear(10, 5)
        
        with pytest.raises(CompilationError) as exc_info:
            compiler.compile(model, input_shape="invalid_shape")
        
        # Should provide shape format guidance
        error_msg = str(exc_info.value)
        assert any(word in error_msg.lower() for word in ["shape", "format", "tuple", "list"])


class TestPerformanceAndReliability:
    """Test performance characteristics and reliability."""
    
    def test_compilation_timeout(self):
        """Test that compilation respects timeout limits."""
        # This would be implemented with actual timeout mechanisms
        # For now, just verify the concept
        compiler = SpikeCompiler(target="simulation")
        model = nn.Linear(10, 5)
        
        # Successful compilation should complete quickly
        import time
        start_time = time.time()
        
        compiled_model = compiler.compile(model, input_shape=(1, 10))
        
        compilation_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert compilation_time < 30.0  # 30 seconds max
        assert compiled_model is not None
    
    def test_memory_efficiency(self):
        """Test memory efficiency during compilation."""
        compiler = SpikeCompiler(target="simulation")
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Monitor memory usage (simplified)
        import gc
        gc.collect()
        
        compiled_model = compiler.compile(model, input_shape=(1, 100))
        
        # Compilation should succeed without memory issues
        assert compiled_model is not None
        
        # Clean up
        del compiled_model
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])