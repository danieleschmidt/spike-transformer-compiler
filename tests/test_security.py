"""Tests for security functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from spike_transformer_compiler.security import (
    SecurityConfig,
    ModelSanitizer,
    InputSanitizer,
    GraphSanitizer,
    SecureCompilationEnvironment,
    create_secure_environment
)
from spike_transformer_compiler.exceptions import ValidationError
from spike_transformer_compiler.ir.spike_graph import SpikeGraph, SpikeNode, NodeType


class TestSecurityConfig:
    """Test security configuration."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert config.max_model_size_mb == 1000
        assert config.max_input_dimensions == 5
        assert config.allow_pickle_loading == False
        assert "simulation" in config.allowed_targets
    
    @patch.dict(os.environ, {"SPIKE_MAX_MODEL_SIZE_MB": "500"})
    def test_environment_config(self):
        """Test configuration from environment variables."""
        config = SecurityConfig()
        assert config.max_model_size_mb == 500


class TestModelSanitizer:
    """Test model path and content sanitization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SecurityConfig()
        self.sanitizer = ModelSanitizer(self.config)
    
    def test_valid_model_path(self):
        """Test sanitizing valid model path."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            f.write(b'dummy model content')
            temp_path = f.name
        
        try:
            result = self.sanitizer.sanitize_model_path(temp_path)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_model_path(self):
        """Test error for non-existent model path."""
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_model_path('/nonexistent/model.pth')
        
        assert exc_info.value.error_code == "MODEL_FILE_NOT_FOUND"
    
    def test_directory_as_model_path(self):
        """Test error when model path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError) as exc_info:
                self.sanitizer.sanitize_model_path(temp_dir)
            
            assert exc_info.value.error_code == "MODEL_PATH_NOT_FILE"
    
    def test_oversized_model_file(self):
        """Test error for oversized model file."""
        # Create a config with very small max size
        config = SecurityConfig()
        config.max_model_size_mb = 0.001  # 1KB
        sanitizer = ModelSanitizer(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            f.write(b'x' * 2048)  # 2KB file
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                sanitizer.sanitize_model_path(temp_path)
            
            assert exc_info.value.error_code == "MODEL_FILE_TOO_LARGE"
        finally:
            os.unlink(temp_path)
    
    def test_model_content_validation(self):
        """Test model content validation."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            # Write some dummy content
            f.write(b'dummy model content')
            temp_path = f.name
        
        try:
            result = self.sanitizer.validate_model_content(Path(temp_path))
            assert 'file_size_mb' in result
            assert 'file_hash' in result
            assert 'security_warnings' in result
        finally:
            os.unlink(temp_path)


class TestInputSanitizer:
    """Test input sanitization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SecurityConfig()
        self.sanitizer = InputSanitizer(self.config)
    
    def test_valid_input_shape(self):
        """Test sanitizing valid input shape."""
        shape = [1, 3, 224, 224]
        result = self.sanitizer.sanitize_input_shape(shape)
        assert result == (1, 3, 224, 224)
        assert isinstance(result, tuple)
    
    def test_invalid_shape_type(self):
        """Test error for invalid shape type."""
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_input_shape("invalid")
        
        assert exc_info.value.error_code == "INVALID_SHAPE_TYPE"
    
    def test_too_many_dimensions(self):
        """Test error for too many dimensions."""
        shape = [1, 2, 3, 4, 5, 6, 7]  # 7 dimensions
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_input_shape(shape)
        
        assert exc_info.value.error_code == "TOO_MANY_DIMENSIONS"
    
    def test_negative_dimension(self):
        """Test error for negative dimensions."""
        shape = [1, 3, -224, 224]
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_input_shape(shape)
        
        assert exc_info.value.error_code == "INVALID_DIMENSION_VALUE"
    
    def test_non_integer_dimension(self):
        """Test error for non-integer dimensions."""
        shape = [1, 3, 224.5, 224]
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_input_shape(shape)
        
        assert exc_info.value.error_code == "INVALID_DIMENSION_TYPE"
    
    def test_tensor_too_large(self):
        """Test error for tensor that's too large."""
        # Create config with small max tensor size
        config = SecurityConfig()
        config.max_tensor_size = 1000
        sanitizer = InputSanitizer(config)
        
        shape = [10, 10, 10, 10]  # 10,000 elements
        with pytest.raises(ValidationError) as exc_info:
            sanitizer.sanitize_input_shape(shape)
        
        assert exc_info.value.error_code == "TENSOR_TOO_LARGE"
    
    def test_valid_target_sanitization(self):
        """Test sanitizing valid compilation target."""
        result = self.sanitizer.sanitize_compilation_target("SIMULATION")
        assert result == "simulation"
    
    def test_invalid_target_type(self):
        """Test error for invalid target type."""
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_compilation_target(123)
        
        assert exc_info.value.error_code == "INVALID_TARGET_TYPE"
    
    def test_disallowed_target(self):
        """Test error for disallowed target."""
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.sanitize_compilation_target("malicious_target")
        
        assert exc_info.value.error_code == "TARGET_NOT_ALLOWED"
    
    def test_optimization_level_clamping(self):
        """Test optimization level is clamped to valid range."""
        assert self.sanitizer.sanitize_optimization_level(-1) == 0
        assert self.sanitizer.sanitize_optimization_level(5) == 3
        assert self.sanitizer.sanitize_optimization_level(2) == 2
    
    def test_time_steps_validation(self):
        """Test time steps validation and clamping."""
        assert self.sanitizer.sanitize_time_steps(0) == 1
        assert self.sanitizer.sanitize_time_steps(2000) == 1000
        assert self.sanitizer.sanitize_time_steps(10) == 10


class TestGraphSanitizer:
    """Test graph sanitization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SecurityConfig()
        self.sanitizer = GraphSanitizer(self.config)
    
    def test_valid_graph_size(self):
        """Test validation of reasonably sized graph."""
        graph = SpikeGraph("test")
        
        # Add a few nodes
        for i in range(10):
            node = SpikeNode(
                id=f"node_{i}",
                node_type=NodeType.SPIKE_NEURON,
                operation="test_op",
                inputs=[],
                outputs=[],
                parameters={"param": 1.0},
                metadata={}
            )
            graph.add_node(node)
        
        # Should not raise an exception
        self.sanitizer.validate_graph_size(graph)
    
    def test_graph_too_large(self):
        """Test error for graph with too many nodes."""
        config = SecurityConfig()
        config.max_nodes = 5
        sanitizer = GraphSanitizer(config)
        
        graph = SpikeGraph("test")
        
        # Add more nodes than allowed
        for i in range(10):
            node = SpikeNode(
                id=f"node_{i}",
                node_type=NodeType.SPIKE_NEURON,
                operation="test_op",
                inputs=[],
                outputs=[],
                parameters={"param": 1.0},
                metadata={}
            )
            graph.add_node(node)
        
        with pytest.raises(ValidationError) as exc_info:
            sanitizer.validate_graph_size(graph)
        
        assert exc_info.value.error_code == "GRAPH_TOO_LARGE"
    
    def test_suspicious_node_parameters(self):
        """Test detection of suspicious node parameters."""
        node = SpikeNode(
            id="suspicious_node",
            node_type=NodeType.SPIKE_NEURON,
            operation="test_op",
            inputs=[],
            outputs=[],
            parameters={"malicious_code": "exec('import os; os.system(\"rm -rf /\")')"},
            metadata={}
        )
        
        with pytest.raises(ValidationError) as exc_info:
            self.sanitizer.validate_node_parameters(node)
        
        assert exc_info.value.error_code == "SUSPICIOUS_PARAMETER"
    
    def test_large_parameter_values(self):
        """Test warning for extremely large parameter values."""
        node = SpikeNode(
            id="large_param_node",
            node_type=NodeType.SPIKE_NEURON,
            operation="test_op",
            inputs=[],
            outputs=[],
            parameters={"large_value": 1e12},
            metadata={}
        )
        
        # Should not raise exception but may generate warning
        with pytest.warns(None) as warnings:
            self.sanitizer.validate_node_parameters(node)
        
        # Check if warning was generated
        warning_messages = [str(w.message) for w in warnings.list]
        assert any("Large parameter value" in msg for msg in warning_messages)


class TestSecureCompilationEnvironment:
    """Test secure compilation environment."""
    
    def test_environment_context_manager(self):
        """Test secure environment context manager."""
        with create_secure_environment() as env:
            assert env.temp_dir is not None
            assert os.path.exists(env.temp_dir)
            temp_dir = env.temp_dir
        
        # Directory should be cleaned up
        assert not os.path.exists(temp_dir)
    
    def test_input_validation_and_sanitization(self):
        """Test complete input validation and sanitization."""
        with create_secure_environment() as env:
            # Create a temporary model file
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                f.write(b'dummy model content')
                temp_model_path = f.name
            
            try:
                sanitized = env.validate_and_sanitize_inputs(
                    model_path=temp_model_path,
                    input_shape=[1, 3, 224, 224],
                    target="SIMULATION",
                    optimization_level=2,
                    time_steps=4
                )
                
                assert isinstance(sanitized['model_path'], Path)
                assert sanitized['input_shape'] == (1, 3, 224, 224)
                assert sanitized['target'] == "simulation"
                assert sanitized['optimization_level'] == 2
                assert sanitized['time_steps'] == 4
                assert 'model_validation' in sanitized
                
            finally:
                os.unlink(temp_model_path)


class TestSecurityIntegration:
    """Test security integration with main compiler."""
    
    def test_security_config_singleton(self):
        """Test that security config is a singleton."""
        from spike_transformer_compiler.security import get_security_config
        
        config1 = get_security_config()
        config2 = get_security_config()
        
        assert config1 is config2
    
    def test_hardware_target_validation(self):
        """Test hardware target validation."""
        from spike_transformer_compiler.security import validate_hardware_target
        
        assert validate_hardware_target("simulation") == True
        assert validate_hardware_target("invalid_target") == False
    
    def test_model_hash_generation(self):
        """Test secure model hash generation."""
        from spike_transformer_compiler.security import get_model_hash
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            f.write(b'test model content')
            temp_path = f.name
        
        try:
            hash1 = get_model_hash(temp_path)
            hash2 = get_model_hash(temp_path)
            
            assert hash1 == hash2  # Same file should have same hash
            assert len(hash1) == 64  # SHA256 hash length
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])