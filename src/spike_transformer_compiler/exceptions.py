"""Custom exceptions for Spike-Transformer-Compiler."""

from typing import Optional, Any


class SpikeCompilerError(Exception):
    """Base exception for all spike compiler errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.error_code:
            return f"[{self.error_code}] {base_msg}"
        return base_msg


class CompilationError(SpikeCompilerError):
    """Exception raised during model compilation."""
    pass


class ValidationError(SpikeCompilerError):
    """Exception raised during input validation."""
    pass


class ModelParsingError(SpikeCompilerError):
    """Exception raised during model parsing."""
    pass


class BackendError(SpikeCompilerError):
    """Exception raised by backend implementations."""
    pass


class HardwareError(SpikeCompilerError):
    """Exception raised during hardware operations."""
    pass


class OptimizationError(SpikeCompilerError):
    """Exception raised during optimization passes."""
    pass


class ResourceError(SpikeCompilerError):
    """Exception raised when resource constraints are violated."""
    pass


class IRError(SpikeCompilerError):
    """Exception raised during IR operations."""
    pass


class ConfigurationError(SpikeCompilerError):
    """Exception raised for configuration issues."""
    pass


class InferenceError(SpikeCompilerError):
    """Exception raised during model inference."""
    pass


# Validation utilities
class ValidationUtils:
    """Utilities for input validation."""
    
    @staticmethod
    def validate_input_shape(shape: tuple, min_dims: int = 2, max_dims: int = 5) -> None:
        """Validate input tensor shape."""
        if not isinstance(shape, (tuple, list)):
            raise ValidationError(
                "Input shape must be a tuple or list",
                error_code="INVALID_SHAPE_TYPE",
                details={'provided_type': type(shape).__name__}
            )
        
        if len(shape) < min_dims or len(shape) > max_dims:
            raise ValidationError(
                f"Input shape must have {min_dims}-{max_dims} dimensions",
                error_code="INVALID_SHAPE_DIMS",
                details={
                    'provided_dims': len(shape),
                    'min_dims': min_dims,
                    'max_dims': max_dims,
                    'shape': shape
                }
            )
        
        for i, dim in enumerate(shape):
            if not isinstance(dim, int) or dim <= 0:
                raise ValidationError(
                    f"All shape dimensions must be positive integers",
                    error_code="INVALID_SHAPE_VALUE",
                    details={
                        'dimension_index': i,
                        'dimension_value': dim,
                        'shape': shape
                    }
                )
    
    @staticmethod
    def validate_optimization_level(level: int) -> None:
        """Validate optimization level."""
        if not isinstance(level, int):
            raise ValidationError(
                "Optimization level must be an integer",
                error_code="INVALID_OPT_LEVEL_TYPE",
                details={'provided_type': type(level).__name__}
            )
        
        if level < 0 or level > 3:
            raise ValidationError(
                "Optimization level must be between 0 and 3",
                error_code="INVALID_OPT_LEVEL_VALUE",
                details={'provided_level': level}
            )
    
    @staticmethod
    def validate_time_steps(time_steps: int) -> None:
        """Validate time steps parameter."""
        if not isinstance(time_steps, int):
            raise ValidationError(
                "Time steps must be an integer",
                error_code="INVALID_TIME_STEPS_TYPE",
                details={'provided_type': type(time_steps).__name__}
            )
        
        if time_steps <= 0 or time_steps > 1000:
            raise ValidationError(
                "Time steps must be between 1 and 1000",
                error_code="INVALID_TIME_STEPS_VALUE",
                details={'provided_time_steps': time_steps}
            )
    
    @staticmethod
    def validate_target(target: str, available_targets: list) -> None:
        """Validate compilation target."""
        if not isinstance(target, str):
            raise ValidationError(
                "Target must be a string",
                error_code="INVALID_TARGET_TYPE",
                details={'provided_type': type(target).__name__}
            )
        
        if target not in available_targets:
            raise ValidationError(
                f"Unsupported target: {target}",
                error_code="UNSUPPORTED_TARGET",
                details={
                    'provided_target': target,
                    'available_targets': available_targets
                }
            )
    
    @staticmethod
    def validate_model(model: Any) -> None:
        """Validate PyTorch model."""
        if model is None:
            raise ValidationError(
                "Model cannot be None",
                error_code="NULL_MODEL"
            )
        
        # Check if it's a PyTorch model (if torch is available)
        try:
            import torch.nn as nn
            if not isinstance(model, nn.Module):
                raise ValidationError(
                    "Model must be a PyTorch nn.Module",
                    error_code="INVALID_MODEL_TYPE",
                    details={'provided_type': type(model).__name__}
                )
        except ImportError:
            # If torch is not available, we can't validate the model type strictly
            # But we can check if the model has the expected interface
            if not (hasattr(model, 'named_modules') or hasattr(model, '__call__')):
                raise ValidationError(
                    "Model must have PyTorch-like interface (named_modules method)",
                    error_code="INVALID_MODEL_INTERFACE",
                    details={'provided_type': type(model).__name__}
                )
    
    @staticmethod
    def validate_node_parameters(node_type: str, parameters: dict) -> None:
        """Validate node parameters based on node type."""
        required_params = {
            'spike_neuron': ['neuron_model', 'threshold'],
            'spike_conv': ['in_channels', 'out_channels', 'kernel_size'],
            'spike_linear': ['in_features', 'out_features'],
            'spike_attention': ['embed_dim', 'num_heads']
        }
        
        if node_type in required_params:
            for param in required_params[node_type]:
                if param not in parameters:
                    raise ValidationError(
                        f"Missing required parameter '{param}' for node type '{node_type}'",
                        error_code="MISSING_NODE_PARAMETER",
                        details={
                            'node_type': node_type,
                            'missing_parameter': param,
                            'provided_parameters': list(parameters.keys())
                        }
                    )
    
    @staticmethod
    def validate_resource_constraints(
        required_resources: dict, 
        available_resources: dict
    ) -> None:
        """Validate resource requirements against available resources."""
        for resource, required in required_resources.items():
            if resource in available_resources:
                available = available_resources[resource]
                if required > available:
                    raise ResourceError(
                        f"Insufficient {resource}: required {required}, available {available}",
                        error_code="INSUFFICIENT_RESOURCES",
                        details={
                            'resource_type': resource,
                            'required': required,
                            'available': available
                        }
                    )


# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and fallback strategies."""
    
    @staticmethod
    def suggest_fix_for_shape_error(provided_shape: tuple, expected_pattern: str) -> str:
        """Suggest fix for shape validation errors."""
        suggestions = {
            'image_2d': "For 2D images, use shape (batch, channels, height, width), e.g., (1, 3, 224, 224)",
            'image_1d': "For 1D data, use shape (batch, features), e.g., (1, 784)",
            'sequence': "For sequences, use shape (batch, seq_len, features), e.g., (1, 196, 768)"
        }
        
        base_msg = f"Provided shape {provided_shape} is invalid. "
        return base_msg + suggestions.get(expected_pattern, "Please check the documentation for valid shapes.")
    
    @staticmethod
    def suggest_target_fallback(failed_target: str, available_targets: list) -> str:
        """Suggest fallback target when compilation fails."""
        if 'simulation' in available_targets:
            return f"Target '{failed_target}' failed. Try 'simulation' target for testing."
        elif available_targets:
            return f"Target '{failed_target}' failed. Available alternatives: {', '.join(available_targets)}"
        else:
            return f"Target '{failed_target}' failed and no alternatives available."
    
    @staticmethod
    def suggest_optimization_fallback(failed_level: int) -> str:
        """Suggest lower optimization level on failure."""
        if failed_level > 0:
            return f"Optimization level {failed_level} failed. Try level {failed_level - 1} or 0 (no optimization)."
        else:
            return "Even basic compilation failed. Check model architecture and input shapes."


# Context manager for error handling
class ErrorContext:
    """Context manager for enhanced error reporting."""
    
    def __init__(self, operation: str, **context_info):
        self.operation = operation
        self.context_info = context_info
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type and issubclass(exc_type, SpikeCompilerError):
            # Enhance the error with context information
            exc_value.details.update({
                'operation': self.operation,
                'context': self.context_info
            })
        return False  # Don't suppress the exception