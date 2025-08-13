"""Validation utilities for spike transformer compilation."""

from typing import Any, List, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    class torch:
        class Tensor:
            pass
    class nn:
        class Module:
            pass


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidationUtils:
    """Static validation utilities."""
    
    @staticmethod
    def validate_target(target: str, available_targets: List[str]) -> None:
        """Validate compilation target."""
        if target not in available_targets:
            raise ValidationError(
                f"Unsupported target '{target}'. Available targets: {available_targets}"
            )
    
    @staticmethod
    def validate_optimization_level(level: int) -> None:
        """Validate optimization level."""
        if not isinstance(level, int) or level < 0 or level > 3:
            raise ValidationError(
                f"Optimization level must be 0-3, got {level}"
            )
    
    @staticmethod
    def validate_time_steps(time_steps: int) -> None:
        """Validate time steps parameter."""
        if not isinstance(time_steps, int) or time_steps < 1:
            raise ValidationError(
                f"Time steps must be positive integer, got {time_steps}"
            )
    
    @staticmethod
    def validate_model(model: Any) -> None:
        """Validate model for compilation."""
        if model is None:
            raise ValidationError("Model cannot be None")
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            # PyTorch model validation
            try:
                # Check if model is in evaluation mode for consistent compilation
                if model.training:
                    model.eval()
            except Exception as e:
                raise ValidationError(f"Failed to validate PyTorch model: {e}")
        
        # Check if model has callable forward method
        if not hasattr(model, '__call__') and not hasattr(model, 'forward'):
            raise ValidationError(
                "Model must be callable or have a forward method"
            )
    
    @staticmethod
    def validate_input_shape(input_shape: Tuple[int, ...]) -> None:
        """Validate input tensor shape."""
        if not isinstance(input_shape, (tuple, list)):
            raise ValidationError(
                f"Input shape must be tuple or list, got {type(input_shape)}"
            )
        
        if len(input_shape) == 0:
            raise ValidationError("Input shape cannot be empty")
        
        for dim in input_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ValidationError(
                    f"All dimensions must be positive integers, got shape {input_shape}"
                )
        
        # Check for reasonable size limits
        if NUMPY_AVAILABLE:
            total_elements = np.prod(input_shape)
        else:
            total_elements = 1
            for dim in input_shape:
                total_elements *= dim
                
        if total_elements > 1e8:  # 100M elements
            raise ValidationError(
                f"Input shape {input_shape} is too large ({total_elements} elements)"
            )


class ErrorContext:
    """Context manager for error tracking."""
    
    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.context = kwargs
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Add context to exception if it's a ValidationError
            if issubclass(exc_type, ValidationError):
                enhanced_msg = f"During {self.operation}: {str(exc_val)}"
                if self.context:
                    enhanced_msg += f" (Context: {self.context})"
                # Re-raise with enhanced message
                raise ValidationError(enhanced_msg) from exc_val
        return False


class ErrorRecovery:
    """Error recovery and suggestion utilities."""
    
    @staticmethod
    def suggest_fix_for_shape_error(input_shape: Tuple[int, ...], expected_type: str) -> str:
        """Suggest fixes for shape-related errors."""
        if expected_type == "image_2d" and len(input_shape) != 4:
            return (
                f"For 2D image input, expected shape (batch, channels, height, width), "
                f"got {input_shape}. Try reshaping your input."
            )
        elif expected_type == "sequence" and len(input_shape) != 3:
            return (
                f"For sequence input, expected shape (batch, sequence, features), "
                f"got {input_shape}. Consider adding batch or sequence dimensions."
            )
        
        return "Check input shape and model requirements."
    
    @staticmethod
    def suggest_target_fallback(failed_target: str, available_targets: List[str]) -> str:
        """Suggest alternative targets when compilation fails."""
        fallbacks = {
            "loihi3": "simulation",
            "loihi2": "simulation", 
            "fpga": "simulation"
        }
        
        fallback = fallbacks.get(failed_target, "simulation")
        if fallback in available_targets:
            return f"Consider using '{fallback}' target as fallback."
        
        return f"Available targets: {', '.join(available_targets)}"