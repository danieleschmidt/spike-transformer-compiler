"""Model analysis utilities for frontend parsing."""

from typing import Dict, Any, Tuple


class ModelAnalyzer:
    """Analyzer for model structure and compatibility."""
    
    def __init__(self):
        self.supported_types = {"TestModel", "Module", "torch.nn.Module"}
    
    def analyze(self, model: Any) -> Dict[str, Any]:
        """Analyze model structure."""
        return analyze_model_structure(model)
    
    def validate_compatibility(self, model: Any, target: str = "simulation") -> bool:
        """Validate model compatibility."""
        return validate_model_compatibility(model, target)


def analyze_model_structure(model: Any) -> Dict[str, Any]:
    """Analyze model structure and extract metadata."""
    
    analysis = {
        "model_type": type(model).__name__,
        "has_forward_method": hasattr(model, 'forward'),
        "is_callable": callable(model),
        "estimated_parameters": 0,
        "estimated_layers": 1,
        "supported": True
    }
    
    # Try to count parameters if it's a PyTorch-like model
    try:
        if hasattr(model, 'parameters'):
            params = list(model.parameters())
            analysis["estimated_parameters"] = sum(p.numel() for p in params if hasattr(p, 'numel'))
            analysis["parameter_count"] = len(params)
        elif hasattr(model, '__dict__'):
            # Count attributes as a rough proxy for complexity
            analysis["estimated_layers"] = len([k for k in model.__dict__.keys() if not k.startswith('_')])
    except Exception:
        # Fallback to defaults
        pass
    
    return analysis


def validate_model_compatibility(model: Any, target: str = "simulation") -> bool:
    """Check if model is compatible with target backend."""
    
    # Basic compatibility checks
    if not callable(model) and not hasattr(model, 'forward'):
        return False
    
    # Target-specific checks
    if target == "loihi3":
        # Stricter requirements for hardware
        analysis = analyze_model_structure(model)
        if analysis["estimated_parameters"] > 1000000:  # 1M parameter limit
            return False
    
    return True


def extract_input_requirements(model: Any, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Extract input requirements from model."""
    
    requirements = {
        "input_shape": input_shape,
        "batch_dimension": 0 if input_shape and input_shape[0] == 1 else None,
        "spatial_dimensions": len(input_shape) - 2 if len(input_shape) > 2 else 0,
        "feature_dimensions": input_shape[-1] if input_shape else 0,
        "memory_estimate_mb": 0.0
    }
    
    # Estimate memory usage
    if input_shape:
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        
        # Assume float32 (4 bytes per element)
        requirements["memory_estimate_mb"] = (total_elements * 4) / (1024 * 1024)
    
    return requirements