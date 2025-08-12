"""Security measures and input sanitization for Spike-Transformer-Compiler."""

import os
import hashlib
import pickle
import tempfile
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import warnings
from .exceptions import ValidationError, ConfigurationError
from .logging_config import compiler_logger


class SecurityConfig:
    """Security configuration and policies."""
    
    def __init__(self):
        # Model loading security settings
        self.allow_pickle_loading = os.getenv("SPIKE_ALLOW_PICKLE", "false").lower() == "true"
        self.max_model_size_mb = int(os.getenv("SPIKE_MAX_MODEL_SIZE_MB", "1000"))  # 1GB default
        self.allowed_model_paths = self._get_allowed_paths()
        
        # Input validation settings
        self.max_input_dimensions = int(os.getenv("SPIKE_MAX_INPUT_DIMS", "5"))
        self.max_tensor_size = int(os.getenv("SPIKE_MAX_TENSOR_SIZE", "1000000000"))  # 1B elements
        self.max_batch_size = int(os.getenv("SPIKE_MAX_BATCH_SIZE", "1024"))
        
        # Compilation limits
        self.max_nodes = int(os.getenv("SPIKE_MAX_NODES", "100000"))
        self.max_edges = int(os.getenv("SPIKE_MAX_EDGES", "500000"))
        self.compilation_timeout = int(os.getenv("SPIKE_COMPILATION_TIMEOUT", "3600"))  # 1 hour
        
        # Hardware security
        self.allowed_targets = self._get_allowed_targets()
        self.require_hardware_verification = os.getenv("SPIKE_REQUIRE_HW_VERIFICATION", "false").lower() == "true"
        
        compiler_logger.logger.info("Security configuration loaded")
    
    def _get_allowed_paths(self) -> List[str]:
        """Get allowed model paths from environment."""
        paths_env = os.getenv("SPIKE_ALLOWED_MODEL_PATHS", "")
        if paths_env:
            return [path.strip() for path in paths_env.split(",")]
        return []  # Empty list means no restrictions
    
    def _get_allowed_targets(self) -> List[str]:
        """Get allowed compilation targets."""
        targets_env = os.getenv("SPIKE_ALLOWED_TARGETS", "simulation,loihi3")
        return [target.strip() for target in targets_env.split(",")]


class ModelSanitizer:
    """Sanitize and validate models before compilation."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
    
    def sanitize_model_path(self, model_path: str) -> Path:
        """Sanitize and validate model file path."""
        path = Path(model_path)
        
        # Convert to absolute path and resolve symlinks
        try:
            path = path.resolve()
        except OSError as e:
            raise ValidationError(
                f"Invalid model path: {e}",
                error_code="INVALID_MODEL_PATH"
            )
        
        # Check if file exists
        if not path.exists():
            raise ValidationError(
                f"Model file not found: {path}",
                error_code="MODEL_FILE_NOT_FOUND"
            )
        
        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValidationError(
                f"Model path is not a file: {path}",
                error_code="MODEL_PATH_NOT_FILE"
            )
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_model_size_mb:
            raise ValidationError(
                f"Model file too large: {file_size_mb:.1f}MB > {self.config.max_model_size_mb}MB",
                error_code="MODEL_FILE_TOO_LARGE"
            )
        
        # Check allowed paths if configured
        if self.config.allowed_model_paths:
            path_str = str(path)
            allowed = any(path_str.startswith(allowed_path) for allowed_path in self.config.allowed_model_paths)
            if not allowed:
                raise ValidationError(
                    f"Model path not in allowed directories: {path}",
                    error_code="MODEL_PATH_NOT_ALLOWED"
                )
        
        return path
    
    def validate_model_content(self, model_path: Path) -> Dict[str, Any]:
        """Validate model file content for security issues."""
        validation_info = {
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
            "file_hash": self._compute_file_hash(model_path),
            "is_pickle": False,
            "security_warnings": []
        }
        
        # Check file extension and content
        suffix = model_path.suffix.lower()
        
        if suffix in ['.pth', '.pt']:
            # PyTorch model - check if it's pickle-based
            if self._is_pickle_file(model_path):
                validation_info["is_pickle"] = True
                if not self.config.allow_pickle_loading:
                    raise ValidationError(
                        "Pickle-based model loading is disabled for security",
                        error_code="PICKLE_LOADING_DISABLED",
                        details={"file_path": str(model_path)}
                    )
                validation_info["security_warnings"].append(
                    "Model uses pickle format which can execute arbitrary code"
                )
        
        elif suffix == '.onnx':
            # ONNX model - generally safer
            validation_info["security_warnings"].append(
                "ONNX format is generally safe but verify model source"
            )
        
        else:
            validation_info["security_warnings"].append(
                f"Unknown model format: {suffix}"
            )
        
        return validation_info
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _is_pickle_file(self, file_path: Path) -> bool:
        """Check if file contains pickle data."""
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to check for pickle magic
                header = f.read(10)
                # Pickle files typically start with these bytes
                pickle_markers = [b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05']
                return any(header.startswith(marker) for marker in pickle_markers)
        except Exception:
            return False


class InputSanitizer:
    """Sanitize compilation inputs."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
    
    def sanitize_input_shape(self, input_shape: Union[tuple, list]) -> tuple:
        """Sanitize input tensor shape."""
        if not isinstance(input_shape, (tuple, list)):
            raise ValidationError(
                "Input shape must be tuple or list",
                error_code="INVALID_SHAPE_TYPE"
            )
        
        # Convert to tuple
        shape = tuple(input_shape)
        
        # Check dimensions
        if len(shape) > self.config.max_input_dimensions:
            raise ValidationError(
                f"Too many input dimensions: {len(shape)} > {self.config.max_input_dimensions}",
                error_code="TOO_MANY_DIMENSIONS"
            )
        
        # Check individual dimensions
        for i, dim in enumerate(shape):
            if not isinstance(dim, int):
                raise ValidationError(
                    f"All shape dimensions must be integers, got {type(dim)} at index {i}",
                    error_code="INVALID_DIMENSION_TYPE"
                )
            
            if dim <= 0:
                raise ValidationError(
                    f"All dimensions must be positive, got {dim} at index {i}",
                    error_code="INVALID_DIMENSION_VALUE"
                )
            
            # Check for extremely large dimensions
            if dim > self.config.max_batch_size and i == 0:
                warnings.warn(f"Large batch size detected: {dim}")
        
        # Check total tensor size
        total_size = 1
        for dim in shape:
            total_size *= dim
            if total_size > self.config.max_tensor_size:
                raise ValidationError(
                    f"Input tensor too large: {total_size} elements > {self.config.max_tensor_size}",
                    error_code="TENSOR_TOO_LARGE"
                )
        
        return shape
    
    def sanitize_compilation_target(self, target: str) -> str:
        """Sanitize compilation target."""
        if not isinstance(target, str):
            raise ValidationError(
                "Target must be a string",
                error_code="INVALID_TARGET_TYPE"
            )
        
        # Clean the target string
        target = target.strip().lower()
        
        # Check against allowed targets
        if target not in self.config.allowed_targets:
            raise ValidationError(
                f"Target '{target}' not allowed. Allowed targets: {self.config.allowed_targets}",
                error_code="TARGET_NOT_ALLOWED"
            )
        
        return target
    
    def sanitize_optimization_level(self, level: int) -> int:
        """Sanitize optimization level."""
        if not isinstance(level, int):
            raise ValidationError(
                "Optimization level must be an integer",
                error_code="INVALID_OPT_LEVEL_TYPE"
            )
        
        # Clamp to valid range
        level = max(0, min(3, level))
        
        return level
    
    def sanitize_time_steps(self, time_steps: int) -> int:
        """Sanitize time steps parameter."""
        if not isinstance(time_steps, int):
            raise ValidationError(
                "Time steps must be an integer",
                error_code="INVALID_TIME_STEPS_TYPE"
            )
        
        # Reasonable limits for time steps
        if time_steps < 1:
            time_steps = 1
        elif time_steps > 1000:
            warnings.warn(f"Very large time steps: {time_steps}")
            time_steps = min(time_steps, 1000)
        
        return time_steps


class GraphSanitizer:
    """Sanitize spike computation graphs."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
    
    def validate_graph_size(self, graph) -> None:
        """Validate graph is not too large."""
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        
        if num_nodes > self.config.max_nodes:
            raise ValidationError(
                f"Graph too large: {num_nodes} nodes > {self.config.max_nodes}",
                error_code="GRAPH_TOO_LARGE"
            )
        
        if num_edges > self.config.max_edges:
            raise ValidationError(
                f"Graph too complex: {num_edges} edges > {self.config.max_edges}",
                error_code="GRAPH_TOO_COMPLEX"
            )
    
    def validate_node_parameters(self, node) -> None:
        """Validate node parameters for security issues."""
        # Check for suspicious parameter values
        for param, value in node.parameters.items():
            if isinstance(value, str):
                # Check for potential code injection
                suspicious_patterns = ['eval(', 'exec(', '__import__', 'open(', 'subprocess']
                if any(pattern in value.lower() for pattern in suspicious_patterns):
                    raise ValidationError(
                        f"Suspicious parameter value detected in node {node.id}: {param}",
                        error_code="SUSPICIOUS_PARAMETER"
                    )
            
            elif isinstance(value, (int, float)):
                # Check for extremely large values that could cause memory issues
                if abs(value) > 1e10:
                    warnings.warn(f"Large parameter value in node {node.id}: {param}={value}")


class SecureCompilationEnvironment:
    """Create a secure environment for model compilation."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.model_sanitizer = ModelSanitizer(security_config)
        self.input_sanitizer = InputSanitizer(security_config)
        self.graph_sanitizer = GraphSanitizer(security_config)
        self.temp_dir = None
    
    def __enter__(self):
        """Set up secure compilation environment."""
        # Create temporary directory for secure operations
        self.temp_dir = tempfile.mkdtemp(prefix="spike_compiler_")
        compiler_logger.logger.info(f"Created secure compilation environment: {self.temp_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up secure environment."""
        if self.temp_dir:
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                compiler_logger.logger.info("Cleaned up secure compilation environment")
            except Exception as e:
                compiler_logger.logger.warning(f"Failed to clean up temp directory: {e}")
    
    def validate_and_sanitize_inputs(
        self,
        model_path: str,
        input_shape: Union[tuple, list],
        target: str,
        optimization_level: int,
        time_steps: int
    ) -> Dict[str, Any]:
        """Validate and sanitize all compilation inputs."""
        sanitized = {}
        
        # Sanitize model path
        sanitized['model_path'] = self.model_sanitizer.sanitize_model_path(model_path)
        
        # Validate model content
        model_validation = self.model_sanitizer.validate_model_content(sanitized['model_path'])
        
        # Log security warnings
        for warning in model_validation["security_warnings"]:
            compiler_logger.logger.warning(f"Model security warning: {warning}")
        
        # Sanitize other inputs
        sanitized['input_shape'] = self.input_sanitizer.sanitize_input_shape(input_shape)
        sanitized['target'] = self.input_sanitizer.sanitize_compilation_target(target)
        sanitized['optimization_level'] = self.input_sanitizer.sanitize_optimization_level(optimization_level)
        sanitized['time_steps'] = self.input_sanitizer.sanitize_time_steps(time_steps)
        
        # Store validation info
        sanitized['model_validation'] = model_validation
        
        compiler_logger.logger.info("Input sanitization completed successfully")
        return sanitized


# Global security configuration
_security_config = None

def get_security_config() -> SecurityConfig:
    """Get global security configuration."""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def create_secure_environment() -> SecureCompilationEnvironment:
    """Create a secure compilation environment."""
    return SecureCompilationEnvironment(get_security_config())


# Security utilities
def is_safe_pickle_loading() -> bool:
    """Check if pickle loading is considered safe in current environment."""
    config = get_security_config()
    return config.allow_pickle_loading


def get_model_hash(model_path: str) -> str:
    """Get secure hash of model file."""
    sanitizer = ModelSanitizer(get_security_config())
    path = Path(model_path)
    return sanitizer._compute_file_hash(path)


def validate_hardware_target(target: str) -> bool:
    """Validate if hardware target is allowed and available."""
    config = get_security_config()
    
    # Check if target is allowed
    if target not in config.allowed_targets:
        return False
    
    # Additional hardware verification if required
    if config.require_hardware_verification:
        # This would integrate with actual hardware verification
        # For now, just log the requirement
        compiler_logger.logger.info(f"Hardware verification required for target: {target}")
    
    return True


class SecurityValidator:
    """Comprehensive security validation for compilation pipeline."""
    
    def __init__(self):
        self.config = get_security_config()
        self.model_sanitizer = ModelSanitizer(self.config)
        self.input_sanitizer = InputSanitizer(self.config)
        self.graph_sanitizer = GraphSanitizer(self.config)
        self.security_incidents = []
    
    def validate_model_security(self, model) -> None:
        """Validate model security."""
        try:
            # Check for suspicious model attributes
            if hasattr(model, '__dict__'):
                for attr_name, attr_value in model.__dict__.items():
                    if callable(attr_value) and attr_name.startswith('_'):
                        compiler_logger.logger.warning(f"Suspicious private method in model: {attr_name}")
        except Exception as e:
            compiler_logger.logger.warning(f"Could not validate model security: {e}")
    
    def validate_input_safety(self, input_shape: tuple) -> None:
        """Validate input safety."""
        self.input_sanitizer.sanitize_input_shape(input_shape)
    
    def validate_graph_size(self, graph) -> None:
        """Validate graph size constraints."""
        self.graph_sanitizer.validate_graph_size(graph)
    
    def validate_graph_complexity(self, graph) -> None:
        """Validate graph complexity."""
        # Additional complexity checks
        if hasattr(graph, 'analyze_resources'):
            resources = graph.analyze_resources()
            if resources.get('total_computation_ops', 0) > 1e12:
                compiler_logger.logger.warning("Very high computational complexity detected")
    
    def validate_memory_requirements(self, graph) -> None:
        """Validate memory requirements are reasonable."""
        if hasattr(graph, 'analyze_resources'):
            resources = graph.analyze_resources()
            memory_gb = resources.get('total_memory_bytes', 0) / (1024**3)
            if memory_gb > 100:  # 100GB limit
                raise ValidationError(
                    f"Excessive memory requirement: {memory_gb:.1f}GB",
                    error_code="EXCESSIVE_MEMORY"
                )
    
    def validate_node_parameters(self, node) -> None:
        """Validate node parameters."""
        self.graph_sanitizer.validate_node_parameters(node)
    
    def check_node_security_constraints(self, node) -> None:
        """Check node-specific security constraints."""
        # Additional node security checks can be added here
        pass
    
    def validate_optimization_safety(self, optimizer) -> None:
        """Validate optimization pipeline safety."""
        if hasattr(optimizer, 'passes'):
            if len(optimizer.passes) > 100:
                compiler_logger.logger.warning("Very long optimization pipeline detected")
    
    def validate_backend_config(self, backend, target: str) -> None:
        """Validate backend configuration."""
        if not validate_hardware_target(target):
            raise ValidationError(
                f"Hardware target not allowed or unavailable: {target}",
                error_code="INVALID_HARDWARE_TARGET"
            )
    
    def validate_compiled_model_safety(self, compiled_model) -> None:
        """Validate compiled model safety."""
        if hasattr(compiled_model, 'energy_per_inference'):
            if compiled_model.energy_per_inference > 1000:  # 1000 nJ limit
                compiler_logger.logger.warning(f"High energy consumption: {compiled_model.energy_per_inference} nJ")
    
    def log_security_incident(self, incident_type: str, details: str) -> None:
        """Log security incident."""
        incident = {
            'type': incident_type,
            'details': details,
            'timestamp': compiler_logger.get_timestamp()
        }
        self.security_incidents.append(incident)
        compiler_logger.logger.error(f"Security incident: {incident_type} - {details}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security validation report."""
        return {
            'total_incidents': len(self.security_incidents),
            'incidents': self.security_incidents,
            'config': {
                'max_model_size_mb': self.config.max_model_size_mb,
                'allowed_targets': self.config.allowed_targets,
                'max_nodes': self.config.max_nodes,
                'max_edges': self.config.max_edges
            }
        }