"""Configuration management for Spike-Transformer-Compiler."""

import os
import json
import yaml
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from .exceptions import ConfigurationError
from .logging_config import compiler_logger


@dataclass
class CompilerConfig:
    """Main compiler configuration."""
    
    # Default compilation settings
    default_target: str = "simulation"
    default_optimization_level: int = 2
    default_time_steps: int = 4
    enable_debug: bool = False
    enable_verbose: bool = False
    
    # Hardware settings
    hardware_timeout_seconds: int = 300
    max_hardware_retries: int = 3
    hardware_verification_required: bool = False
    
    # Performance settings
    max_compilation_time_seconds: int = 3600
    max_memory_usage_gb: float = 8.0
    enable_parallel_compilation: bool = True
    compilation_cache_enabled: bool = True
    cache_directory: str = "~/.spike_compiler_cache"
    
    # Optimization settings
    optimization_passes: Dict[str, bool] = field(default_factory=lambda: {
        "dead_code_elimination": True,
        "common_subexpression_elimination": True,
        "spike_fusion": True,
        "memory_optimization": True,
        "temporal_fusion": True
    })
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "spike_compiler.log"
    metrics_enabled: bool = True
    performance_profiling: bool = False
    
    # Security settings
    security_mode: bool = True
    allow_unsafe_operations: bool = False
    model_verification_required: bool = True
    input_sanitization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CompilerConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.default_optimization_level < 0 or self.default_optimization_level > 3:
            raise ConfigurationError(
                f"Invalid optimization level: {self.default_optimization_level}",
                error_code="INVALID_OPTIMIZATION_LEVEL"
            )
        
        if self.default_time_steps < 1 or self.default_time_steps > 1000:
            raise ConfigurationError(
                f"Invalid time steps: {self.default_time_steps}",
                error_code="INVALID_TIME_STEPS"
            )
        
        if self.max_compilation_time_seconds <= 0:
            raise ConfigurationError(
                f"Invalid compilation timeout: {self.max_compilation_time_seconds}",
                error_code="INVALID_TIMEOUT"
            )
        
        if self.max_memory_usage_gb <= 0:
            raise ConfigurationError(
                f"Invalid memory limit: {self.max_memory_usage_gb}",
                error_code="INVALID_MEMORY_LIMIT"
            )
        
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in log_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.log_level}. Must be one of {log_levels}",
                error_code="INVALID_LOG_LEVEL"
            )


@dataclass
class TargetConfig:
    """Configuration for specific compilation targets."""
    
    name: str
    enabled: bool = True
    priority: int = 0
    
    # Hardware-specific settings
    max_neurons: int = 1000000
    max_synapses: int = 10000000
    cores_per_chip: int = 128
    max_chips: int = 1
    
    # Backend settings
    backend_module: str = ""
    backend_class: str = ""
    config_file: Optional[str] = None
    
    # Optimization settings
    target_specific_passes: List[str] = field(default_factory=list)
    energy_model: str = "default"
    
    def validate(self) -> None:
        """Validate target configuration."""
        if self.max_neurons <= 0:
            raise ConfigurationError(
                f"Invalid max neurons for target {self.name}: {self.max_neurons}",
                error_code="INVALID_TARGET_CONFIG"
            )


class ConfigurationManager:
    """Manage compiler configuration from multiple sources."""
    
    def __init__(self):
        self.config: CompilerConfig = CompilerConfig()
        self.target_configs: Dict[str, TargetConfig] = {}
        self._config_sources: List[str] = []
        
        # Load configuration in order of precedence
        self._load_default_config()
        self._load_environment_config()
        self._load_file_configs()
        
        # Validate final configuration
        self.config.validate()
        
        compiler_logger.logger.info(f"Configuration loaded from: {', '.join(self._config_sources)}")
    
    def _load_default_config(self):
        """Load default configuration."""
        # Default config is already set in CompilerConfig dataclass
        self._config_sources.append("defaults")
        
        # Load default target configurations
        self.target_configs = {
            "simulation": TargetConfig(
                name="simulation",
                enabled=True,
                priority=1,
                backend_module="spike_transformer_compiler.backend.simulation_backend",
                backend_class="SimulationBackend"
            ),
            "loihi3": TargetConfig(
                name="loihi3",
                enabled=True,
                priority=2,
                max_neurons=1024000,
                max_synapses=100000000,
                cores_per_chip=128,
                backend_module="spike_transformer_compiler.backend.loihi3_backend",
                backend_class="Loihi3Backend",
                energy_model="loihi3"
            )
        }
    
    def _load_environment_config(self):
        """Load configuration from environment variables."""
        env_mapping = {
            "SPIKE_DEFAULT_TARGET": "default_target",
            "SPIKE_OPTIMIZATION_LEVEL": ("default_optimization_level", int),
            "SPIKE_TIME_STEPS": ("default_time_steps", int),
            "SPIKE_DEBUG": ("enable_debug", lambda x: x.lower() == "true"),
            "SPIKE_VERBOSE": ("enable_verbose", lambda x: x.lower() == "true"),
            "SPIKE_LOG_LEVEL": "log_level",
            "SPIKE_LOG_TO_FILE": ("log_to_file", lambda x: x.lower() == "true"),
            "SPIKE_CACHE_ENABLED": ("compilation_cache_enabled", lambda x: x.lower() == "true"),
            "SPIKE_CACHE_DIR": "cache_directory",
            "SPIKE_MAX_MEMORY_GB": ("max_memory_usage_gb", float),
            "SPIKE_COMPILATION_TIMEOUT": ("max_compilation_time_seconds", int),
            "SPIKE_SECURITY_MODE": ("security_mode", lambda x: x.lower() == "true"),
            "SPIKE_METRICS_ENABLED": ("metrics_enabled", lambda x: x.lower() == "true"),
        }
        
        env_loaded = False
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                if isinstance(config_key, tuple):
                    attr_name, converter = config_key
                    try:
                        value = converter(value)
                    except (ValueError, TypeError) as e:
                        compiler_logger.logger.warning(
                            f"Invalid environment variable {env_var}={value}: {e}"
                        )
                        continue
                else:
                    attr_name = config_key
                
                setattr(self.config, attr_name, value)
                env_loaded = True
        
        if env_loaded:
            self._config_sources.append("environment")
    
    def _load_file_configs(self):
        """Load configuration from files."""
        config_paths = [
            Path.cwd() / "spike_compiler.yaml",
            Path.cwd() / "spike_compiler.json",
            Path.home() / ".config" / "spike_compiler" / "config.yaml",
            Path.home() / ".spike_compiler.yaml",
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    self._load_config_file(config_path)
                    self._config_sources.append(str(config_path))
                    break  # Use first found config file
                except Exception as e:
                    compiler_logger.logger.warning(
                        f"Failed to load config file {config_path}: {e}"
                    )
    
    def _load_config_file(self, config_path: Path):
        """Load configuration from a specific file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {config_path.suffix}",
                    error_code="UNSUPPORTED_CONFIG_FORMAT"
                )
        
        # Update main configuration
        if 'compiler' in config_data:
            self._update_config_from_dict(config_data['compiler'])
        
        # Update target configurations
        if 'targets' in config_data:
            for target_name, target_config in config_data['targets'].items():
                if target_name in self.target_configs:
                    # Update existing target config
                    for key, value in target_config.items():
                        setattr(self.target_configs[target_name], key, value)
                else:
                    # Create new target config
                    self.target_configs[target_name] = TargetConfig(
                        name=target_name,
                        **target_config
                    )
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                compiler_logger.logger.warning(f"Unknown configuration key: {key}")
    
    def get_compiler_config(self) -> CompilerConfig:
        """Get main compiler configuration."""
        return self.config
    
    def get_target_config(self, target_name: str) -> Optional[TargetConfig]:
        """Get configuration for specific target."""
        return self.target_configs.get(target_name)
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets."""
        return [name for name, config in self.target_configs.items() if config.enabled]
    
    def update_config(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ConfigurationError(
                    f"Unknown configuration key: {key}",
                    error_code="UNKNOWN_CONFIG_KEY"
                )
        
        # Re-validate after updates
        self.config.validate()
    
    def save_config(self, config_path: Path, format: str = "yaml"):
        """Save current configuration to file."""
        config_data = {
            'compiler': self.config.to_dict(),
            'targets': {
                name: asdict(config) for name, config in self.target_configs.items()
            }
        }
        
        with open(config_path, 'w') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_data, f, indent=2)
            else:
                raise ConfigurationError(
                    f"Unsupported save format: {format}",
                    error_code="UNSUPPORTED_SAVE_FORMAT"
                )
        
        compiler_logger.logger.info(f"Configuration saved to {config_path}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = CompilerConfig()
        self._load_default_config()
        compiler_logger.logger.info("Configuration reset to defaults")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "compiler_config": self.config.to_dict(),
            "available_targets": self.get_available_targets(),
            "config_sources": self._config_sources,
            "target_count": len(self.target_configs)
        }


# Global configuration manager
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_compiler_config() -> CompilerConfig:
    """Get compiler configuration."""
    return get_config_manager().get_compiler_config()


def get_target_config(target_name: str) -> Optional[TargetConfig]:
    """Get target-specific configuration."""
    return get_config_manager().get_target_config(target_name)


def update_config(**kwargs):
    """Update global configuration."""
    get_config_manager().update_config(**kwargs)


def save_config(config_path: Union[str, Path], format: str = "yaml"):
    """Save configuration to file."""
    path = Path(config_path)
    get_config_manager().save_config(path, format)


def reset_config():
    """Reset configuration to defaults."""
    global _config_manager
    _config_manager = None
    # Next call to get_config_manager() will create a new instance


# Configuration validation utilities
def validate_target_availability(target: str) -> bool:
    """Check if target is available and properly configured."""
    config_manager = get_config_manager()
    target_config = config_manager.get_target_config(target)
    
    if not target_config or not target_config.enabled:
        return False
    
    # Additional validation could check if backend modules are available
    try:
        if target_config.backend_module:
            __import__(target_config.backend_module)
        return True
    except ImportError:
        compiler_logger.logger.warning(
            f"Backend module not available for target {target}: {target_config.backend_module}"
        )
        return False


def get_optimization_passes(target: str) -> List[str]:
    """Get enabled optimization passes for target."""
    compiler_config = get_compiler_config()
    target_config = get_target_config(target)
    
    # Start with general optimization passes
    enabled_passes = [
        pass_name for pass_name, enabled in compiler_config.optimization_passes.items()
        if enabled
    ]
    
    # Add target-specific passes
    if target_config and target_config.target_specific_passes:
        enabled_passes.extend(target_config.target_specific_passes)
    
    return enabled_passes


# Example configuration file template
EXAMPLE_CONFIG = {
    "compiler": {
        "default_target": "simulation",
        "default_optimization_level": 2,
        "default_time_steps": 4,
        "log_level": "INFO",
        "metrics_enabled": True,
        "security_mode": True,
        "max_memory_usage_gb": 8.0
    },
    "targets": {
        "simulation": {
            "enabled": True,
            "priority": 1,
            "max_neurons": 1000000
        },
        "loihi3": {
            "enabled": True,
            "priority": 2,
            "max_neurons": 1024000,
            "cores_per_chip": 128,
            "energy_model": "loihi3"
        }
    }
}


def create_example_config(config_path: Union[str, Path]):
    """Create an example configuration file."""
    path = Path(config_path)
    
    with open(path, 'w') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(EXAMPLE_CONFIG, f, default_flow_style=False, indent=2)
        else:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
    
    print(f"Example configuration created at {path}")