"""Exception classes for spike transformer compiler."""


class SpikeCompilerError(Exception):
    """Base exception for spike transformer compiler."""
    pass


class CompilationError(SpikeCompilerError):
    """Raised when compilation fails."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(SpikeCompilerError):
    """Raised when configuration is invalid."""
    pass


class BackendError(SpikeCompilerError):
    """Raised when backend operation fails."""
    pass


class HardwareError(SpikeCompilerError):
    """Raised when hardware operation fails."""
    pass


class OptimizationError(SpikeCompilerError):
    """Raised when optimization fails."""
    pass