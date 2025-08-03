"""Backend factory for creating target-specific backends."""

from typing import Optional, Any
from .simulation_backend import SimulationBackend


class BackendFactory:
    """Factory for creating backend instances."""
    
    @staticmethod
    def create_backend(
        target: str,
        chip_config: Optional[str] = None,
        resource_allocator: Optional[Any] = None
    ) -> Any:
        """Create backend for target hardware."""
        
        if target == "simulation":
            return SimulationBackend(resource_allocator=resource_allocator)
        elif target == "loihi3":
            try:
                from .loihi3_backend import Loihi3Backend
                return Loihi3Backend(
                    chip_config=chip_config,
                    resource_allocator=resource_allocator
                )
            except ImportError:
                print("Warning: Loihi3 backend not available, falling back to simulation")
                return SimulationBackend(resource_allocator=resource_allocator)
        else:
            raise ValueError(f"Unsupported target: {target}")
            
    @staticmethod
    def get_available_targets() -> list:
        """Get list of available compilation targets."""
        targets = ["simulation"]
        
        try:
            import nxsdk
            targets.append("loihi3")
        except ImportError:
            pass
            
        return targets