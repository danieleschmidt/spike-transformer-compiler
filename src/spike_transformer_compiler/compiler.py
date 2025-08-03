"""Main compiler interface for Spike-Transformer-Compiler."""

from typing import Any, Dict, Optional, Union
import torch.nn as nn


class CompilationError(Exception):
    """Exception raised when compilation fails."""
    pass


class SpikeCompiler:
    """Main compiler interface for converting SpikeFormer models to neuromorphic binaries.
    
    This class provides the primary API for compiling PyTorch SpikeFormer models
    to target neuromorphic hardware like Intel Loihi 3.
    
    Args:
        target: Target hardware platform ("loihi3", "simulation")
        optimization_level: Optimization level (0-3)
        time_steps: Number of time steps for spiking simulation
        debug: Enable debug mode
        verbose: Enable verbose output
    """
    
    def __init__(
        self,
        target: str = "loihi3",
        optimization_level: int = 2,
        time_steps: int = 4,
        debug: bool = False,
        verbose: bool = False,
    ):
        self.target = target
        self.optimization_level = optimization_level
        self.time_steps = time_steps
        self.debug = debug
        self.verbose = verbose
        
    def compile(
        self,
        model: nn.Module,
        input_shape: tuple,
        chip_config: Optional[str] = None,
        optimizer: Optional[Any] = None,
        resource_allocator: Optional[Any] = None,
        profile_energy: bool = False,
    ) -> "CompiledModel":
        """Compile a PyTorch model to target hardware.
        
        Args:
            model: PyTorch model to compile
            input_shape: Input tensor shape
            chip_config: Hardware chip configuration
            optimizer: Optional optimization pipeline
            resource_allocator: Optional resource allocation strategy
            profile_energy: Enable energy profiling
            
        Returns:
            CompiledModel: Compiled model ready for deployment
            
        Raises:
            CompilationError: If compilation fails
        """
        from .frontend.pytorch_parser import PyTorchParser
        from .ir.builder import SpikeIRBuilder
        from .ir.passes import PassManager, DeadCodeElimination, SpikeFusion
        from .backend.factory import BackendFactory
        
        if self.verbose:
            print(f"Compiling model for {self.target} target...")
            
        try:
            # Stage 1: Frontend parsing
            parser = PyTorchParser()
            spike_graph = parser.parse_model(model, input_shape, self.time_steps)
            
            if self.verbose:
                print(f"Parsed model: {len(spike_graph.nodes)} nodes, {len(spike_graph.edges)} edges")
            
            # Stage 2: Optimization passes
            if optimizer is None:
                optimizer = self.create_optimizer()
                
            optimized_graph = optimizer.run_all(spike_graph)
            
            if self.verbose:
                print(f"Optimized model: {len(optimized_graph.nodes)} nodes")
                
            # Stage 3: Backend code generation
            backend = BackendFactory.create_backend(
                self.target, 
                chip_config=chip_config,
                resource_allocator=resource_allocator
            )
            
            compiled_model = backend.compile_graph(
                optimized_graph,
                profile_energy=profile_energy,
                debug=self.debug
            )
            
            if self.verbose:
                print("Compilation completed successfully")
                
            return compiled_model
            
        except Exception as e:
            raise CompilationError(f"Compilation failed: {str(e)}") from e
        
    def create_optimizer(self) -> "PassManager":
        """Create optimization pipeline for the compiler."""
        from .ir.passes import PassManager, DeadCodeElimination, SpikeFusion, CommonSubexpressionElimination, MemoryOptimization
        
        optimizer = PassManager()
        
        # Add optimization passes based on optimization level
        if self.optimization_level >= 1:
            optimizer.add_pass(DeadCodeElimination())
            
        if self.optimization_level >= 2:
            optimizer.add_pass(CommonSubexpressionElimination())
            optimizer.add_pass(SpikeFusion())
            
        if self.optimization_level >= 3:
            optimizer.add_pass(MemoryOptimization())
            
        return optimizer
        
    def set_debug_options(self, dump_ir: bool = False, dump_passes: bool = False) -> None:
        """Set debug options for compilation pipeline."""
        self.debug_dump_ir = dump_ir
        self.debug_dump_passes = dump_passes


class CompiledModel:
    """Represents a compiled neuromorphic model ready for deployment."""
    
    def __init__(self):
        self.energy_per_inference = 0.0
        self.utilization = 0.0
        
    def run(self, input_data: Any, time_steps: int = 4, return_spike_trains: bool = False) -> Any:
        """Run inference on the compiled model."""
        if not hasattr(self, 'executor'):
            raise RuntimeError("Model not properly compiled - missing executor")
            
        # Execute the model
        output = self.executor.run(
            input_data, 
            time_steps=time_steps,
            return_spike_trains=return_spike_trains
        )
        
        # Update energy metrics if profiling enabled
        if hasattr(self.executor, 'get_energy_consumption'):
            self.energy_per_inference = self.executor.get_energy_consumption()
            
        return output
        
    def debug_trace(self) -> None:
        """Show execution trace for debugging."""
        if not hasattr(self, 'executor'):
            print("Model not compiled - no execution trace available")
            return
            
        if hasattr(self.executor, 'get_debug_trace'):
            trace = self.executor.get_debug_trace()
            print("Execution Trace:")
            print("=" * 50)
            for step in trace:
                print(f"Step {step['step']}: {step['operation']} - {step['duration']:.3f}ms")
                if 'spikes' in step:
                    print(f"  Spikes: {step['spikes']} events")
                if 'memory' in step:
                    print(f"  Memory: {step['memory']} bytes")
            print("=" * 50)
        else:
            print("Debug tracing not available for this backend")