"""Main compiler interface for Spike-Transformer-Compiler."""

from typing import Any, Dict, Optional, Union
import torch.nn as nn
from .exceptions import (
    CompilationError, ValidationError, ValidationUtils, ErrorContext, ErrorRecovery
)
from .logging_config import compiler_logger, HealthMonitor
from .security import create_secure_environment, SecurityValidator
from .performance import PerformanceProfiler, ResourceMonitor


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
        # Validate initialization parameters
        from .backend.factory import BackendFactory
        available_targets = BackendFactory.get_available_targets()
        
        ValidationUtils.validate_target(target, available_targets)
        ValidationUtils.validate_optimization_level(optimization_level)
        ValidationUtils.validate_time_steps(time_steps)
        
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
        secure_mode: bool = True,
    ) -> "CompiledModel":
        """Compile a PyTorch model to target hardware.
        
        Args:
            model: PyTorch model to compile
            input_shape: Input tensor shape
            chip_config: Hardware chip configuration
            optimizer: Optional optimization pipeline
            resource_allocator: Optional resource allocation strategy
            profile_energy: Enable energy profiling
            secure_mode: Enable security checks and sanitization
            
        Returns:
            CompiledModel: Compiled model ready for deployment
            
        Raises:
            CompilationError: If compilation fails
        """
        # Set up secure compilation environment if requested
        if secure_mode:
            # Note: For live models, we can't sanitize file path, but we validate the model itself
            from .security import get_security_config, InputSanitizer, GraphSanitizer
            security_config = get_security_config()
            input_sanitizer = InputSanitizer(security_config)
            graph_sanitizer = GraphSanitizer(security_config)
            
            # Sanitize inputs
            input_shape = input_sanitizer.sanitize_input_shape(input_shape)
            target = input_sanitizer.sanitize_compilation_target(self.target)
            self.target = target  # Update with sanitized value
            
        # Initialize comprehensive monitoring
        health_monitor = HealthMonitor()
        health_monitor.start_monitoring()
        
        perf_profiler = PerformanceProfiler()
        resource_monitor = ResourceMonitor()
        security_validator = SecurityValidator() if secure_mode else None
        
        perf_profiler.start_compilation_profiling()
        
        metrics = compiler_logger.start_compilation(
            target=self.target,
            model_type=type(model).__name__,
            input_shape=input_shape,
            optimization_level=self.optimization_level
        )
        
        # Comprehensive input validation
        with ErrorContext("input_validation", target=self.target, 
                         optimization_level=self.optimization_level):
            ValidationUtils.validate_model(model)
            ValidationUtils.validate_input_shape(input_shape)
        
        # Log model information
        compiler_logger.log_model_info(model, input_shape)
        
        from .frontend.pytorch_parser import PyTorchParser
        from .ir.builder import SpikeIRBuilder
        from .ir.passes import PassManager, DeadCodeElimination, SpikeFusion
        from .backend.factory import BackendFactory
        
        if self.verbose:
            print(f"Compiling model for {self.target} target...")
            print(f"Input shape: {input_shape}")
            print(f"Optimization level: {self.optimization_level}")
            print(f"Time steps: {self.time_steps}")
            
        try:
            # Stage 1: Frontend parsing with enhanced validation
            with compiler_logger.time_operation("frontend_parsing"):
                with ErrorContext("frontend_parsing", model_type=type(model).__name__):
                    parser = PyTorchParser()
                    
                    # Security validation of model if enabled
                    if security_validator:
                        security_validator.validate_model_security(model)
                        security_validator.validate_input_safety(input_shape)
                    
                    # Performance monitoring
                    with perf_profiler.profile_stage("frontend_parsing"):
                        spike_graph = parser.parse_model(model, input_shape, self.time_steps)
                    
                    # Resource monitoring
                    resource_monitor.log_memory_usage("after_parsing")
                    
                    # Enhanced security validation of generated graph
                    if secure_mode:
                        graph_sanitizer.validate_graph_size(spike_graph)
                        graph_sanitizer.validate_graph_complexity(spike_graph)
                        graph_sanitizer.validate_memory_requirements(spike_graph)
                        
                        for node in spike_graph.nodes:
                            graph_sanitizer.validate_node_parameters(node)
                            graph_sanitizer.check_node_security_constraints(node)
                    
                    compiler_logger.log_compilation_stage(
                        "Frontend Parsing",
                        nodes=len(spike_graph.nodes),
                        edges=len(spike_graph.edges),
                        model_type=type(model).__name__
                    )
                    
                    if self.verbose:
                        print(f"✓ Frontend parsing: {len(spike_graph.nodes)} nodes, {len(spike_graph.edges)} edges")
            
            health_monitor.update_peak_memory()
            
            # Stage 2: Enhanced optimization passes with monitoring
            with compiler_logger.time_operation("optimization"):
                with ErrorContext("optimization", optimization_level=self.optimization_level):
                    if optimizer is None:
                        optimizer = self.create_optimizer()
                    
                    # Security check on optimization parameters
                    if security_validator:
                        security_validator.validate_optimization_safety(optimizer)
                    
                    with perf_profiler.profile_stage("optimization"):
                        # Log pre-optimization state
                        pre_optimization = {
                            'nodes': len(spike_graph.nodes),
                            'edges': len(spike_graph.edges)
                        }
                        
                        # Run optimization with resource monitoring
                        resource_monitor.log_memory_usage("before_optimization")
                        optimized_graph = optimizer.run_all(spike_graph)
                        resource_monitor.log_memory_usage("after_optimization")
                    
                    # Log optimization results
                    post_optimization = {
                        'nodes': len(optimized_graph.nodes),
                        'edges': len(optimized_graph.edges)
                    }
                    
                    compiler_logger.log_optimization_results(pre_optimization, post_optimization)
                    compiler_logger.log_compilation_stage(
                        "Optimization",
                        optimization_level=self.optimization_level,
                        nodes_before=pre_optimization['nodes'],
                        nodes_after=post_optimization['nodes']
                    )
                    
                    if self.verbose:
                        print(f"✓ Optimization: {len(optimized_graph.nodes)} nodes")
                        
            health_monitor.update_peak_memory()
                    
            # Stage 3: Enhanced backend code generation
            with compiler_logger.time_operation("backend_compilation"):
                with ErrorContext("backend_compilation", target=self.target):
                    # Create backend with security validation
                    backend = BackendFactory.create_backend(
                        self.target, 
                        chip_config=chip_config,
                        resource_allocator=resource_allocator
                    )
                    
                    # Security validation of backend configuration
                    if security_validator:
                        security_validator.validate_backend_config(backend, self.target)
                    
                    with perf_profiler.profile_stage("backend_compilation"):
                        resource_monitor.log_memory_usage("before_backend_compilation")
                        compiled_model = backend.compile_graph(
                            optimized_graph,
                            profile_energy=profile_energy,
                            debug=self.debug
                        )
                        resource_monitor.log_memory_usage("after_backend_compilation")
                    
                    # Log backend results
                    compiler_logger.log_compilation_stage(
                        "Backend Compilation",
                        target=self.target,
                        energy_per_inference=getattr(compiled_model, 'energy_per_inference', 0),
                        utilization=getattr(compiled_model, 'utilization', 0)
                    )
                    
                    if self.verbose:
                        print("✓ Backend compilation completed successfully")
                        if hasattr(compiled_model, 'energy_per_inference'):
                            print(f"  Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
                        if hasattr(compiled_model, 'utilization'):
                            print(f"  Hardware utilization: {compiled_model.utilization:.1%}")
            
            # Comprehensive resource and performance logging
            memory_stats = health_monitor.get_memory_stats()
            perf_stats = perf_profiler.get_compilation_stats()
            resource_stats = resource_monitor.get_resource_summary()
            
            compiler_logger.log_resource_usage(**memory_stats)
            compiler_logger.log_performance_metrics(perf_stats)
            compiler_logger.log_resource_summary(resource_stats)
            
            # Final security validation
            if security_validator:
                security_validator.validate_compiled_model_safety(compiled_model)
            
            # Mark compilation as successful
            compiler_logger.end_compilation(success=True)
            perf_profiler.end_compilation_profiling()
            
            return compiled_model
            
        except ValidationError as e:
            # Enhanced error logging and cleanup
            compiler_logger.end_compilation(success=False, error=str(e))
            perf_profiler.end_compilation_profiling(failed=True)
            resource_monitor.log_compilation_failure("validation_error")
            
            # Re-raise validation errors with additional context
            suggestion = ErrorRecovery.suggest_fix_for_shape_error(input_shape, 'image_2d')
            raise CompilationError(
                f"Validation failed: {str(e)}\n{suggestion}",
                error_code="VALIDATION_FAILED"
            ) from e
            
        except Exception as e:
            # Comprehensive error handling and cleanup
            compiler_logger.end_compilation(success=False, error=str(e))
            perf_profiler.end_compilation_profiling(failed=True)
            resource_monitor.log_compilation_failure("general_error")
            
            # Security logging if enabled
            if security_validator:
                security_validator.log_security_incident("compilation_failure", str(e))
            
            # Provide fallback suggestions for other errors
            fallback_msg = ErrorRecovery.suggest_target_fallback(
                self.target, BackendFactory.get_available_targets()
            )
            
            # Collect comprehensive error context
            error_details = {
                'target': self.target,
                'model_type': type(model).__name__,
                'input_shape': input_shape,
                'optimization_level': self.optimization_level,
                'memory_stats': memory_stats if 'memory_stats' in locals() else {},
                'performance_stats': perf_stats if 'perf_stats' in locals() else {},
                'compilation_stage': 'unknown'
            }
            
            raise CompilationError(
                f"Compilation failed: {str(e)}\n{fallback_msg}",
                error_code="COMPILATION_FAILED",
                details=error_details
            ) from e
        
        finally:
            # Cleanup monitoring resources
            health_monitor.stop_monitoring()
            perf_profiler.cleanup()
            resource_monitor.cleanup()
        
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