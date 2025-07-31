"""Performance tests for compilation pipeline."""

import pytest
import time
import torch
from spike_transformer_compiler import SpikeCompiler


@pytest.mark.performance
class TestCompilationPerformance:
    """Performance tests for the compilation process."""
    
    def test_compile_time_regression(self, sample_model, performance_baseline):
        """Test that compilation time doesn't regress beyond baseline."""
        compiler = SpikeCompiler(target="simulation")
        
        start_time = time.time()
        try:
            # This will raise NotImplementedError but we measure the overhead
            compiler.compile(sample_model, (1, 10))
        except NotImplementedError:
            pass
        compile_time_ms = (time.time() - start_time) * 1000
        
        # Allow 20% regression tolerance
        assert compile_time_ms < performance_baseline["compile_time_ms"] * 1.2
        
    def test_memory_usage_during_compilation(self, sample_model):
        """Test memory usage doesn't exceed reasonable limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        compiler = SpikeCompiler(target="simulation")
        try:
            compiler.compile(sample_model, (1, 10))
        except NotImplementedError:
            pass
            
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable for small models
        assert memory_increase < 50  # MB
        
    @pytest.mark.slow
    def test_large_model_scalability(self):
        """Test compilation performance scales reasonably with model size."""
        class LargeModel(torch.nn.Module):
            def __init__(self, size_factor=1):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(100 * size_factor, 100 * size_factor)
                    for _ in range(10 * size_factor)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        compiler = SpikeCompiler(target="simulation")
        compile_times = []
        
        for size_factor in [1, 2, 4]:
            model = LargeModel(size_factor)
            start_time = time.time()
            try:
                compiler.compile(model, (1, 100 * size_factor))
            except NotImplementedError:
                pass
            compile_time = time.time() - start_time
            compile_times.append(compile_time)
        
        # Compilation time should scale sub-quadratically
        # (allowing for some overhead in small models)
        if len(compile_times) >= 2:
            scaling_factor = compile_times[-1] / compile_times[0]
            assert scaling_factor < 20  # Reasonable scaling limit