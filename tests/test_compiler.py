"""Tests for the main compiler interface."""

import pytest
from spike_transformer_compiler import SpikeCompiler


class TestSpikeCompiler:
    """Test cases for SpikeCompiler class."""
    
    def test_compiler_initialization(self):
        """Test compiler initialization with default parameters."""
        compiler = SpikeCompiler()
        assert compiler.target == "loihi3"
        assert compiler.optimization_level == 2
        assert compiler.time_steps == 4
        assert compiler.debug is False
        assert compiler.verbose is False
        
    def test_compiler_custom_parameters(self):
        """Test compiler initialization with custom parameters."""
        compiler = SpikeCompiler(
            target="simulation",
            optimization_level=3,
            time_steps=8,
            debug=True,
            verbose=True,
        )
        assert compiler.target == "simulation"
        assert compiler.optimization_level == 3
        assert compiler.time_steps == 8
        assert compiler.debug is True
        assert compiler.verbose is True
        
    def test_compile_not_implemented(self):
        """Test that compile method raises NotImplementedError."""
        compiler = SpikeCompiler()
        with pytest.raises(NotImplementedError):
            compiler.compile(None, (1, 3, 224, 224))
            
    def test_create_optimizer_not_implemented(self):
        """Test that create_optimizer method raises NotImplementedError."""
        compiler = SpikeCompiler()
        with pytest.raises(NotImplementedError):
            compiler.create_optimizer()
            
    def test_set_debug_options(self):
        """Test setting debug options."""
        compiler = SpikeCompiler()
        compiler.set_debug_options(dump_ir=True, dump_passes=True)
        assert compiler.debug_dump_ir is True
        assert compiler.debug_dump_passes is True