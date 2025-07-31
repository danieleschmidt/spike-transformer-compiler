"""Tests for optimization passes."""

import pytest
from spike_transformer_compiler.optimization import OptimizationPass, Optimizer


class TestOptimizationPass:
    """Test cases for OptimizationPass enum."""
    
    def test_optimization_pass_values(self):
        """Test that optimization pass enum values are correct."""
        assert OptimizationPass.SPIKE_COMPRESSION.value == "spike_compression"
        assert OptimizationPass.WEIGHT_QUANTIZATION.value == "weight_quantization"
        assert OptimizationPass.NEURON_PRUNING.value == "neuron_pruning"
        assert OptimizationPass.TEMPORAL_FUSION.value == "temporal_fusion"


class TestOptimizer:
    """Test cases for Optimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = Optimizer()
        assert optimizer.passes == []
        
    def test_add_pass(self):
        """Test adding optimization passes."""
        optimizer = Optimizer()
        optimizer.add_pass(OptimizationPass.SPIKE_COMPRESSION)
        optimizer.add_pass(OptimizationPass.WEIGHT_QUANTIZATION, bits=4)
        
        assert len(optimizer.passes) == 2
        assert optimizer.passes[0][0] == OptimizationPass.SPIKE_COMPRESSION
        assert optimizer.passes[1][0] == OptimizationPass.WEIGHT_QUANTIZATION
        assert optimizer.passes[1][1] == {"bits": 4}
        
    def test_run_optimizer(self):
        """Test running optimization pipeline."""
        optimizer = Optimizer()
        optimizer.add_pass(OptimizationPass.SPIKE_COMPRESSION)
        
        # For now, this should return the input unchanged
        dummy_ir = "dummy_ir"
        result = optimizer.run(dummy_ir)
        assert result == dummy_ir