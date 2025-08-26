"""Comprehensive test suite for breakthrough research systems."""

import pytest
import numpy as np
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('../src')

from spike_transformer_compiler.neuromorphic_breakthrough_engine import (
    BreakthroughOrchestrator, TemporalPlasticityEngine, SynapticDynamicsBreakthrough,
    EmergentComputationEngine, BreakthroughMetrics
)
from spike_transformer_compiler.adaptive_neural_evolution import (
    AdaptiveNeuralEvolution, ArchitectureGenome, NeuromorphicArchitectureBuilder,
    FitnessEvaluator, EvolutionStrategy
)


class TestBreakthroughOrchestrator:
    """Test breakthrough orchestrator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = BreakthroughOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator is not None
        assert hasattr(self.orchestrator, 'plasticity_engine')
        assert hasattr(self.orchestrator, 'dynamics_engine')
        assert hasattr(self.orchestrator, 'emergence_engine')
        assert len(self.orchestrator.results_history) == 0
    
    def test_run_breakthrough_experiment(self):
        """Test running a breakthrough experiment."""
        config = {
            'id': 'test_experiment',
            'algorithm': 'adaptive_stdp',
            'num_neurons': 50,
            'duration': 500
        }
        
        result = self.orchestrator.run_breakthrough_experiment(config)
        
        assert isinstance(result, BreakthroughMetrics)
        assert result.algorithm_name == 'adaptive_stdp'
        assert result.innovation_index >= 0
        assert len(self.orchestrator.results_history) == 1
    
    def test_generate_test_spike_trains(self):
        """Test spike train generation."""
        spike_trains = self.orchestrator._generate_test_spike_trains(10, 1000)
        
        assert len(spike_trains) == 10
        for train in spike_trains:
            assert isinstance(train, np.ndarray)
            assert all(spike < 1000 for spike in train)
    
    def test_run_plasticity_experiment(self):
        """Test plasticity experiment execution."""
        spike_trains = self.orchestrator._generate_test_spike_trains(5, 1000)
        config = {'duration': 1000}
        
        result = self.orchestrator._run_plasticity_experiment(
            'adaptive_stdp', spike_trains, config
        )
        
        assert 'weight_changes' in result
        assert 'convergence_metrics' in result
        assert len(result['weight_changes']) > 0
    
    @pytest.mark.asyncio
    async def test_parallel_experiments(self):
        """Test parallel experiment execution."""
        experiment_configs = [
            {'id': f'parallel_exp_{i}', 'algorithm': 'adaptive_stdp', 'num_neurons': 20}
            for i in range(3)
        ]
        
        results = await self.orchestrator.run_parallel_experiments(experiment_configs)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, BreakthroughMetrics)
    
    def test_get_best_algorithms(self):
        """Test getting best performing algorithms."""
        # Run some experiments first
        for i in range(3):
            config = {
                'id': f'test_exp_{i}',
                'algorithm': 'adaptive_stdp',
                'num_neurons': 20
            }
            self.orchestrator.run_breakthrough_experiment(config)
        
        best_algorithms = self.orchestrator.get_best_algorithms(top_k=2)
        
        assert len(best_algorithms) <= 2
        # Check if sorted by innovation index
        if len(best_algorithms) > 1:
            assert (best_algorithms[0]['metrics'].innovation_index >= 
                   best_algorithms[1]['metrics'].innovation_index)


class TestTemporalPlasticityEngine:
    """Test temporal plasticity engine."""
    
    def setup_method(self):
        """Set up test environment."""
        self.engine = TemporalPlasticityEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        assert len(self.engine.plasticity_rules) > 0
        assert 'adaptive_stdp' in self.engine.plasticity_rules
        assert 'meta_plasticity' in self.engine.plasticity_rules
    
    def test_adaptive_stdp(self):
        """Test adaptive STDP implementation."""
        pre_spikes = np.array([10, 30, 50])
        post_spikes = np.array([15, 35, 55])
        context = {'duration': 100, 'weights': [0.1, 0.2, 0.3]}
        
        weight_changes = self.engine._adaptive_stdp(pre_spikes, post_spikes, context)
        
        assert isinstance(weight_changes, np.ndarray)
        assert len(weight_changes) == len(context['weights'])
    
    def test_meta_plasticity(self):
        """Test meta-plasticity implementation."""
        pre_spikes = np.array([10, 30])
        post_spikes = np.array([15, 35])
        context = {
            'duration': 100, 
            'weights': [0.1, 0.2],
            'plasticity_history': [[0.01, 0.02], [0.015, 0.025], [0.02, 0.03]]
        }
        
        weight_changes = self.engine._meta_plasticity(pre_spikes, post_spikes, context)
        
        assert isinstance(weight_changes, np.ndarray)
        assert len(weight_changes) == len(context['weights'])
    
    def test_homeostatic_stdp(self):
        """Test homeostatic STDP."""
        pre_spikes = np.array([10, 30, 50])
        post_spikes = np.array([15, 35])  # Lower firing rate
        context = {
            'duration': 1000,
            'weights': [0.1, 0.2],
            'target_firing_rate': 5.0
        }
        
        weight_changes = self.engine._homeostatic_stdp(pre_spikes, post_spikes, context)
        
        assert isinstance(weight_changes, np.ndarray)
        # Should adjust for lower firing rate
        assert np.any(weight_changes != 0)
    
    def test_quantum_inspired_stdp(self):
        """Test quantum-inspired STDP."""
        pre_spikes = np.array([10, 30])
        post_spikes = np.array([15, 35])
        context = {
            'duration': 100,
            'weights': np.array([0.1, 0.2, 0.3]),
            'quantum_phase': np.pi / 4
        }
        
        weight_changes = self.engine._quantum_inspired_stdp(pre_spikes, post_spikes, context)
        
        assert isinstance(weight_changes, np.ndarray)
        assert len(weight_changes) > 0


class TestSynapticDynamicsBreakthrough:
    """Test synaptic dynamics breakthrough systems."""
    
    def setup_method(self):
        """Set up test environment."""
        self.dynamics = SynapticDynamicsBreakthrough()
    
    def test_initialization(self):
        """Test dynamics engine initialization."""
        assert len(self.dynamics.dynamics_models) > 0
        assert 'liquid_state' in self.dynamics.dynamics_models
        assert 'echo_state' in self.dynamics.dynamics_models
    
    def test_liquid_state_dynamics(self):
        """Test liquid state machine dynamics."""
        inputs = np.random.randn(10)
        state = {'reservoir_size': 100}
        
        output, new_state = self.dynamics._liquid_state_dynamics(inputs, state)
        
        assert isinstance(output, np.ndarray)
        assert len(output) > 0
        assert 'reservoir_weights' in new_state
        assert 'internal_state' in new_state
    
    def test_echo_state_dynamics(self):
        """Test echo state network dynamics."""
        inputs = np.random.randn(8)
        state = {}
        
        output, new_state = self.dynamics._echo_state_dynamics(inputs, state)
        
        assert isinstance(output, np.ndarray)
        assert 'echo_reservoir' in new_state
        assert 'echo_state' in new_state
    
    def test_reservoir_dynamics(self):
        """Test multi-timescale reservoir dynamics."""
        inputs = np.random.randn(5)
        state = {}
        
        output, new_state = self.dynamics._reservoir_dynamics(inputs, state)
        
        assert isinstance(output, np.ndarray)
        assert len(output) > 0  # Should have concatenated outputs
        assert 'multi_reservoir' in new_state
    
    def test_memory_network_dynamics(self):
        """Test memory network dynamics."""
        inputs = np.random.randn(12)
        state = {}
        
        output, new_state = self.dynamics._memory_network_dynamics(inputs, state)
        
        assert isinstance(output, np.ndarray)
        assert len(output) == len(inputs)
        assert 'memory_matrix' in new_state
        assert 'memory_strengths' in new_state


class TestEmergentComputationEngine:
    """Test emergent computation engine."""
    
    def setup_method(self):
        """Set up test environment."""
        self.engine = EmergentComputationEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        assert len(self.engine.emergence_detectors) > 0
        assert 'synchronization' in self.engine.emergence_detectors
        assert 'oscillations' in self.engine.emergence_detectors
    
    def test_detect_synchronization(self):
        """Test synchronization detection."""
        # Create synchronized spike trains
        base_times = np.array([10, 30, 50, 70])
        spike_trains = [
            base_times + np.random.normal(0, 1, len(base_times)),
            base_times + np.random.normal(0, 1, len(base_times)),
            base_times + np.random.normal(0, 1, len(base_times))
        ]
        
        sync_score = self.engine._detect_synchronization(spike_trains)
        
        assert 0 <= sync_score <= 1
        # Synchronized trains should have higher score than random
        assert sync_score > 0.3
    
    def test_detect_oscillations(self):
        """Test oscillation detection."""
        # Create periodic spike pattern
        frequency = 40  # Hz
        duration = 1000  # ms
        period = 1000 / frequency
        
        spike_trains = []
        for i in range(3):
            times = np.arange(0, duration, period) + i * 2
            spike_trains.append(times)
        
        osc_score = self.engine._detect_oscillations(spike_trains)
        
        assert 0 <= osc_score <= 1
    
    def test_detect_avalanches(self):
        """Test neural avalanche detection."""
        # Create avalanche-like spike pattern
        spike_trains = []
        avalanche_times = [10, 11, 12, 50, 51, 52, 53, 100, 101]
        
        for i in range(3):
            times = np.array(avalanche_times) + i * 0.5
            spike_trains.append(times)
        
        avalanche_score = self.engine._detect_avalanches(spike_trains)
        
        assert 0 <= avalanche_score <= 1
    
    def test_detect_criticality(self):
        """Test overall criticality detection."""
        spike_trains = [
            np.array([10, 30, 50, 70]),
            np.array([12, 32, 52, 72]),
            np.array([11, 31, 51, 71])
        ]
        
        criticality = self.engine._detect_criticality(spike_trains)
        
        assert 0 <= criticality <= 1


class TestAdaptiveNeuralEvolution:
    """Test adaptive neural evolution system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.evolution = AdaptiveNeuralEvolution(population_size=10)
    
    def test_initialization(self):
        """Test evolution system initialization."""
        assert self.evolution.population_size == 10
        assert len(self.evolution.population) == 0
        assert self.evolution.generation == 0
    
    def test_initialize_population(self):
        """Test population initialization."""
        self.evolution.initialize_population()
        
        assert len(self.evolution.population) == 10
        for genome in self.evolution.population:
            assert isinstance(genome, ArchitectureGenome)
            assert genome.num_layers >= 2
            assert len(genome.layer_sizes) == genome.num_layers
    
    def test_create_random_genome(self):
        """Test random genome creation."""
        genome = self.evolution._create_random_genome()
        
        assert isinstance(genome, ArchitectureGenome)
        assert 2 <= genome.num_layers <= 8
        assert len(genome.layer_sizes) == genome.num_layers
        assert len(genome.connectivity_patterns) == genome.num_layers - 1
    
    def test_evolve_generation(self):
        """Test single generation evolution."""
        self.evolution.initialize_population()
        initial_generation = self.evolution.generation
        
        self.evolution.evolve_generation()
        
        assert self.evolution.generation == initial_generation + 1
        assert len(self.evolution.population) == 10
    
    def test_genetic_algorithm_reproduction(self):
        """Test genetic algorithm reproduction."""
        self.evolution.initialize_population()
        
        # Manually set fitness for testing
        for i, genome in enumerate(self.evolution.population):
            genome.fitness = float(i) / 10.0  # Increasing fitness
        
        new_population = self.evolution._genetic_algorithm_reproduction()
        
        assert len(new_population) == self.evolution.population_size
        # Best individuals should be preserved (elitism)
        best_fitness = max(genome.fitness for genome in new_population)
        assert best_fitness >= 0.8  # High fitness should be preserved
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        self.evolution.initialize_population()
        
        # Set different fitness values
        for i, genome in enumerate(self.evolution.population):
            genome.fitness = float(i)
        
        selected = self.evolution._tournament_selection(self.evolution.population)
        
        assert selected in self.evolution.population
        # Should tend to select higher fitness individuals
        assert selected.fitness >= 0.0
    
    def test_crossover(self):
        """Test genetic crossover operation."""
        parent1 = self.evolution._create_random_genome()
        parent2 = self.evolution._create_random_genome()
        
        child1, child2 = self.evolution._crossover(parent1, parent2)
        
        assert isinstance(child1, ArchitectureGenome)
        assert isinstance(child2, ArchitectureGenome)
        # Children should have different genome IDs
        assert child1.genome_id != child2.genome_id
    
    def test_mutation(self):
        """Test genetic mutation operation."""
        original_genome = self.evolution._create_random_genome()
        mutated_genome = self.evolution._mutate(original_genome)
        
        assert isinstance(mutated_genome, ArchitectureGenome)
        assert mutated_genome.genome_id != original_genome.genome_id
        # Some difference should exist after mutation
        assert (mutated_genome.num_layers != original_genome.num_layers or
                mutated_genome.layer_sizes != original_genome.layer_sizes)
    
    def test_run_evolution(self):
        """Test full evolution run."""
        best_genome = self.evolution.run_evolution(num_generations=3)
        
        assert isinstance(best_genome, ArchitectureGenome)
        assert self.evolution.generation == 3
        assert len(self.evolution.evolution_history) == 3


class TestArchitectureGenome:
    """Test architecture genome functionality."""
    
    def test_genome_creation(self):
        """Test genome creation with default values."""
        genome = ArchitectureGenome()
        
        assert len(genome.genome_id) == 8  # MD5 hash prefix
        assert genome.num_layers == 4
        assert len(genome.layer_sizes) == 4
        assert genome.fitness == 0.0
        assert genome.generation == 0
    
    def test_genome_with_custom_values(self):
        """Test genome creation with custom values."""
        layer_sizes = [128, 64, 32]
        neuron_models = ['LIF', 'AdEx', 'Izhikevich']
        
        genome = ArchitectureGenome(
            num_layers=3,
            layer_sizes=layer_sizes,
            neuron_models=neuron_models
        )
        
        assert genome.num_layers == 3
        assert genome.layer_sizes == layer_sizes
        assert genome.neuron_models == neuron_models


class TestNeuromorphicArchitectureBuilder:
    """Test neuromorphic architecture builder."""
    
    def setup_method(self):
        """Set up test environment."""
        self.builder = NeuromorphicArchitectureBuilder()
    
    def test_initialization(self):
        """Test builder initialization."""
        assert len(self.builder.neuron_model_registry) > 0
        assert len(self.builder.connectivity_registry) > 0
        assert 'LIF' in self.builder.neuron_model_registry
        assert 'dense' in self.builder.connectivity_registry
    
    def test_build_architecture(self):
        """Test architecture building from genome."""
        genome = ArchitectureGenome(
            num_layers=3,
            layer_sizes=[10, 20, 5],
            neuron_models=['LIF', 'AdEx', 'LIF']
        )
        
        architecture = self.builder.build_architecture(genome)
        
        assert 'genome_id' in architecture
        assert 'layers' in architecture
        assert 'connections' in architecture
        assert len(architecture['layers']) == 3
        assert len(architecture['connections']) == 2  # n-1 connections
        assert architecture['metadata']['total_neurons'] == 35
    
    def test_create_connection(self):
        """Test connection creation between layers."""
        pre_layer = {'layer_id': 0, 'size': 10}
        post_layer = {'layer_id': 1, 'size': 5}
        
        connection = self.builder._create_connection(
            pre_layer, post_layer, 'dense', 0.8, 1.0, 'STDP', 0.01
        )
        
        assert connection['pre_layer'] == 0
        assert connection['post_layer'] == 1
        assert connection['pattern'] == 'dense'
        assert connection['density'] == 0.8
        assert 'connection_matrix' in connection
        assert 'weights' in connection
    
    def test_connectivity_patterns(self):
        """Test different connectivity patterns."""
        patterns = ['dense', 'sparse', 'topological', 'random']
        
        for pattern in patterns:
            matrix = getattr(self.builder, f'_create_{pattern}_connectivity')(10, 8, 0.5)
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (10, 8)


class TestFitnessEvaluator:
    """Test fitness evaluation system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.evaluator = FitnessEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization."""
        assert len(self.evaluator.evaluation_metrics) > 0
        assert 'classification_accuracy' in self.evaluator.evaluation_metrics
        assert 'energy_efficiency' in self.evaluator.evaluation_metrics
    
    def test_evaluate_architecture(self):
        """Test architecture evaluation."""
        builder = NeuromorphicArchitectureBuilder()
        genome = ArchitectureGenome(num_layers=3, layer_sizes=[20, 10, 5])
        architecture = builder.build_architecture(genome)
        
        fitness, metrics = self.evaluator.evaluate_architecture(architecture)
        
        assert isinstance(fitness, float)
        assert 0 <= fitness <= 1
        assert isinstance(metrics, dict)
        assert 'classification_accuracy' in metrics
        assert 'energy_efficiency' in metrics
    
    def test_generate_test_data(self):
        """Test test data generation."""
        config = {'dataset_size': 100, 'num_classes': 5}
        test_data = self.evaluator._generate_test_data(config)
        
        assert 'patterns' in test_data
        assert 'labels' in test_data
        assert test_data['patterns'].shape[0] == 100
        assert len(np.unique(test_data['labels'])) <= 5
    
    def test_detect_weight_anomalies(self):
        """Test weight anomaly detection."""
        model_data = {
            'weights': {
                'layer1': np.random.randn(100, 50),
                'layer2': np.ones((50, 10)) * 1000  # Extreme weights
            }
        }
        
        threats = self.evaluator._detect_weight_anomalies(model_data)
        
        assert len(threats) > 0  # Should detect extreme weights
        assert any('extreme_weights' in threat.threat_id for threat in threats)
    
    def test_architectural_anomalies(self):
        """Test architectural anomaly detection."""
        model_data = {
            'architecture': {
                'layers': [{'size': 100000} for _ in range(2000)]  # Very large
            }
        }
        
        threats = self.evaluator._detect_architectural_anomalies(model_data)
        
        assert len(threats) > 0  # Should detect anomalies
    
    def test_calculate_composite_fitness(self):
        """Test composite fitness calculation."""
        metrics = {
            'classification_accuracy': 0.8,
            'energy_efficiency': 0.7,
            'spike_efficiency': 0.6,
            'convergence_rate': 0.9,
            'robustness': 0.7,
            'novelty': 0.5
        }
        
        fitness = self.evaluator._calculate_composite_fitness(metrics, {})
        
        assert 0 <= fitness <= 1
        # Should be weighted average of metrics
        assert fitness > 0.5  # Given good metrics


class TestEvolutionIntegration:
    """Integration tests for evolution system."""
    
    @pytest.mark.slow
    def test_full_evolution_cycle(self):
        """Test complete evolution cycle."""
        evolution = AdaptiveNeuralEvolution(population_size=5, strategy=EvolutionStrategy.GENETIC_ALGORITHM)
        
        # Run evolution
        best_genome = evolution.run_evolution(num_generations=2, target_fitness=0.5)
        
        assert isinstance(best_genome, ArchitectureGenome)
        assert best_genome.fitness >= 0.0
        assert evolution.generation >= 2 or best_genome.fitness >= 0.5
    
    def test_different_evolution_strategies(self):
        """Test different evolution strategies."""
        strategies = [EvolutionStrategy.GENETIC_ALGORITHM, EvolutionStrategy.EVOLUTIONARY_STRATEGY]
        
        for strategy in strategies:
            evolution = AdaptiveNeuralEvolution(population_size=5, strategy=strategy)
            evolution.initialize_population()
            
            initial_gen = evolution.generation
            evolution.evolve_generation()
            
            assert evolution.generation == initial_gen + 1
    
    def test_export_evolution_results(self):
        """Test evolution results export."""
        evolution = AdaptiveNeuralEvolution(population_size=5)
        evolution.initialize_population()
        evolution.evolve_generation()
        
        # Mock file operations
        with patch('builtins.open', mock_open()):
            with patch('json.dump') as mock_dump:
                evolution.export_evolution_results('test_results.json')
                mock_dump.assert_called_once()


def mock_open(*args, **kwargs):
    """Mock file open function."""
    return MagicMock()


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_breakthrough_experiment_performance(self, benchmark):
        """Benchmark breakthrough experiment performance."""
        orchestrator = BreakthroughOrchestrator()
        config = {
            'id': 'benchmark_test',
            'algorithm': 'adaptive_stdp',
            'num_neurons': 100,
            'duration': 1000
        }
        
        result = benchmark(orchestrator.run_breakthrough_experiment, config)
        assert isinstance(result, BreakthroughMetrics)
    
    @pytest.mark.benchmark
    def test_evolution_generation_performance(self, benchmark):
        """Benchmark evolution generation performance."""
        evolution = AdaptiveNeuralEvolution(population_size=20)
        evolution.initialize_population()
        
        benchmark(evolution.evolve_generation)
        assert evolution.generation == 1
    
    @pytest.mark.benchmark
    def test_architecture_evaluation_performance(self, benchmark):
        """Benchmark architecture evaluation performance."""
        builder = NeuromorphicArchitectureBuilder()
        evaluator = FitnessEvaluator()
        
        genome = ArchitectureGenome(num_layers=4, layer_sizes=[100, 200, 100, 50])
        architecture = builder.build_architecture(genome)
        
        fitness, metrics = benchmark(evaluator.evaluate_architecture, architecture)
        assert isinstance(fitness, float)


# Integration tests
class TestSystemIntegration:
    """System integration tests."""
    
    def test_breakthrough_to_evolution_integration(self):
        """Test integration between breakthrough engine and evolution."""
        # Run breakthrough experiments
        orchestrator = BreakthroughOrchestrator()
        experiment_configs = [
            {'id': f'integration_exp_{i}', 'algorithm': 'adaptive_stdp', 'num_neurons': 20}
            for i in range(3)
        ]
        
        breakthrough_results = []
        for config in experiment_configs:
            result = orchestrator.run_breakthrough_experiment(config)
            breakthrough_results.append(result)
        
        # Use results to inform evolution
        evolution = AdaptiveNeuralEvolution(population_size=10)
        evolution.initialize_population()
        
        # Set fitness based on breakthrough results
        for i, genome in enumerate(evolution.population[:len(breakthrough_results)]):
            genome.fitness = breakthrough_results[i % len(breakthrough_results)].innovation_index
        
        # Evolve
        evolution.evolve_generation()
        
        assert evolution.generation == 1
        assert any(genome.fitness > 0 for genome in evolution.population)
    
    def test_architecture_builder_evaluation_integration(self):
        """Test integration between architecture builder and evaluator."""
        builder = NeuromorphicArchitectureBuilder()
        evaluator = FitnessEvaluator()
        
        # Create diverse genomes
        genomes = []
        for i in range(5):
            genome = ArchitectureGenome(
                num_layers=3 + i,
                layer_sizes=[50 + i*10, 30 + i*5, 10 + i]
            )
            genomes.append(genome)
        
        # Build and evaluate architectures
        results = []
        for genome in genomes:
            architecture = builder.build_architecture(genome)
            fitness, metrics = evaluator.evaluate_architecture(architecture)
            results.append((fitness, metrics))
        
        assert len(results) == 5
        for fitness, metrics in results:
            assert 0 <= fitness <= 1
            assert len(metrics) > 0


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=spike_transformer_compiler.neuromorphic_breakthrough_engine",
        "--cov=spike_transformer_compiler.adaptive_neural_evolution",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "-m", "not slow"
    ])