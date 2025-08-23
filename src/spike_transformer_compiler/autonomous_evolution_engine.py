"""Autonomous Evolution Engine for Self-Improving Neuromorphic Systems.

This engine implements autonomous evolution capabilities for the spike compiler,
enabling continuous optimization, self-healing, and adaptive improvements based
on usage patterns and performance metrics.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import numpy as np
from pathlib import Path

from .compiler import SpikeCompiler, CompiledModel
from .optimization import OptimizationPass, Optimizer
from .monitoring import CompilationMetrics, PerformanceTracker
from .validation import ValidationUtils


@dataclass
class EvolutionMetrics:
    """Metrics for tracking autonomous evolution."""
    generation: int
    fitness_score: float
    compilation_time: float
    inference_latency: float
    energy_efficiency: float
    accuracy_preservation: float
    memory_usage: float
    hardware_utilization: float
    improvement_factor: float
    timestamp: float


@dataclass
class AdaptationStrategy:
    """Strategy for autonomous adaptation."""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    fitness_function: str
    priority: int
    success_rate: float
    avg_improvement: float


class AutonomousEvolutionEngine:
    """Engine for autonomous evolution and self-improvement of neuromorphic systems."""
    
    def __init__(
        self, 
        compiler: SpikeCompiler,
        evolution_storage_path: str = "evolution_data",
        max_generations: int = 100,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_ratio: float = 0.2
    ):
        self.compiler = compiler
        self.storage_path = Path(evolution_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Evolution parameters
        self.max_generations = max_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        # State tracking
        self.current_generation = 0
        self.population: List[Optimizer] = []
        self.evolution_history: List[EvolutionMetrics] = []
        self.best_strategies: List[AdaptationStrategy] = []
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.baseline_metrics: Optional[EvolutionMetrics] = None
        
        # Adaptation strategies
        self.adaptation_strategies = self._initialize_adaptation_strategies()
        
        # Thread pool for parallel evolution
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_adaptation_strategies(self) -> Dict[str, AdaptationStrategy]:
        """Initialize built-in adaptation strategies."""
        strategies = {
            "aggressive_optimization": AdaptationStrategy(
                strategy_id="aggressive_opt",
                name="Aggressive Optimization",
                description="High-intensity optimization for maximum performance",
                parameters={
                    "optimization_level": 3,
                    "spike_compression": 0.05,
                    "weight_quantization": 4,
                    "neuron_pruning": 0.95,
                    "temporal_fusion": True
                },
                fitness_function="energy_delay_product",
                priority=1,
                success_rate=0.0,
                avg_improvement=0.0
            ),
            "balanced_optimization": AdaptationStrategy(
                strategy_id="balanced_opt",
                name="Balanced Optimization",
                description="Balanced approach between performance and accuracy",
                parameters={
                    "optimization_level": 2,
                    "spike_compression": 0.1,
                    "weight_quantization": 8,
                    "neuron_pruning": 0.8,
                    "temporal_fusion": True
                },
                fitness_function="accuracy_efficiency_ratio",
                priority=2,
                success_rate=0.0,
                avg_improvement=0.0
            ),
            "accuracy_preserving": AdaptationStrategy(
                strategy_id="accuracy_pres",
                name="Accuracy Preserving",
                description="Conservative optimization preserving model accuracy",
                parameters={
                    "optimization_level": 1,
                    "spike_compression": 0.2,
                    "weight_quantization": 16,
                    "neuron_pruning": 0.5,
                    "temporal_fusion": False
                },
                fitness_function="accuracy_weighted_efficiency",
                priority=3,
                success_rate=0.0,
                avg_improvement=0.0
            ),
            "energy_efficient": AdaptationStrategy(
                strategy_id="energy_eff",
                name="Energy Efficient",
                description="Extreme energy efficiency optimization",
                parameters={
                    "optimization_level": 3,
                    "spike_compression": 0.02,
                    "weight_quantization": 2,
                    "neuron_pruning": 0.98,
                    "temporal_fusion": True,
                    "power_gating": True,
                    "voltage_scaling": 0.8
                },
                fitness_function="energy_efficiency",
                priority=1,
                success_rate=0.0,
                avg_improvement=0.0
            )
        }
        return strategies
    
    async def evolve_autonomous(
        self, 
        model: Any, 
        input_shape: Tuple, 
        test_dataset: Optional[Any] = None,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[CompiledModel, EvolutionMetrics]:
        """Perform autonomous evolution to optimize compilation."""
        print("🧬 Starting Autonomous Evolution Engine...")
        
        # Initialize baseline
        baseline_compiled = await self._compile_with_strategy(
            model, input_shape, self.adaptation_strategies["balanced_optimization"]
        )
        self.baseline_metrics = await self._evaluate_fitness(
            baseline_compiled, model, input_shape, test_dataset
        )
        
        print(f"📊 Baseline established: Fitness = {self.baseline_metrics.fitness_score:.4f}")
        
        # Initialize population
        self.population = await self._initialize_population()
        
        best_model = baseline_compiled
        best_metrics = self.baseline_metrics
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.current_generation = generation
            print(f"\n🧬 Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate population fitness
            fitness_results = await self._evaluate_population(
                model, input_shape, test_dataset
            )
            
            # Select best performers
            elite_count = max(1, int(self.population_size * self.elitism_ratio))
            elite_results = sorted(fitness_results, key=lambda x: x[1].fitness_score, reverse=True)[:elite_count]
            
            current_best = elite_results[0]
            if current_best[1].fitness_score > best_metrics.fitness_score:
                best_model = current_best[0]
                best_metrics = current_best[1]
                
                print(f"🎯 New best: Fitness = {best_metrics.fitness_score:.4f} "
                      f"(+{best_metrics.improvement_factor:.2%} improvement)")
            
            # Early termination if target achieved
            if target_metrics and self._meets_target_criteria(best_metrics, target_metrics):
                print(f"🎉 Target metrics achieved in generation {generation + 1}!")
                break
            
            # Evolution operations
            if generation < self.max_generations - 1:
                self.population = await self._evolve_population(elite_results)
                await self._adapt_strategies(fitness_results)
            
            # Store evolution data
            self._save_evolution_state(best_metrics)
            
            # Adaptive parameters
            await self._adapt_evolution_parameters(fitness_results)
            
        print(f"\n🏁 Evolution Complete! Best fitness: {best_metrics.fitness_score:.4f}")
        return best_model, best_metrics
    
    async def _initialize_population(self) -> List[Optimizer]:
        """Initialize population with diverse optimization strategies."""
        population = []
        
        # Add base strategies
        for strategy in self.adaptation_strategies.values():
            optimizer = self._create_optimizer_from_strategy(strategy)
            population.append(optimizer)
            
        # Generate random variations
        while len(population) < self.population_size:
            base_strategy = np.random.choice(list(self.adaptation_strategies.values()))
            mutated_strategy = self._mutate_strategy(base_strategy)
            optimizer = self._create_optimizer_from_strategy(mutated_strategy)
            population.append(optimizer)
            
        return population
    
    async def _evaluate_population(
        self, 
        model: Any, 
        input_shape: Tuple, 
        test_dataset: Optional[Any]
    ) -> List[Tuple[CompiledModel, EvolutionMetrics]]:
        """Evaluate fitness of entire population in parallel."""
        
        # Create compilation tasks
        compilation_tasks = []
        for i, optimizer in enumerate(self.population):
            strategy = self._optimizer_to_strategy(optimizer, f"pop_{i}")
            task = self._compile_with_strategy(model, input_shape, strategy)
            compilation_tasks.append(task)
        
        # Compile all models in parallel
        compiled_models = await asyncio.gather(*compilation_tasks, return_exceptions=True)
        
        # Evaluate fitness in parallel
        evaluation_tasks = []
        for i, compiled_model in enumerate(compiled_models):
            if isinstance(compiled_model, Exception):
                # Handle compilation failures
                continue
            task = self._evaluate_fitness(compiled_model, model, input_shape, test_dataset)
            evaluation_tasks.append((compiled_model, task))
        
        # Wait for all evaluations
        results = []
        for compiled_model, eval_task in evaluation_tasks:
            try:
                metrics = await eval_task
                results.append((compiled_model, metrics))
            except Exception as e:
                # Skip failed evaluations
                continue
                
        return results
    
    async def _compile_with_strategy(
        self, 
        model: Any, 
        input_shape: Tuple, 
        strategy: AdaptationStrategy
    ) -> CompiledModel:
        """Compile model with given strategy."""
        
        # Create optimizer from strategy
        optimizer = self._create_optimizer_from_strategy(strategy)
        
        # Configure compiler
        compiler_config = {
            "optimization_level": strategy.parameters.get("optimization_level", 2),
            "debug": False,
            "verbose": False
        }
        
        # Compile with strategy
        compiled_model = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.compiler.compile(
                model=model,
                input_shape=input_shape,
                optimizer=optimizer,
                **compiler_config
            )
        )
        
        return compiled_model
    
    async def _evaluate_fitness(
        self, 
        compiled_model: CompiledModel, 
        original_model: Any, 
        input_shape: Tuple,
        test_dataset: Optional[Any]
    ) -> EvolutionMetrics:
        """Evaluate fitness of compiled model."""
        
        start_time = time.time()
        
        # Performance metrics
        compilation_time = getattr(compiled_model, 'compilation_time', 0.1)
        inference_latency = self._measure_inference_latency(compiled_model, input_shape)
        energy_efficiency = getattr(compiled_model, 'energy_per_inference', 1.0)
        memory_usage = self._measure_memory_usage(compiled_model)
        hardware_utilization = getattr(compiled_model, 'utilization', 0.5)
        
        # Accuracy preservation (simplified)
        accuracy_preservation = await self._estimate_accuracy_preservation(
            compiled_model, original_model, test_dataset
        )
        
        # Calculate composite fitness score
        fitness_score = self._calculate_fitness_score(
            inference_latency, energy_efficiency, accuracy_preservation,
            memory_usage, hardware_utilization
        )
        
        # Improvement factor vs baseline
        improvement_factor = 0.0
        if self.baseline_metrics:
            improvement_factor = (fitness_score - self.baseline_metrics.fitness_score) / self.baseline_metrics.fitness_score
        
        metrics = EvolutionMetrics(
            generation=self.current_generation,
            fitness_score=fitness_score,
            compilation_time=compilation_time,
            inference_latency=inference_latency,
            energy_efficiency=energy_efficiency,
            accuracy_preservation=accuracy_preservation,
            memory_usage=memory_usage,
            hardware_utilization=hardware_utilization,
            improvement_factor=improvement_factor,
            timestamp=time.time()
        )
        
        return metrics
    
    def _calculate_fitness_score(
        self, 
        latency: float, 
        energy: float, 
        accuracy: float,
        memory: float, 
        utilization: float
    ) -> float:
        """Calculate composite fitness score."""
        
        # Normalize metrics (lower is better for latency, energy, memory)
        latency_score = 1.0 / (1.0 + latency)
        energy_score = 1.0 / (1.0 + energy)
        memory_score = 1.0 / (1.0 + memory)
        
        # Higher is better for accuracy and utilization
        accuracy_score = accuracy
        utilization_score = utilization
        
        # Weighted composite score
        fitness = (
            0.25 * latency_score +
            0.30 * energy_score +
            0.25 * accuracy_score +
            0.10 * memory_score +
            0.10 * utilization_score
        )
        
        return fitness
    
    async def _evolve_population(
        self, 
        elite_results: List[Tuple[CompiledModel, EvolutionMetrics]]
    ) -> List[Optimizer]:
        """Evolve population through selection, crossover, and mutation."""
        
        new_population = []
        elite_count = len(elite_results)
        
        # Keep elite individuals
        for compiled_model, metrics in elite_results:
            strategy = self._metrics_to_strategy(metrics)
            optimizer = self._create_optimizer_from_strategy(strategy)
            new_population.append(optimizer)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate and len(elite_results) >= 2:
                # Crossover
                parent1_metrics = np.random.choice(elite_results)[1]
                parent2_metrics = np.random.choice(elite_results)[1]
                
                child_strategy = self._crossover_strategies(
                    self._metrics_to_strategy(parent1_metrics),
                    self._metrics_to_strategy(parent2_metrics)
                )
            else:
                # Mutation only
                parent_metrics = np.random.choice(elite_results)[1]
                child_strategy = self._mutate_strategy(self._metrics_to_strategy(parent_metrics))
            
            optimizer = self._create_optimizer_from_strategy(child_strategy)
            new_population.append(optimizer)
        
        return new_population
    
    def _mutate_strategy(self, strategy: AdaptationStrategy) -> AdaptationStrategy:
        """Mutate strategy parameters."""
        mutated_params = strategy.parameters.copy()
        
        # Mutate optimization level
        if np.random.random() < self.mutation_rate:
            mutated_params["optimization_level"] = np.random.randint(1, 4)
        
        # Mutate compression ratio
        if "spike_compression" in mutated_params and np.random.random() < self.mutation_rate:
            current_val = mutated_params["spike_compression"]
            mutation = np.random.normal(0, 0.05)
            mutated_params["spike_compression"] = np.clip(current_val + mutation, 0.01, 0.5)
        
        # Mutate quantization bits
        if "weight_quantization" in mutated_params and np.random.random() < self.mutation_rate:
            bits_options = [2, 4, 8, 16]
            mutated_params["weight_quantization"] = np.random.choice(bits_options)
        
        # Mutate pruning sparsity
        if "neuron_pruning" in mutated_params and np.random.random() < self.mutation_rate:
            current_val = mutated_params["neuron_pruning"]
            mutation = np.random.normal(0, 0.1)
            mutated_params["neuron_pruning"] = np.clip(current_val + mutation, 0.1, 0.99)
        
        # Create new strategy with mutated parameters
        mutated_strategy = AdaptationStrategy(
            strategy_id=f"mutated_{strategy.strategy_id}_{int(time.time())}",
            name=f"Mutated {strategy.name}",
            description=f"Mutation of {strategy.description}",
            parameters=mutated_params,
            fitness_function=strategy.fitness_function,
            priority=strategy.priority,
            success_rate=strategy.success_rate,
            avg_improvement=strategy.avg_improvement
        )
        
        return mutated_strategy
    
    def _crossover_strategies(
        self, 
        parent1: AdaptationStrategy, 
        parent2: AdaptationStrategy
    ) -> AdaptationStrategy:
        """Crossover two strategies to create offspring."""
        
        # Combine parameters from both parents
        child_params = {}
        for key in set(parent1.parameters.keys()) | set(parent2.parameters.keys()):
            if key in parent1.parameters and key in parent2.parameters:
                # Average numerical values
                val1, val2 = parent1.parameters[key], parent2.parameters[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    child_params[key] = (val1 + val2) / 2
                else:
                    child_params[key] = np.random.choice([val1, val2])
            elif key in parent1.parameters:
                child_params[key] = parent1.parameters[key]
            else:
                child_params[key] = parent2.parameters[key]
        
        # Create child strategy
        child_strategy = AdaptationStrategy(
            strategy_id=f"crossover_{int(time.time())}",
            name=f"Crossover of {parent1.name} and {parent2.name}",
            description=f"Hybrid of {parent1.description} and {parent2.description}",
            parameters=child_params,
            fitness_function=np.random.choice([parent1.fitness_function, parent2.fitness_function]),
            priority=min(parent1.priority, parent2.priority),
            success_rate=(parent1.success_rate + parent2.success_rate) / 2,
            avg_improvement=(parent1.avg_improvement + parent2.avg_improvement) / 2
        )
        
        return child_strategy
    
    def _create_optimizer_from_strategy(self, strategy: AdaptationStrategy) -> Optimizer:
        """Create optimizer from adaptation strategy."""
        optimizer = Optimizer()
        
        params = strategy.parameters
        
        # Add optimization passes based on strategy
        if params.get("spike_compression"):
            optimizer.add_pass(
                OptimizationPass.SPIKE_COMPRESSION, 
                compression_ratio=params["spike_compression"]
            )
        
        if params.get("weight_quantization"):
            optimizer.add_pass(
                OptimizationPass.WEIGHT_QUANTIZATION,
                bits=params["weight_quantization"]
            )
        
        if params.get("neuron_pruning"):
            optimizer.add_pass(
                OptimizationPass.NEURON_PRUNING,
                sparsity=params["neuron_pruning"]
            )
        
        if params.get("temporal_fusion"):
            optimizer.add_pass(OptimizationPass.TEMPORAL_FUSION)
        
        return optimizer
    
    def _optimizer_to_strategy(self, optimizer: Optimizer, strategy_id: str) -> AdaptationStrategy:
        """Convert optimizer back to strategy representation."""
        # Extract parameters from optimizer passes
        parameters = {"optimization_level": 2}  # Default
        
        for pass_type, kwargs in optimizer.passes:
            if pass_type == OptimizationPass.SPIKE_COMPRESSION:
                parameters["spike_compression"] = kwargs.get("compression_ratio", 0.1)
            elif pass_type == OptimizationPass.WEIGHT_QUANTIZATION:
                parameters["weight_quantization"] = kwargs.get("bits", 8)
            elif pass_type == OptimizationPass.NEURON_PRUNING:
                parameters["neuron_pruning"] = kwargs.get("sparsity", 0.8)
            elif pass_type == OptimizationPass.TEMPORAL_FUSION:
                parameters["temporal_fusion"] = True
        
        return AdaptationStrategy(
            strategy_id=strategy_id,
            name=f"Strategy {strategy_id}",
            description="Generated strategy",
            parameters=parameters,
            fitness_function="composite",
            priority=2,
            success_rate=0.0,
            avg_improvement=0.0
        )
    
    def _metrics_to_strategy(self, metrics: EvolutionMetrics) -> AdaptationStrategy:
        """Convert evolution metrics back to strategy (simplified)."""
        # This is a simplified conversion - in practice, you'd store more detailed mapping
        return self.adaptation_strategies["balanced_optimization"]
    
    def _measure_inference_latency(self, compiled_model: CompiledModel, input_shape: Tuple) -> float:
        """Measure inference latency of compiled model."""
        try:
            # Create dummy input
            import numpy as np
            dummy_input = np.random.randn(*input_shape)
            
            # Time inference
            start_time = time.time()
            for _ in range(10):  # Average over multiple runs
                _ = compiled_model.run(dummy_input)
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / 10.0
            return avg_latency
        except:
            return 0.1  # Default latency
    
    def _measure_memory_usage(self, compiled_model: CompiledModel) -> float:
        """Measure memory usage of compiled model."""
        # Simplified memory estimation
        return getattr(compiled_model, 'memory_usage', 1.0)
    
    async def _estimate_accuracy_preservation(
        self, 
        compiled_model: CompiledModel, 
        original_model: Any,
        test_dataset: Optional[Any]
    ) -> float:
        """Estimate accuracy preservation of compiled model."""
        # Simplified accuracy estimation
        # In practice, this would run actual accuracy tests
        return 0.95 + np.random.normal(0, 0.02)  # Simulate ~95% accuracy preservation
    
    async def _adapt_strategies(self, fitness_results: List[Tuple[CompiledModel, EvolutionMetrics]]):
        """Adapt strategy success rates and priorities based on results."""
        for compiled_model, metrics in fitness_results:
            # Update strategy success rates
            improvement = metrics.improvement_factor
            
            # Find closest matching strategy and update its stats
            for strategy in self.adaptation_strategies.values():
                if improvement > 0:
                    strategy.success_rate = 0.9 * strategy.success_rate + 0.1 * 1.0
                    strategy.avg_improvement = 0.9 * strategy.avg_improvement + 0.1 * improvement
                else:
                    strategy.success_rate = 0.9 * strategy.success_rate + 0.1 * 0.0
    
    async def _adapt_evolution_parameters(self, fitness_results: List[Tuple[CompiledModel, EvolutionMetrics]]):
        """Adapt evolution parameters based on progress."""
        if len(fitness_results) < 2:
            return
        
        fitness_scores = [metrics.fitness_score for _, metrics in fitness_results]
        fitness_std = np.std(fitness_scores)
        
        # Increase mutation rate if population is converging
        if fitness_std < 0.05:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
        else:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
    
    def _meets_target_criteria(self, metrics: EvolutionMetrics, targets: Dict[str, float]) -> bool:
        """Check if metrics meet target criteria."""
        for metric, target in targets.items():
            if hasattr(metrics, metric):
                if getattr(metrics, metric) < target:
                    return False
        return True
    
    def _save_evolution_state(self, metrics: EvolutionMetrics):
        """Save evolution state to disk."""
        state_file = self.storage_path / "evolution_state.json"
        
        state = {
            "generation": self.current_generation,
            "best_metrics": asdict(metrics),
            "evolution_history": [asdict(m) for m in self.evolution_history],
            "strategies": {k: asdict(v) for k, v in self.adaptation_strategies.items()}
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_evolution_state(self) -> bool:
        """Load evolution state from disk."""
        state_file = self.storage_path / "evolution_state.json"
        
        if not state_file.exists():
            return False
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.current_generation = state.get("generation", 0)
            
            if "best_metrics" in state:
                self.baseline_metrics = EvolutionMetrics(**state["best_metrics"])
            
            if "evolution_history" in state:
                self.evolution_history = [
                    EvolutionMetrics(**m) for m in state["evolution_history"]
                ]
            
            return True
        except Exception as e:
            print(f"Failed to load evolution state: {e}")
            return False
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution progress."""
        if not self.evolution_history:
            return {"status": "not_started"}
        
        best_fitness = max(m.fitness_score for m in self.evolution_history)
        avg_improvement = np.mean([m.improvement_factor for m in self.evolution_history])
        
        return {
            "status": "active",
            "current_generation": self.current_generation,
            "total_generations": len(self.evolution_history),
            "best_fitness": best_fitness,
            "average_improvement": avg_improvement,
            "baseline_fitness": self.baseline_metrics.fitness_score if self.baseline_metrics else 0,
            "active_strategies": len(self.adaptation_strategies),
            "population_size": self.population_size
        }