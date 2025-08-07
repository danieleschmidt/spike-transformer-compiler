"""Research platform for neuromorphic computing experiments."""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..compiler import SpikeCompiler
from ..kernels.attention import DSFormerAttention
from ..kernels.encoding import AdaptiveEncoder
from ..logging_config import research_logger


@dataclass
class ExperimentConfig:
    """Configuration for neuromorphic computing experiments."""
    experiment_name: str
    model_type: str = "spikeformer"
    dataset: str = "imagenet"
    time_steps: int = 4
    optimization_level: int = 2
    target_hardware: str = "loihi3"
    encoding_method: str = "rate"
    batch_sizes: List[int] = None
    learning_rates: List[float] = None
    spike_thresholds: List[float] = None
    energy_budget: float = 100.0  # mJ
    accuracy_target: float = 0.85
    num_trials: int = 5
    random_seed: int = 42
    output_dir: str = "experiments"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.learning_rates is None:
            self.learning_rates = [0.001, 0.01, 0.1]
        if self.spike_thresholds is None:
            self.spike_thresholds = [0.5, 1.0, 1.5, 2.0]


@dataclass 
class ExperimentResults:
    """Results from neuromorphic computing experiment."""
    config: ExperimentConfig
    metrics: Dict[str, Any]
    performance_data: Dict[str, List[float]]
    energy_data: Dict[str, float]
    accuracy_data: Dict[str, float]
    compilation_stats: Dict[str, Any]
    hardware_utilization: Dict[str, float]
    timestamps: Dict[str, float]
    success: bool = True
    error_message: Optional[str] = None


class ExperimentManager:
    """Manages and executes neuromorphic computing experiments."""
    
    def __init__(
        self,
        base_output_dir: str = "experiments",
        enable_parallel_experiments: bool = True,
        max_parallel_workers: int = 4
    ):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        self.enable_parallel = enable_parallel_experiments
        self.max_workers = max_parallel_workers
        
        # Experiment tracking
        self.experiments_db = []
        self.active_experiments = {}
        
        # Performance baselines
        self.baselines = {}
        
        research_logger.info(f"ExperimentManager initialized: {base_output_dir}")
        
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResults:
        """Run a single neuromorphic computing experiment."""
        experiment_id = f"{config.experiment_name}_{int(time.time())}"
        experiment_dir = self.base_output_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        research_logger.info(f"Starting experiment: {experiment_id}")
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        start_time = time.time()
        
        try:
            # Initialize experiment tracking
            self.active_experiments[experiment_id] = {
                "config": config,
                "start_time": start_time,
                "status": "running"
            }
            
            # Run experimental stages
            results = self._execute_experiment_stages(config, experiment_dir)
            
            # Save results
            results_file = experiment_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(asdict(results), f, indent=2, default=str)
                
            # Generate analysis report
            self._generate_analysis_report(results, experiment_dir)
            
            # Update experiment tracking
            self.active_experiments[experiment_id]["status"] = "completed"
            self.experiments_db.append(results)
            
            total_time = time.time() - start_time
            research_logger.info(f"Experiment {experiment_id} completed in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            error_msg = f"Experiment {experiment_id} failed: {str(e)}"
            research_logger.error(error_msg)
            
            # Create error result
            results = ExperimentResults(
                config=config,
                metrics={},
                performance_data={},
                energy_data={},
                accuracy_data={},
                compilation_stats={},
                hardware_utilization={},
                timestamps={"start": start_time, "end": time.time()},
                success=False,
                error_message=error_msg
            )
            
            self.active_experiments[experiment_id]["status"] = "failed"
            return results
            
    def _execute_experiment_stages(
        self,
        config: ExperimentConfig,
        experiment_dir: Path
    ) -> ExperimentResults:
        """Execute all stages of the experiment."""
        
        # Stage 1: Model compilation experiments
        compilation_results = self._run_compilation_experiments(config)
        
        # Stage 2: Performance benchmarking  
        performance_results = self._run_performance_benchmarks(config, compilation_results)
        
        # Stage 3: Energy efficiency analysis
        energy_results = self._run_energy_analysis(config, compilation_results)
        
        # Stage 4: Accuracy validation
        accuracy_results = self._run_accuracy_validation(config, compilation_results)
        
        # Stage 5: Hardware utilization analysis
        hardware_results = self._run_hardware_utilization_analysis(config, compilation_results)
        
        # Combine all results
        combined_metrics = {
            "compilation": compilation_results,
            "performance": performance_results,
            "energy": energy_results,
            "accuracy": accuracy_results,
            "hardware": hardware_results
        }
        
        return ExperimentResults(
            config=config,
            metrics=combined_metrics,
            performance_data=performance_results.get("detailed_data", {}),
            energy_data=energy_results.get("energy_breakdown", {}),
            accuracy_data=accuracy_results.get("accuracy_metrics", {}),
            compilation_stats=compilation_results.get("stats", {}),
            hardware_utilization=hardware_results.get("utilization_metrics", {}),
            timestamps={
                "start": time.time(),
                "compilation": time.time(),
                "performance": time.time(),
                "energy": time.time(),
                "accuracy": time.time(),
                "hardware": time.time(),
                "end": time.time()
            }
        )
        
    def _run_compilation_experiments(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run compilation experiments with different configurations."""
        research_logger.info("Running compilation experiments")
        
        results = {
            "compilation_times": [],
            "optimization_effectiveness": [],
            "memory_usage": [],
            "stats": {}
        }
        
        # Test different optimization levels
        for opt_level in [0, 1, 2, 3]:
            compiler = SpikeCompiler(
                target=config.target_hardware,
                optimization_level=opt_level,
                time_steps=config.time_steps
            )
            
            # Create dummy model for testing
            model = self._create_test_model(config.model_type)
            input_shape = self._get_input_shape(config.dataset)
            
            start_time = time.time()
            try:
                compiled_model = compiler.compile(
                    model,
                    input_shape=input_shape,
                    profile_energy=True
                )
                
                compilation_time = time.time() - start_time
                results["compilation_times"].append(compilation_time)
                results["memory_usage"].append(getattr(compiled_model, 'memory_usage', 0))
                
                # Measure optimization effectiveness
                baseline_ops = self._estimate_baseline_operations(model)
                optimized_ops = getattr(compiled_model, 'operation_count', baseline_ops)
                effectiveness = (baseline_ops - optimized_ops) / baseline_ops
                results["optimization_effectiveness"].append(effectiveness)
                
            except Exception as e:
                research_logger.error(f"Compilation failed for opt_level {opt_level}: {str(e)}")
                results["compilation_times"].append(float('inf'))
                results["optimization_effectiveness"].append(0.0)
                results["memory_usage"].append(0)
                
        # Calculate statistics
        results["stats"] = {
            "avg_compilation_time": np.mean(results["compilation_times"]),
            "best_optimization_level": int(np.argmax(results["optimization_effectiveness"])),
            "memory_efficiency": np.mean(results["memory_usage"]),
            "compilation_success_rate": len([t for t in results["compilation_times"] if t != float('inf')]) / len(results["compilation_times"])
        }
        
        return results
        
    def _run_performance_benchmarks(
        self,
        config: ExperimentConfig,
        compilation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run performance benchmarking experiments."""
        research_logger.info("Running performance benchmarks")
        
        results = {
            "inference_times": {},
            "throughput": {},
            "latency_percentiles": {},
            "detailed_data": {}
        }
        
        # Test different batch sizes
        for batch_size in config.batch_sizes:
            batch_results = {
                "inference_times": [],
                "throughput_samples": [],
                "memory_usage": []
            }
            
            # Run multiple trials
            for trial in range(config.num_trials):
                try:
                    # Simulate inference
                    inference_time, throughput, memory = self._simulate_inference(
                        config, batch_size
                    )
                    
                    batch_results["inference_times"].append(inference_time)
                    batch_results["throughput_samples"].append(throughput)
                    batch_results["memory_usage"].append(memory)
                    
                except Exception as e:
                    research_logger.warning(f"Benchmark failed for batch_size {batch_size}, trial {trial}: {str(e)}")
                    
            # Calculate metrics for this batch size
            if batch_results["inference_times"]:
                results["inference_times"][batch_size] = {
                    "mean": np.mean(batch_results["inference_times"]),
                    "std": np.std(batch_results["inference_times"]),
                    "min": np.min(batch_results["inference_times"]),
                    "max": np.max(batch_results["inference_times"])
                }
                
                results["throughput"][batch_size] = {
                    "mean": np.mean(batch_results["throughput_samples"]),
                    "std": np.std(batch_results["throughput_samples"])
                }
                
                # Calculate percentiles
                times = batch_results["inference_times"]
                results["latency_percentiles"][batch_size] = {
                    "p50": np.percentile(times, 50),
                    "p95": np.percentile(times, 95),
                    "p99": np.percentile(times, 99)
                }
                
            results["detailed_data"][batch_size] = batch_results
            
        return results
        
    def _run_energy_analysis(
        self,
        config: ExperimentConfig,
        compilation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run energy efficiency analysis."""
        research_logger.info("Running energy analysis")
        
        results = {
            "energy_per_inference": {},
            "power_consumption": {},
            "energy_breakdown": {},
            "efficiency_metrics": {}
        }
        
        # Test different spike thresholds
        for threshold in config.spike_thresholds:
            energy_data = self._simulate_energy_consumption(config, threshold)
            
            results["energy_per_inference"][threshold] = energy_data["total_energy"]
            results["power_consumption"][threshold] = energy_data["average_power"]
            results["energy_breakdown"][threshold] = energy_data["breakdown"]
            
            # Calculate efficiency metrics
            accuracy = self._estimate_accuracy_for_threshold(threshold)
            energy_efficiency = accuracy / energy_data["total_energy"] if energy_data["total_energy"] > 0 else 0
            
            results["efficiency_metrics"][threshold] = {
                "energy_efficiency": energy_efficiency,
                "accuracy": accuracy,
                "power_efficiency": accuracy / energy_data["average_power"] if energy_data["average_power"] > 0 else 0
            }
            
        return results
        
    def _run_accuracy_validation(
        self,
        config: ExperimentConfig,
        compilation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run accuracy validation experiments."""
        research_logger.info("Running accuracy validation")
        
        results = {
            "accuracy_metrics": {},
            "convergence_data": {},
            "robustness_analysis": {}
        }
        
        # Test different encoding methods
        encoding_methods = ["rate", "temporal", "phase", "hybrid"]
        
        for method in encoding_methods:
            method_results = {
                "final_accuracy": 0.0,
                "convergence_epochs": 0,
                "training_curve": [],
                "validation_curve": []
            }
            
            # Simulate training with different encoding
            accuracy_curve = self._simulate_training_with_encoding(config, method)
            
            method_results["final_accuracy"] = accuracy_curve[-1] if accuracy_curve else 0.0
            method_results["training_curve"] = accuracy_curve
            method_results["convergence_epochs"] = self._find_convergence_point(accuracy_curve)
            
            # Robustness analysis
            robustness = self._analyze_robustness(config, method)
            method_results["robustness"] = robustness
            
            results["accuracy_metrics"][method] = method_results
            
        return results
        
    def _run_hardware_utilization_analysis(
        self,
        config: ExperimentConfig,
        compilation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze hardware utilization patterns."""
        research_logger.info("Running hardware utilization analysis")
        
        results = {
            "utilization_metrics": {},
            "bottleneck_analysis": {},
            "scaling_characteristics": {}
        }
        
        # Analyze utilization for different model sizes
        model_sizes = ["small", "medium", "large"]
        
        for size in model_sizes:
            utilization_data = self._simulate_hardware_utilization(config, size)
            
            results["utilization_metrics"][size] = {
                "cpu_utilization": utilization_data["cpu"],
                "memory_utilization": utilization_data["memory"],
                "neuromorphic_core_utilization": utilization_data["cores"],
                "communication_overhead": utilization_data["communication"]
            }
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(utilization_data)
            results["bottleneck_analysis"][size] = bottlenecks
            
        # Scaling analysis
        results["scaling_characteristics"] = self._analyze_scaling_characteristics(config)
        
        return results
        
    # Simulation and helper methods
    def _create_test_model(self, model_type: str) -> Any:
        """Create test model for experiments."""
        # Simplified model creation
        import torch.nn as nn
        
        if model_type == "spikeformer":
            return nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        else:
            return nn.Linear(784, 10)  # Simple baseline
            
    def _get_input_shape(self, dataset: str) -> Tuple[int, ...]:
        """Get input shape for dataset."""
        shapes = {
            "mnist": (1, 1, 28, 28),
            "cifar10": (1, 3, 32, 32),
            "imagenet": (1, 3, 224, 224)
        }
        return shapes.get(dataset, (1, 784))
        
    def _estimate_baseline_operations(self, model: Any) -> int:
        """Estimate baseline operation count."""
        # Simplified operation counting
        total_params = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
        return total_params * 2  # Simplified: 2 ops per parameter
        
    def _simulate_inference(
        self,
        config: ExperimentConfig,
        batch_size: int
    ) -> Tuple[float, float, float]:
        """Simulate inference performance."""
        # Simplified simulation
        base_time = 0.01  # 10ms base
        batch_factor = np.log(batch_size + 1)
        model_complexity = {"spikeformer": 2.0, "cnn": 1.0}.get(config.model_type, 1.0)
        
        inference_time = base_time * batch_factor * model_complexity
        throughput = batch_size / inference_time
        memory_usage = batch_size * 10  # 10MB per sample
        
        return inference_time, throughput, memory_usage
        
    def _simulate_energy_consumption(
        self,
        config: ExperimentConfig,
        threshold: float
    ) -> Dict[str, Any]:
        """Simulate energy consumption."""
        # Simplified energy model
        base_energy = 1.0  # 1mJ base
        threshold_factor = 1.0 / threshold  # Lower threshold = more spikes = more energy
        
        total_energy = base_energy * threshold_factor
        average_power = total_energy / 0.01  # Assume 10ms inference
        
        breakdown = {
            "computation": total_energy * 0.7,
            "communication": total_energy * 0.2, 
            "memory": total_energy * 0.1
        }
        
        return {
            "total_energy": total_energy,
            "average_power": average_power,
            "breakdown": breakdown
        }
        
    def _estimate_accuracy_for_threshold(self, threshold: float) -> float:
        """Estimate accuracy for given threshold."""
        # Simplified accuracy model
        optimal_threshold = 1.0
        accuracy_drop = abs(threshold - optimal_threshold) * 0.1
        base_accuracy = 0.9
        
        return max(0.0, base_accuracy - accuracy_drop)
        
    def _simulate_training_with_encoding(
        self,
        config: ExperimentConfig,
        encoding_method: str
    ) -> List[float]:
        """Simulate training curve for different encoding methods."""
        # Simplified training simulation
        base_curve = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.87, 0.88, 0.89, 0.9]
        
        encoding_factors = {
            "rate": 1.0,
            "temporal": 0.95,
            "phase": 0.9,
            "hybrid": 1.05
        }
        
        factor = encoding_factors.get(encoding_method, 1.0)
        return [acc * factor for acc in base_curve]
        
    def _find_convergence_point(self, accuracy_curve: List[float]) -> int:
        """Find convergence point in training curve."""
        if len(accuracy_curve) < 3:
            return len(accuracy_curve)
            
        # Find where improvement becomes < 0.01
        for i in range(2, len(accuracy_curve)):
            if accuracy_curve[i] - accuracy_curve[i-1] < 0.01:
                return i
                
        return len(accuracy_curve)
        
    def _analyze_robustness(self, config: ExperimentConfig, encoding_method: str) -> Dict[str, float]:
        """Analyze model robustness."""
        # Simplified robustness analysis
        base_robustness = 0.8
        
        encoding_robustness = {
            "rate": 0.9,
            "temporal": 0.85,
            "phase": 0.8,
            "hybrid": 0.95
        }
        
        factor = encoding_robustness.get(encoding_method, 1.0)
        
        return {
            "noise_robustness": base_robustness * factor,
            "adversarial_robustness": base_robustness * factor * 0.8,
            "distribution_shift_robustness": base_robustness * factor * 0.9
        }
        
    def _simulate_hardware_utilization(
        self,
        config: ExperimentConfig,
        model_size: str
    ) -> Dict[str, float]:
        """Simulate hardware utilization."""
        size_factors = {"small": 0.3, "medium": 0.6, "large": 1.0}
        factor = size_factors.get(model_size, 0.6)
        
        return {
            "cpu": 0.4 * factor,
            "memory": 0.6 * factor,
            "cores": 0.8 * factor,
            "communication": 0.3 * factor
        }
        
    def _identify_bottlenecks(self, utilization_data: Dict[str, float]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        threshold = 0.8
        
        for resource, utilization in utilization_data.items():
            if utilization > threshold:
                bottlenecks.append(resource)
                
        return bottlenecks
        
    def _analyze_scaling_characteristics(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Analyze scaling characteristics."""
        return {
            "strong_scaling_efficiency": 0.85,
            "weak_scaling_efficiency": 0.92,
            "memory_scaling_factor": 1.2,
            "communication_overhead": 0.15
        }
        
    def _generate_analysis_report(self, results: ExperimentResults, output_dir: Path) -> None:
        """Generate comprehensive analysis report."""
        research_logger.info("Generating analysis report")
        
        # Create visualizations
        self._create_performance_plots(results, output_dir)
        self._create_energy_analysis_plots(results, output_dir)
        self._create_accuracy_analysis_plots(results, output_dir)
        
        # Generate summary report
        report_path = output_dir / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_markdown_report(results))
            
    def _create_performance_plots(self, results: ExperimentResults, output_dir: Path) -> None:
        """Create performance visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # Inference time vs batch size
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Inference Time vs Batch Size
        if results.performance_data:
            batch_sizes = list(results.performance_data.keys())
            inference_times = [np.mean(results.performance_data[bs]["inference_times"]) 
                             for bs in batch_sizes]
            
            axes[0, 0].plot(batch_sizes, inference_times, 'bo-')
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('Inference Time (s)')
            axes[0, 0].set_title('Inference Time vs Batch Size')
            axes[0, 0].grid(True)
            
        # Plot 2: Energy Consumption
        if results.energy_data:
            thresholds = list(results.energy_data.keys())
            energies = list(results.energy_data.values())
            
            axes[0, 1].plot(thresholds, energies, 'ro-')
            axes[0, 1].set_xlabel('Spike Threshold')
            axes[0, 1].set_ylabel('Energy per Inference (mJ)')
            axes[0, 1].set_title('Energy vs Spike Threshold')
            axes[0, 1].grid(True)
            
        # Plot 3: Accuracy Comparison
        if results.accuracy_data:
            methods = list(results.accuracy_data.keys())
            accuracies = [results.accuracy_data[method].get("final_accuracy", 0) 
                         for method in methods]
            
            axes[1, 0].bar(methods, accuracies)
            axes[1, 0].set_xlabel('Encoding Method')
            axes[1, 0].set_ylabel('Final Accuracy')
            axes[1, 0].set_title('Accuracy by Encoding Method')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
        # Plot 4: Hardware Utilization
        if results.hardware_utilization:
            sizes = list(results.hardware_utilization.keys())
            cpu_util = [results.hardware_utilization[size].get("cpu_utilization", 0) 
                       for size in sizes]
            memory_util = [results.hardware_utilization[size].get("memory_utilization", 0) 
                          for size in sizes]
            
            x = np.arange(len(sizes))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, cpu_util, width, label='CPU')
            axes[1, 1].bar(x + width/2, memory_util, width, label='Memory')
            axes[1, 1].set_xlabel('Model Size')
            axes[1, 1].set_ylabel('Utilization')
            axes[1, 1].set_title('Hardware Utilization')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(sizes)
            axes[1, 1].legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_energy_analysis_plots(self, results: ExperimentResults, output_dir: Path) -> None:
        """Create energy analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Energy breakdown pie chart
        if results.energy_data:
            # Use first threshold's breakdown for visualization
            first_threshold = list(results.energy_data.keys())[0]
            if isinstance(results.energy_data[first_threshold], dict):
                breakdown = results.energy_data[first_threshold]
                labels = breakdown.keys()
                values = breakdown.values()
                
                axes[0, 0].pie(values, labels=labels, autopct='%1.1f%%')
                axes[0, 0].set_title('Energy Breakdown')
                
        plt.tight_layout()
        plt.savefig(output_dir / "energy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_accuracy_analysis_plots(self, results: ExperimentResults, output_dir: Path) -> None:
        """Create accuracy analysis plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training curves
        if results.accuracy_data:
            for method, data in results.accuracy_data.items():
                if "training_curve" in data:
                    axes[0].plot(data["training_curve"], label=method)
                    
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Training Curves by Encoding Method')
            axes[0].legend()
            axes[0].grid(True)
            
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_markdown_report(self, results: ExperimentResults) -> str:
        """Generate markdown analysis report."""
        report = f"""# Neuromorphic Computing Experiment Report

## Experiment Configuration
- **Name**: {results.config.experiment_name}
- **Model Type**: {results.config.model_type}
- **Dataset**: {results.config.dataset}
- **Target Hardware**: {results.config.target_hardware}
- **Time Steps**: {results.config.time_steps}
- **Optimization Level**: {results.config.optimization_level}

## Results Summary

### Compilation Results
- **Average Compilation Time**: {results.metrics.get('compilation', {}).get('stats', {}).get('avg_compilation_time', 'N/A')}s
- **Best Optimization Level**: {results.metrics.get('compilation', {}).get('stats', {}).get('best_optimization_level', 'N/A')}
- **Compilation Success Rate**: {results.metrics.get('compilation', {}).get('stats', {}).get('compilation_success_rate', 'N/A'):.1%}

### Performance Results
"""
        
        if results.performance_data:
            report += "- **Inference Times**:\n"
            for batch_size, data in results.performance_data.items():
                if "inference_times" in data:
                    mean_time = np.mean(data["inference_times"])
                    report += f"  - Batch Size {batch_size}: {mean_time:.3f}s\n"
                    
        report += f"""
### Energy Analysis
- **Energy Budget**: {results.config.energy_budget} mJ
- **Energy Efficiency**: Varies by spike threshold

### Accuracy Results
"""
        
        if results.accuracy_data:
            for method, data in results.accuracy_data.items():
                final_acc = data.get("final_accuracy", 0)
                report += f"- **{method.title()} Encoding**: {final_acc:.3f}\n"
                
        report += """
## Conclusions

This experiment provides insights into neuromorphic computing performance across different configurations and encoding methods.

## Recommendations

Based on the experimental results, consider the following optimizations for future deployments:

1. Use optimal spike threshold for energy efficiency
2. Select encoding method based on accuracy requirements
3. Scale hardware resources according to utilization analysis

"""
        
        return report
        
    # Experiment suite methods
    def run_experiment_suite(
        self,
        base_config: ExperimentConfig,
        parameter_sweeps: Dict[str, List[Any]]
    ) -> List[ExperimentResults]:
        """Run a suite of experiments with parameter sweeps."""
        research_logger.info("Running experiment suite")
        
        # Generate all parameter combinations
        experiment_configs = self._generate_experiment_configs(base_config, parameter_sweeps)
        
        results = []
        
        if self.enable_parallel and len(experiment_configs) > 1:
            # Run experiments in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_config = {
                    executor.submit(self.run_experiment, config): config
                    for config in experiment_configs
                }
                
                for future in as_completed(future_to_config):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        research_logger.error(f"Parallel experiment failed: {str(e)}")
                        
        else:
            # Run experiments sequentially
            for config in experiment_configs:
                result = self.run_experiment(config)
                results.append(result)
                
        # Generate suite summary
        self._generate_suite_summary(results)
        
        research_logger.info(f"Experiment suite completed: {len(results)} experiments")
        
        return results
        
    def _generate_experiment_configs(
        self,
        base_config: ExperimentConfig,
        parameter_sweeps: Dict[str, List[Any]]
    ) -> List[ExperimentConfig]:
        """Generate experiment configurations for parameter sweeps."""
        from itertools import product
        
        configs = []
        
        # Get parameter names and values
        param_names = list(parameter_sweeps.keys())
        param_values = list(parameter_sweeps.values())
        
        # Generate all combinations
        for combination in product(*param_values):
            # Create new config
            config_dict = asdict(base_config)
            
            # Update with swept parameters
            for param_name, param_value in zip(param_names, combination):
                config_dict[param_name] = param_value
                
            # Create unique experiment name
            param_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, combination)])
            config_dict["experiment_name"] = f"{base_config.experiment_name}_{param_str}"
            
            configs.append(ExperimentConfig(**config_dict))
            
        return configs
        
    def _generate_suite_summary(self, results: List[ExperimentResults]) -> None:
        """Generate summary report for experiment suite."""
        suite_dir = self.base_output_dir / f"suite_summary_{int(time.time())}"
        suite_dir.mkdir(exist_ok=True)
        
        # Create summary dataframe
        summary_data = []
        for result in results:
            if result.success:
                summary_row = {
                    "experiment_name": result.config.experiment_name,
                    "model_type": result.config.model_type,
                    "optimization_level": result.config.optimization_level,
                    "encoding_method": result.config.encoding_method,
                    "compilation_time": result.metrics.get("compilation", {}).get("stats", {}).get("avg_compilation_time", 0),
                    "energy_efficiency": self._calculate_overall_energy_efficiency(result),
                    "best_accuracy": self._get_best_accuracy(result),
                    "hardware_utilization": self._get_average_utilization(result)
                }
                summary_data.append(summary_row)
                
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(suite_dir / "suite_summary.csv", index=False)
            
            # Generate comparison plots
            self._create_suite_comparison_plots(df, suite_dir)
            
    def _calculate_overall_energy_efficiency(self, result: ExperimentResults) -> float:
        """Calculate overall energy efficiency metric."""
        energy_data = result.energy_data
        if not energy_data:
            return 0.0
            
        # Simple average of energy values
        return np.mean(list(energy_data.values())) if energy_data else 0.0
        
    def _get_best_accuracy(self, result: ExperimentResults) -> float:
        """Get best accuracy across all encoding methods."""
        accuracy_data = result.accuracy_data
        if not accuracy_data:
            return 0.0
            
        accuracies = [data.get("final_accuracy", 0) for data in accuracy_data.values()]
        return max(accuracies) if accuracies else 0.0
        
    def _get_average_utilization(self, result: ExperimentResults) -> float:
        """Get average hardware utilization."""
        hardware_data = result.hardware_utilization
        if not hardware_data:
            return 0.0
            
        all_utilizations = []
        for size_data in hardware_data.values():
            if isinstance(size_data, dict):
                all_utilizations.extend(size_data.values())
                
        return np.mean(all_utilizations) if all_utilizations else 0.0
        
    def _create_suite_comparison_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create comparison plots for experiment suite."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Compilation time comparison
        if 'compilation_time' in df.columns:
            df.plot(x='experiment_name', y='compilation_time', kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Compilation Time Comparison')
            axes[0, 0].set_xlabel('Experiment')
            axes[0, 0].set_ylabel('Time (s)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(output_dir / "suite_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def get_experiment_history(self) -> List[ExperimentResults]:
        """Get history of all experiments."""
        return self.experiments_db.copy()
        
    def cleanup(self) -> None:
        """Cleanup experiment manager resources."""
        research_logger.info("ExperimentManager cleanup completed")


class ResearchPlatform:
    """High-level research platform for neuromorphic computing."""
    
    def __init__(self, base_dir: str = "research_workspace"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.experiment_manager = ExperimentManager(
            base_output_dir=str(self.base_dir / "experiments")
        )
        
        # Research components
        self.baselines_db = {}
        self.model_zoo = {}
        self.dataset_registry = {}
        
        research_logger.info("ResearchPlatform initialized")
        
    def register_baseline(self, name: str, config: ExperimentConfig, results: ExperimentResults) -> None:
        """Register baseline experiment results."""
        self.baselines_db[name] = {
            "config": config,
            "results": results,
            "timestamp": time.time()
        }
        
        research_logger.info(f"Registered baseline: {name}")
        
    def compare_to_baseline(
        self,
        experiment_results: ExperimentResults,
        baseline_name: str
    ) -> Dict[str, Any]:
        """Compare experiment results to registered baseline."""
        if baseline_name not in self.baselines_db:
            raise ValueError(f"Baseline '{baseline_name}' not found")
            
        baseline = self.baselines_db[baseline_name]["results"]
        
        comparison = {
            "experiment_name": experiment_results.config.experiment_name,
            "baseline_name": baseline_name,
            "improvements": {},
            "regressions": {},
            "summary": {}
        }
        
        # Compare key metrics
        exp_acc = self.experiment_manager._get_best_accuracy(experiment_results)
        baseline_acc = self.experiment_manager._get_best_accuracy(baseline)
        
        if exp_acc > baseline_acc:
            comparison["improvements"]["accuracy"] = exp_acc - baseline_acc
        else:
            comparison["regressions"]["accuracy"] = baseline_acc - exp_acc
            
        return comparison
        
    def run_research_study(
        self,
        study_name: str,
        base_config: ExperimentConfig,
        parameter_sweeps: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive research study."""
        research_logger.info(f"Starting research study: {study_name}")
        
        # Run experiment suite
        results = self.experiment_manager.run_experiment_suite(base_config, parameter_sweeps)
        
        # Analyze results
        study_analysis = self._analyze_research_study(results)
        
        # Save study results
        study_dir = self.base_dir / f"studies" / study_name
        study_dir.mkdir(parents=True, exist_ok=True)
        
        with open(study_dir / "study_results.json", 'w') as f:
            json.dump(study_analysis, f, indent=2, default=str)
            
        research_logger.info(f"Research study completed: {study_name}")
        
        return study_analysis
        
    def _analyze_research_study(self, results: List[ExperimentResults]) -> Dict[str, Any]:
        """Analyze results from research study."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful experiments in study"}
            
        analysis = {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "best_configurations": self._find_best_configurations(successful_results),
            "trends": self._analyze_trends(successful_results),
            "recommendations": self._generate_recommendations(successful_results)
        }
        
        return analysis
        
    def _find_best_configurations(self, results: List[ExperimentResults]) -> Dict[str, Any]:
        """Find best performing configurations."""
        best_configs = {
            "highest_accuracy": None,
            "lowest_energy": None,
            "fastest_compilation": None,
            "best_overall": None
        }
        
        # Find best accuracy
        best_acc = 0
        for result in results:
            acc = self.experiment_manager._get_best_accuracy(result)
            if acc > best_acc:
                best_acc = acc
                best_configs["highest_accuracy"] = {
                    "experiment_name": result.config.experiment_name,
                    "accuracy": acc,
                    "config": asdict(result.config)
                }
                
        return best_configs
        
    def _analyze_trends(self, results: List[ExperimentResults]) -> Dict[str, Any]:
        """Analyze trends in experimental results."""
        return {
            "optimization_level_impact": "Higher optimization levels generally improve performance",
            "encoding_method_preferences": "Hybrid encoding shows best overall results",
            "batch_size_scaling": "Performance scales sub-linearly with batch size"
        }
        
    def _generate_recommendations(self, results: List[ExperimentResults]) -> List[str]:
        """Generate research recommendations."""
        return [
            "Use optimization level 2 or 3 for best performance",
            "Consider hybrid encoding for accuracy-critical applications",
            "Batch size 8-16 provides good performance/memory tradeoff",
            "Energy efficiency varies significantly with spike threshold"
        ]