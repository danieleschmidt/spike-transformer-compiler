#!/usr/bin/env python3
"""
Generation 3 Adaptive Quality System - MAKE IT SCALE
Auto-scaling quality gates with predictive optimization and ML-driven insights.
"""

import time
import asyncio
import threading
import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import statistics
import concurrent.futures

from .exceptions import ValidationError
from .monitoring import MetricsCollector
from .quality_gates import ProgressiveQualityGateSystem, QualityGateResult, Generation
from .quality_monitoring import AdaptiveQualityMonitor, QualityAlert
from .resilience import CircuitBreaker
from .scaling import AdaptiveScaler


@dataclass
class QualityPrediction:
    """Quality prediction result."""
    metric_name: str
    predicted_value: float
    confidence: float
    time_horizon: int  # minutes ahead
    prediction_timestamp: datetime
    model_version: str


@dataclass
class OptimizationRecommendation:
    """System optimization recommendation."""
    recommendation_id: str
    category: str  # performance, security, reliability, scalability
    priority: str  # low, medium, high, critical
    description: str
    estimated_impact: float  # 0-100 scale
    implementation_effort: str  # trivial, low, medium, high
    affected_components: List[str]
    timestamp: datetime


class MLQualityPredictor:
    """Machine learning-based quality predictor."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger("ml_quality_predictor")
        self.model_path = model_path or "quality_prediction_model.pkl"
        self.model = None
        self.feature_history = deque(maxlen=1000)
        self.prediction_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Load or initialize model
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize a new one."""
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info("Loaded existing quality prediction model")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self._initialize_simple_model()
        else:
            self._initialize_simple_model()
    
    def _initialize_simple_model(self):
        """Initialize a simple prediction model."""
        # For demonstration, use a simple statistical model
        self.model = {
            "type": "statistical",
            "parameters": {},
            "training_data": [],
            "version": "1.0"
        }
        self.logger.info("Initialized simple statistical prediction model")
    
    def add_training_data(self, features: Dict[str, float], target: Dict[str, float]):
        """Add training data to the model."""
        training_point = {
            "features": features,
            "target": target,
            "timestamp": datetime.now()
        }
        
        self.feature_history.append(training_point)
        
        if self.model["type"] == "statistical":
            self.model["training_data"].append(training_point)
            
            # Retrain every 100 data points
            if len(self.model["training_data"]) % 100 == 0:
                self._retrain_statistical_model()
    
    def _retrain_statistical_model(self):
        """Retrain the statistical model."""
        training_data = self.model["training_data"]
        
        if len(training_data) < 10:
            return
        
        # Calculate correlation patterns
        feature_correlations = defaultdict(dict)
        
        for metric_name in ["compilation_time", "memory_usage", "test_coverage"]:
            feature_values = []
            target_values = []
            
            for point in training_data[-100:]:  # Last 100 points
                if metric_name in point["target"]:
                    features = point["features"]
                    target = point["target"][metric_name]
                    
                    feature_values.append([
                        features.get("code_complexity", 0),
                        features.get("file_count", 0),
                        features.get("test_count", 0),
                        features.get("time_of_day", 12)  # Hour of day
                    ])
                    target_values.append(target)
            
            if len(feature_values) >= 5:
                # Calculate simple linear relationships
                feature_names = ["code_complexity", "file_count", "test_count", "time_of_day"]
                correlations = {}
                
                for i, feature_name in enumerate(feature_names):
                    feature_vals = [fv[i] for fv in feature_values]
                    if len(set(feature_vals)) > 1:  # Has variation
                        correlation = np.corrcoef(feature_vals, target_values)[0, 1]
                        correlations[feature_name] = correlation if not np.isnan(correlation) else 0
                
                feature_correlations[metric_name] = correlations
        
        self.model["parameters"] = feature_correlations
        self.logger.info("Retrained statistical model with correlation patterns")
    
    def predict_quality_metrics(self, features: Dict[str, float], time_horizon: int = 30) -> List[QualityPrediction]:
        """Predict quality metrics based on current features."""
        cache_key = f"{hash(str(features))}_{time_horizon}"
        
        # Check cache
        if cache_key in self.prediction_cache:
            cached_result, cache_time = self.prediction_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_result
        
        predictions = []
        
        if self.model["type"] == "statistical":
            correlations = self.model["parameters"]
            
            for metric_name, feature_corrs in correlations.items():
                predicted_value = self._predict_statistical(features, feature_corrs, metric_name)
                
                # Calculate confidence based on training data size and correlation strength
                training_size = len(self.model["training_data"])
                avg_correlation = statistics.mean(abs(c) for c in feature_corrs.values()) if feature_corrs else 0
                
                confidence = min(0.95, (training_size / 100) * avg_correlation)
                
                prediction = QualityPrediction(
                    metric_name=metric_name,
                    predicted_value=predicted_value,
                    confidence=confidence,
                    time_horizon=time_horizon,
                    prediction_timestamp=datetime.now(),
                    model_version=self.model["version"]
                )
                
                predictions.append(prediction)
        
        # Cache result
        self.prediction_cache[cache_key] = (predictions, datetime.now())
        
        return predictions
    
    def _predict_statistical(self, features: Dict[str, float], correlations: Dict[str, float], metric_name: str) -> float:
        """Make statistical prediction for a metric."""
        # Base prediction from historical mean
        training_data = self.model["training_data"]
        historical_values = [
            point["target"].get(metric_name, 0) 
            for point in training_data 
            if metric_name in point["target"]
        ]
        
        if historical_values:
            base_prediction = statistics.mean(historical_values)
        else:
            # Default baseline values
            baseline_values = {
                "compilation_time": 1.0,
                "memory_usage": 100.0,
                "test_coverage": 85.0,
                "code_quality_score": 80.0,
                "error_rate": 0.02
            }
            base_prediction = baseline_values.get(metric_name, 50.0)
        
        # Adjust based on feature correlations
        adjustment = 0
        for feature_name, correlation in correlations.items():
            if feature_name in features:
                feature_value = features[feature_name]
                # Simple linear adjustment
                adjustment += correlation * feature_value * 0.1
        
        predicted_value = base_prediction + adjustment
        
        # Apply reasonable bounds
        if metric_name == "test_coverage":
            predicted_value = max(0, min(100, predicted_value))
        elif metric_name == "error_rate":
            predicted_value = max(0, min(1, predicted_value))
        elif metric_name in ["compilation_time", "memory_usage"]:
            predicted_value = max(0, predicted_value)
        
        return predicted_value
    
    def save_model(self):
        """Save the current model to disk."""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")


class QualityOptimizationEngine:
    """Generates optimization recommendations based on quality analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("optimization_engine")
        self.recommendation_history = deque(maxlen=500)
        self.implementation_tracking = {}
        
    def analyze_and_recommend(self, 
                            quality_dashboard: Dict[str, Any],
                            predictions: List[QualityPrediction],
                            recent_alerts: List[QualityAlert]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance optimization recommendations
        recommendations.extend(self._analyze_performance(quality_dashboard, predictions))
        
        # Security recommendations
        recommendations.extend(self._analyze_security(recent_alerts))
        
        # Reliability recommendations
        recommendations.extend(self._analyze_reliability(quality_dashboard, recent_alerts))
        
        # Scalability recommendations
        recommendations.extend(self._analyze_scalability(quality_dashboard, predictions))
        
        # Code quality recommendations
        recommendations.extend(self._analyze_code_quality(quality_dashboard))
        
        # Sort by priority and impact
        recommendations.sort(key=lambda r: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[r.priority],
            r.estimated_impact
        ), reverse=True)
        
        # Store recommendations
        for rec in recommendations:
            self.recommendation_history.append(rec)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _analyze_performance(self, dashboard: Dict[str, Any], predictions: List[QualityPrediction]) -> List[OptimizationRecommendation]:
        """Analyze performance and generate recommendations."""
        recommendations = []
        metrics = dashboard.get("metrics", {})
        
        # Check compilation time
        compilation_metric = metrics.get("compilation_time", {})
        if compilation_metric.get("latest_value", 0) > 2.0:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"perf_compilation_{int(time.time())}",
                category="performance",
                priority="high",
                description="Optimize compilation pipeline for faster build times",
                estimated_impact=75.0,
                implementation_effort="medium",
                affected_components=["compiler", "optimization", "caching"],
                timestamp=datetime.now()
            ))
        
        # Check memory usage trends
        memory_prediction = next((p for p in predictions if p.metric_name == "memory_usage"), None)
        if memory_prediction and memory_prediction.predicted_value > 500:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"perf_memory_{int(time.time())}",
                category="performance",
                priority="medium",
                description="Implement memory optimization and garbage collection improvements",
                estimated_impact=60.0,
                implementation_effort="medium",
                affected_components=["memory_manager", "caching", "runtime"],
                timestamp=datetime.now()
            ))
        
        return recommendations
    
    def _analyze_security(self, recent_alerts: List[QualityAlert]) -> List[OptimizationRecommendation]:
        """Analyze security and generate recommendations."""
        recommendations = []
        
        security_alerts = [a for a in recent_alerts if "security" in a.message.lower()]
        
        if len(security_alerts) > 2:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"sec_alerts_{int(time.time())}",
                category="security",
                priority="critical",
                description="Address multiple security alerts and strengthen security validation",
                estimated_impact=90.0,
                implementation_effort="high",
                affected_components=["security", "validation", "authentication"],
                timestamp=datetime.now()
            ))
        
        return recommendations
    
    def _analyze_reliability(self, dashboard: Dict[str, Any], recent_alerts: List[QualityAlert]) -> List[OptimizationRecommendation]:
        """Analyze reliability and generate recommendations."""
        recommendations = []
        
        error_alerts = [a for a in recent_alerts if a.severity in ["ERROR", "CRITICAL"]]
        
        if len(error_alerts) > 3:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"rel_errors_{int(time.time())}",
                category="reliability",
                priority="high",
                description="Implement enhanced error handling and circuit breaker patterns",
                estimated_impact=80.0,
                implementation_effort="medium",
                affected_components=["resilience", "error_handling", "monitoring"],
                timestamp=datetime.now()
            ))
        
        # Check overall health
        if dashboard.get("overall_health_score", 100) < 70:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"rel_health_{int(time.time())}",
                category="reliability",
                priority="high",
                description="Improve overall system health through comprehensive diagnostics",
                estimated_impact=85.0,
                implementation_effort="high",
                affected_components=["monitoring", "diagnostics", "health_checks"],
                timestamp=datetime.now()
            ))
        
        return recommendations
    
    def _analyze_scalability(self, dashboard: Dict[str, Any], predictions: List[QualityPrediction]) -> List[OptimizationRecommendation]:
        """Analyze scalability and generate recommendations."""
        recommendations = []
        
        # Check if any metrics show degrading trends
        metrics = dashboard.get("metrics", {})
        degrading_metrics = [
            name for name, info in metrics.items() 
            if info.get("trend") == "degrading"
        ]
        
        if len(degrading_metrics) >= 2:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"scale_trends_{int(time.time())}",
                category="scalability",
                priority="medium",
                description="Implement auto-scaling to handle increasing load patterns",
                estimated_impact=70.0,
                implementation_effort="high",
                affected_components=["scaling", "load_balancing", "resource_management"],
                timestamp=datetime.now()
            ))
        
        return recommendations
    
    def _analyze_code_quality(self, dashboard: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Analyze code quality and generate recommendations."""
        recommendations = []
        
        metrics = dashboard.get("metrics", {})
        code_quality = metrics.get("code_quality_score", {})
        test_coverage = metrics.get("test_coverage", {})
        
        if code_quality.get("latest_value", 100) < 80:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"quality_code_{int(time.time())}",
                category="code_quality",
                priority="medium",
                description="Improve code quality through refactoring and design pattern implementation",
                estimated_impact=65.0,
                implementation_effort="medium",
                affected_components=["code_structure", "design_patterns", "documentation"],
                timestamp=datetime.now()
            ))
        
        if test_coverage.get("latest_value", 100) < 85:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"quality_tests_{int(time.time())}",
                category="code_quality",
                priority="high",
                description="Increase test coverage and implement comprehensive testing strategies",
                estimated_impact=75.0,
                implementation_effort="medium",
                affected_components=["testing", "test_automation", "quality_assurance"],
                timestamp=datetime.now()
            ))
        
        return recommendations


class AdaptiveQualitySystem:
    """
    Generation 3 Adaptive Quality System that scales with demand,
    predicts quality issues, and optimizes system performance autonomously.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("adaptive_quality_system")
        
        # Core components
        self.quality_gates = ProgressiveQualityGateSystem()
        self.quality_monitor = AdaptiveQualityMonitor(config_path)
        self.ml_predictor = MLQualityPredictor()
        self.optimization_engine = QualityOptimizationEngine()
        self.adaptive_scaler = AdaptiveScaler()
        
        # Adaptive system state
        self.system_load = 0.0
        self.prediction_interval = 300  # 5 minutes
        self.optimization_interval = 900  # 15 minutes
        self.auto_optimization_enabled = True
        
        # Execution tracking
        self.last_prediction_time = datetime.now()
        self.last_optimization_time = datetime.now()
        self.execution_history = deque(maxlen=1000)
        
        # Async components
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Performance metrics
        self.performance_tracker = {
            "predictions_made": 0,
            "optimizations_applied": 0,
            "quality_improvements": 0,
            "system_uptime": datetime.now()
        }
    
    async def start_adaptive_system(self):
        """Start the adaptive quality system with full automation."""
        if self.running:
            self.logger.warning("Adaptive system already running")
            return
        
        self.running = True
        self.logger.info("Starting adaptive quality system...")
        
        # Start quality monitoring
        self.quality_monitor.start_monitoring()
        
        # Start adaptive tasks
        tasks = [
            self._adaptive_prediction_loop(),
            self._adaptive_optimization_loop(),
            self._performance_monitoring_loop(),
            self._load_balancing_loop()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in adaptive system: {e}")
        finally:
            await self.stop_adaptive_system()
    
    async def stop_adaptive_system(self):
        """Stop the adaptive quality system."""
        self.running = False
        self.quality_monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        self.logger.info("Adaptive quality system stopped")
    
    async def _adaptive_prediction_loop(self):
        """Continuous prediction and forecasting loop."""
        while self.running:
            try:
                await self._run_predictive_analysis()
                await asyncio.sleep(self.prediction_interval)
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    async def _adaptive_optimization_loop(self):
        """Continuous optimization and recommendation loop."""
        while self.running:
            try:
                await self._run_optimization_cycle()
                await asyncio.sleep(self.optimization_interval)
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)  # Brief pause on error
    
    async def _performance_monitoring_loop(self):
        """Monitor and adapt system performance."""
        while self.running:
            try:
                await self._monitor_system_performance()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _load_balancing_loop(self):
        """Dynamic load balancing and resource allocation."""
        while self.running:
            try:
                await self._balance_system_load()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in load balancing: {e}")
                await asyncio.sleep(10)
    
    async def _run_predictive_analysis(self):
        """Run predictive analysis and forecasting."""
        self.logger.info("Running predictive analysis...")
        
        # Get current system features
        dashboard = self.quality_monitor.get_quality_dashboard()
        current_time = datetime.now()
        
        features = {
            "code_complexity": self._estimate_code_complexity(),
            "file_count": self._count_source_files(),
            "test_count": self._count_test_files(),
            "time_of_day": current_time.hour,
            "day_of_week": current_time.weekday(),
            "system_load": self.system_load
        }
        
        # Generate predictions
        predictions = self.ml_predictor.predict_quality_metrics(features, time_horizon=30)
        
        # Add training data
        if dashboard["metrics"]:
            target_metrics = {
                name: info.get("latest_value", 0)
                for name, info in dashboard["metrics"].items()
                if info.get("latest_value") is not None
            }
            self.ml_predictor.add_training_data(features, target_metrics)
        
        # Check for predicted issues
        await self._handle_predicted_issues(predictions)
        
        self.performance_tracker["predictions_made"] += len(predictions)
        self.last_prediction_time = current_time
        
        self.logger.info(f"Completed predictive analysis: {len(predictions)} predictions generated")
    
    async def _run_optimization_cycle(self):
        """Run optimization cycle and apply recommendations."""
        self.logger.info("Running optimization cycle...")
        
        # Get current system state
        dashboard = self.quality_monitor.get_quality_dashboard()
        recent_alerts = list(self.quality_monitor.alerts)[-20:]  # Last 20 alerts
        
        # Get recent predictions
        features = {
            "code_complexity": self._estimate_code_complexity(),
            "file_count": self._count_source_files(),
            "test_count": self._count_test_files(),
            "time_of_day": datetime.now().hour,
            "system_load": self.system_load
        }
        predictions = self.ml_predictor.predict_quality_metrics(features)
        
        # Generate recommendations
        recommendations = self.optimization_engine.analyze_and_recommend(
            dashboard, predictions, recent_alerts
        )
        
        # Apply high-priority recommendations if auto-optimization is enabled
        if self.auto_optimization_enabled:
            applied_count = await self._apply_recommendations(recommendations)
            self.performance_tracker["optimizations_applied"] += applied_count
        
        self.last_optimization_time = datetime.now()
        
        self.logger.info(f"Completed optimization cycle: {len(recommendations)} recommendations generated")
    
    async def _monitor_system_performance(self):
        """Monitor and adapt system performance parameters."""
        dashboard = self.quality_monitor.get_quality_dashboard()
        
        # Calculate system load based on metrics
        health_score = dashboard.get("overall_health_score", 100)
        alert_count = dashboard.get("total_alerts_today", 0)
        
        # Simple load calculation
        load_factors = [
            (100 - health_score) / 100,  # Health-based load
            min(1.0, alert_count / 10),   # Alert-based load
        ]
        
        self.system_load = statistics.mean(load_factors)
        
        # Adapt intervals based on load
        if self.system_load > 0.7:  # High load
            self.prediction_interval = max(60, self.prediction_interval - 30)
            self.optimization_interval = max(300, self.optimization_interval - 60)
        elif self.system_load < 0.3:  # Low load
            self.prediction_interval = min(600, self.prediction_interval + 30)
            self.optimization_interval = min(1800, self.optimization_interval + 60)
        
        # Scale resources
        await self.adaptive_scaler.scale_based_on_load(self.system_load)
    
    async def _balance_system_load(self):
        """Balance system load across components."""
        # This would implement actual load balancing
        # For now, simulate load balancing decisions
        
        if self.system_load > 0.8:
            self.logger.warning(f"High system load detected: {self.system_load:.2f}")
            # Could trigger additional workers, reduce precision, etc.
        
        # Update adaptive scaler
        await self.adaptive_scaler.balance_resources()
    
    async def _handle_predicted_issues(self, predictions: List[QualityPrediction]):
        """Handle predicted quality issues proactively."""
        for prediction in predictions:
            if prediction.confidence > 0.7:  # High confidence predictions
                
                # Check for concerning predictions
                concerning_thresholds = {
                    "compilation_time": 5.0,
                    "memory_usage": 1000.0,
                    "error_rate": 0.1,
                    "test_coverage": 70.0
                }
                
                threshold = concerning_thresholds.get(prediction.metric_name)
                if threshold:
                    if ((prediction.metric_name in ["compilation_time", "memory_usage", "error_rate"] and 
                         prediction.predicted_value > threshold) or
                        (prediction.metric_name == "test_coverage" and 
                         prediction.predicted_value < threshold)):
                        
                        await self._trigger_preventive_action(prediction)
    
    async def _trigger_preventive_action(self, prediction: QualityPrediction):
        """Trigger preventive action based on prediction."""
        self.logger.warning(
            f"Predicted quality issue: {prediction.metric_name} = {prediction.predicted_value:.2f} "
            f"(confidence: {prediction.confidence:.2f})"
        )
        
        # Could trigger preventive measures:
        # - Pre-emptive scaling
        # - Cache warming
        # - Resource cleanup
        # - Quality gate adjustments
        
        preventive_actions = {
            "compilation_time": "Enable compilation caching and parallel processing",
            "memory_usage": "Trigger garbage collection and memory optimization",
            "error_rate": "Increase error monitoring and validation",
            "test_coverage": "Schedule additional test execution"
        }
        
        action = preventive_actions.get(prediction.metric_name, "Monitor closely")
        self.logger.info(f"Preventive action for {prediction.metric_name}: {action}")
    
    async def _apply_recommendations(self, recommendations: List[OptimizationRecommendation]) -> int:
        """Apply optimization recommendations automatically."""
        applied_count = 0
        
        for rec in recommendations:
            if rec.priority in ["critical", "high"] and rec.implementation_effort in ["trivial", "low"]:
                # Only auto-apply low-effort, high-priority recommendations
                success = await self._implement_recommendation(rec)
                if success:
                    applied_count += 1
                    self.logger.info(f"Auto-applied recommendation: {rec.description}")
        
        return applied_count
    
    async def _implement_recommendation(self, recommendation: OptimizationRecommendation) -> bool:
        """Implement a specific recommendation."""
        # This would contain actual implementation logic
        # For demonstration, simulate implementation
        
        try:
            if "caching" in recommendation.affected_components:
                # Simulate cache optimization
                await asyncio.sleep(0.1)
                return True
            elif "monitoring" in recommendation.affected_components:
                # Simulate monitoring improvement
                await asyncio.sleep(0.1)
                return True
            elif "performance" in recommendation.category:
                # Simulate performance optimization
                await asyncio.sleep(0.1)
                return True
        except Exception as e:
            self.logger.error(f"Failed to implement recommendation {recommendation.recommendation_id}: {e}")
            return False
        
        return False
    
    def _estimate_code_complexity(self) -> float:
        """Estimate code complexity based on file analysis."""
        src_path = Path("src")
        if not src_path.exists():
            return 10.0  # Default complexity
        
        total_lines = 0
        total_functions = 0
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    total_lines += len(content.splitlines())
                    total_functions += content.count("def ")
            except Exception:
                continue
        
        # Simple complexity metric
        complexity = (total_lines / 100) + (total_functions / 10)
        return min(100.0, complexity)
    
    def _count_source_files(self) -> int:
        """Count source files."""
        src_path = Path("src")
        if not src_path.exists():
            return 0
        
        return len(list(src_path.rglob("*.py")))
    
    def _count_test_files(self) -> int:
        """Count test files."""
        test_patterns = ["test_*.py", "*_test.py", "tests/*.py"]
        test_count = 0
        
        for pattern in test_patterns:
            test_count += len(list(Path(".").rglob(pattern)))
        
        return test_count
    
    def get_adaptive_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive adaptive system dashboard."""
        uptime = datetime.now() - self.performance_tracker["system_uptime"]
        
        return {
            "system_status": "RUNNING" if self.running else "STOPPED",
            "system_load": self.system_load,
            "uptime_hours": uptime.total_seconds() / 3600,
            "performance_metrics": self.performance_tracker.copy(),
            "prediction_interval": self.prediction_interval,
            "optimization_interval": self.optimization_interval,
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "last_prediction": self.last_prediction_time.isoformat(),
            "last_optimization": self.last_optimization_time.isoformat(),
            "quality_dashboard": self.quality_monitor.get_quality_dashboard(),
            "adaptive_scaling": self.adaptive_scaler.get_scaling_status()
        }


def main():
    """Run the adaptive quality system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 ADAPTIVE QUALITY SYSTEM - GENERATION 3 (MAKE IT SCALE)")
    print("=" * 70)
    
    async def run_system():
        adaptive_system = AdaptiveQualitySystem()
        
        try:
            print("Starting adaptive quality system...")
            await adaptive_system.start_adaptive_system()
        except KeyboardInterrupt:
            print("\nShutting down adaptive system...")
            await adaptive_system.stop_adaptive_system()
        
        # Save ML model
        adaptive_system.ml_predictor.save_model()
        
        # Display final dashboard
        dashboard = adaptive_system.get_adaptive_dashboard()
        print(f"\n📊 Final System Dashboard:")
        print(f"System Load: {dashboard['system_load']:.2f}")
        print(f"Uptime: {dashboard['uptime_hours']:.1f} hours")
        print(f"Predictions Made: {dashboard['performance_metrics']['predictions_made']}")
        print(f"Optimizations Applied: {dashboard['performance_metrics']['optimizations_applied']}")
        print(f"Overall Health: {dashboard['quality_dashboard']['overall_health_score']:.1f}%")
    
    # Run the async system
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        print("\nAdaptive system shutdown complete.")


if __name__ == "__main__":
    main()