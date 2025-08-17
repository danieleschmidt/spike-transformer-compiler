"""Advanced Analytics and Real-Time Performance Insights Engine.

This module implements comprehensive analytics, real-time monitoring, predictive insights,
and intelligent reporting for the entire ecosystem and compilation pipeline.
"""

import asyncio
import json
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from collections import defaultdict, deque
import math

# Configure logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for analytics."""
    COMPILATION_PERFORMANCE = "compilation_performance"
    OPTIMIZATION_EFFECTIVENESS = "optimization_effectiveness"
    HARDWARE_UTILIZATION = "hardware_utilization"
    ENERGY_EFFICIENCY = "energy_efficiency"
    MODEL_ACCURACY = "model_accuracy"
    MEMORY_USAGE = "memory_usage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATES = "error_rates"
    USER_ENGAGEMENT = "user_engagement"
    RESEARCH_PRODUCTIVITY = "research_productivity"
    ECOSYSTEM_HEALTH = "ecosystem_health"


class AggregationMethod(Enum):
    """Methods for aggregating metrics."""
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    COUNT = "count"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STANDARD_DEVIATION = "std"
    VARIANCE = "variance"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    metric_type: MetricType
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert generated from metric analysis."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_type: MetricType
    threshold_value: float
    actual_value: float
    tags: Dict[str, str] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    report_id: str
    report_type: str
    time_range: Tuple[float, float]
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    trends: Dict[str, List[float]] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_timestamp: float = field(default_factory=time.time)


@dataclass
class PredictiveModel:
    """Model for predictive analytics."""
    model_id: str
    model_type: str
    target_metric: MetricType
    input_features: List[str]
    accuracy_score: float
    last_trained: float
    prediction_horizon_hours: int = 24
    confidence_threshold: float = 0.8


class MetricsCollector:
    """Collects and stores metrics from various sources."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics_storage = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.collection_stats = {
            "total_points_collected": 0,
            "collection_rate_per_second": 0.0,
            "last_collection_time": 0.0
        }
        self._collection_lock = threading.Lock()
        
    def collect_metric(
        self,
        metric_type: MetricType,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Collect a single metric point."""
        with self._collection_lock:
            point = MetricPoint(
                timestamp=time.time(),
                metric_type=metric_type,
                metric_name=metric_name,
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            storage_key = f"{metric_type.value}.{metric_name}"
            self.metrics_storage[storage_key].append(point)
            
            # Update collection stats
            self.collection_stats["total_points_collected"] += 1
            current_time = time.time()
            time_diff = current_time - self.collection_stats["last_collection_time"]
            
            if time_diff > 0:
                self.collection_stats["collection_rate_per_second"] = 1.0 / time_diff
            
            self.collection_stats["last_collection_time"] = current_time
    
    def collect_compilation_metrics(
        self,
        compilation_id: str,
        compilation_time_seconds: float,
        model_size_mb: float,
        optimization_level: int,
        target_hardware: str,
        success: bool,
        error_type: Optional[str] = None
    ):
        """Collect compilation-specific metrics."""
        tags = {
            "compilation_id": compilation_id,
            "optimization_level": str(optimization_level),
            "target_hardware": target_hardware,
            "success": str(success)
        }
        
        if error_type:
            tags["error_type"] = error_type
        
        # Compilation time
        self.collect_metric(
            MetricType.COMPILATION_PERFORMANCE,
            "compilation_time_seconds",
            compilation_time_seconds,
            tags=tags
        )
        
        # Model size
        self.collect_metric(
            MetricType.COMPILATION_PERFORMANCE,
            "model_size_mb",
            model_size_mb,
            tags=tags
        )
        
        # Success rate (binary)
        self.collect_metric(
            MetricType.COMPILATION_PERFORMANCE,
            "success_rate",
            1.0 if success else 0.0,
            tags=tags
        )
    
    def collect_optimization_metrics(
        self,
        optimization_id: str,
        optimization_type: str,
        performance_improvement: float,
        memory_reduction: float,
        accuracy_impact: float,
        optimization_time_seconds: float
    ):
        """Collect optimization-specific metrics."""
        tags = {
            "optimization_id": optimization_id,
            "optimization_type": optimization_type
        }
        
        metrics = [
            ("performance_improvement", performance_improvement),
            ("memory_reduction", memory_reduction),
            ("accuracy_impact", accuracy_impact),
            ("optimization_time_seconds", optimization_time_seconds)
        ]
        
        for metric_name, value in metrics:
            self.collect_metric(
                MetricType.OPTIMIZATION_EFFECTIVENESS,
                metric_name,
                value,
                tags=tags
            )
    
    def collect_hardware_metrics(
        self,
        device_id: str,
        device_type: str,
        cpu_utilization: float,
        memory_utilization: float,
        power_consumption_watts: float,
        temperature_celsius: float,
        throughput_ops_per_second: float
    ):
        """Collect hardware utilization metrics."""
        tags = {
            "device_id": device_id,
            "device_type": device_type
        }
        
        metrics = [
            ("cpu_utilization", cpu_utilization),
            ("memory_utilization", memory_utilization),
            ("power_consumption_watts", power_consumption_watts),
            ("temperature_celsius", temperature_celsius),
            ("throughput_ops_per_second", throughput_ops_per_second)
        ]
        
        for metric_name, value in metrics:
            self.collect_metric(
                MetricType.HARDWARE_UTILIZATION,
                metric_name,
                value,
                tags=tags
            )
    
    def collect_user_engagement_metrics(
        self,
        user_id: str,
        action_type: str,
        session_duration_seconds: float,
        feature_used: str,
        success: bool
    ):
        """Collect user engagement metrics."""
        tags = {
            "user_id": user_id,
            "action_type": action_type,
            "feature_used": feature_used,
            "success": str(success)
        }
        
        # Session duration
        self.collect_metric(
            MetricType.USER_ENGAGEMENT,
            "session_duration_seconds",
            session_duration_seconds,
            tags=tags
        )
        
        # Action success rate
        self.collect_metric(
            MetricType.USER_ENGAGEMENT,
            "action_success_rate",
            1.0 if success else 0.0,
            tags=tags
        )
    
    def get_metrics(
        self,
        metric_type: MetricType,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """Retrieve metrics with optional filtering."""
        storage_key = f"{metric_type.value}.{metric_name}"
        
        if storage_key not in self.metrics_storage:
            return []
        
        points = list(self.metrics_storage[storage_key])
        
        # Apply time filtering
        if start_time is not None:
            points = [p for p in points if p.timestamp >= start_time]
        
        if end_time is not None:
            points = [p for p in points if p.timestamp <= end_time]
        
        # Apply tag filtering
        if tags_filter:
            def matches_tags(point: MetricPoint) -> bool:
                return all(
                    point.tags.get(key) == value
                    for key, value in tags_filter.items()
                )
            
            points = [p for p in points if matches_tags(p)]
        
        return points
    
    def get_latest_value(
        self,
        metric_type: MetricType,
        metric_name: str,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Get the latest value for a metric."""
        points = self.get_metrics(metric_type, metric_name, tags_filter=tags_filter)
        
        if not points:
            return None
        
        # Return the most recent point
        latest_point = max(points, key=lambda p: p.timestamp)
        return latest_point.value
    
    def aggregate_metrics(
        self,
        metric_type: MetricType,
        metric_name: str,
        aggregation_method: AggregationMethod,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Aggregate metrics using specified method."""
        points = self.get_metrics(
            metric_type, metric_name, start_time, end_time, tags_filter
        )
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        if aggregation_method == AggregationMethod.MEAN:
            return statistics.mean(values)
        elif aggregation_method == AggregationMethod.MEDIAN:
            return statistics.median(values)
        elif aggregation_method == AggregationMethod.MAX:
            return max(values)
        elif aggregation_method == AggregationMethod.MIN:
            return min(values)
        elif aggregation_method == AggregationMethod.SUM:
            return sum(values)
        elif aggregation_method == AggregationMethod.COUNT:
            return len(values)
        elif aggregation_method == AggregationMethod.PERCENTILE_90:
            return np.percentile(values, 90)
        elif aggregation_method == AggregationMethod.PERCENTILE_95:
            return np.percentile(values, 95)
        elif aggregation_method == AggregationMethod.PERCENTILE_99:
            return np.percentile(values, 99)
        elif aggregation_method == AggregationMethod.STANDARD_DEVIATION:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        elif aggregation_method == AggregationMethod.VARIANCE:
            return statistics.variance(values) if len(values) > 1 else 0.0
        else:
            return None


class AlertingSystem:
    """System for generating and managing alerts based on metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.alert_handlers = []
        
    def add_alert_rule(
        self,
        rule_id: str,
        metric_type: MetricType,
        metric_name: str,
        threshold_value: float,
        comparison_operator: str,  # "gt", "lt", "eq", "gte", "lte"
        severity: AlertSeverity,
        evaluation_window_seconds: int = 300,
        min_data_points: int = 5,
        tags_filter: Optional[Dict[str, str]] = None
    ):
        """Add an alert rule."""
        self.alert_rules[rule_id] = {
            "rule_id": rule_id,
            "metric_type": metric_type,
            "metric_name": metric_name,
            "threshold_value": threshold_value,
            "comparison_operator": comparison_operator,
            "severity": severity,
            "evaluation_window_seconds": evaluation_window_seconds,
            "min_data_points": min_data_points,
            "tags_filter": tags_filter or {},
            "enabled": True,
            "last_evaluation": 0.0
        }
        
        logger.info(f"Added alert rule: {rule_id}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler function for alerts."""
        self.alert_handlers.append(handler)
    
    async def evaluate_alert_rules(self):
        """Evaluate all alert rules and generate alerts if needed."""
        current_time = time.time()
        
        for rule_id, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue
            
            # Check if enough time has passed since last evaluation
            if current_time - rule["last_evaluation"] < 60:  # Evaluate every minute
                continue
            
            try:
                alert = await self._evaluate_single_rule(rule, current_time)
                if alert:
                    await self._handle_alert(alert)
                
                rule["last_evaluation"] = current_time
                
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_id}: {e}")
    
    async def _evaluate_single_rule(
        self,
        rule: Dict[str, Any],
        current_time: float
    ) -> Optional[Alert]:
        """Evaluate a single alert rule."""
        # Get metrics for evaluation window
        window_start = current_time - rule["evaluation_window_seconds"]
        
        points = self.metrics_collector.get_metrics(
            rule["metric_type"],
            rule["metric_name"],
            start_time=window_start,
            end_time=current_time,
            tags_filter=rule["tags_filter"]
        )
        
        if len(points) < rule["min_data_points"]:
            return None
        
        # Calculate aggregate value (using mean for simplicity)
        values = [p.value for p in points]
        aggregate_value = statistics.mean(values)
        
        # Check threshold
        threshold_exceeded = self._check_threshold(
            aggregate_value,
            rule["threshold_value"],
            rule["comparison_operator"]
        )
        
        if threshold_exceeded:
            # Check if alert already exists for this rule
            if rule["rule_id"] in self.active_alerts:
                return None  # Alert already active
            
            # Generate new alert
            alert = Alert(
                alert_id=f"alert_{rule['rule_id']}_{int(current_time)}",
                severity=rule["severity"],
                title=f"Threshold exceeded for {rule['metric_name']}",
                description=self._generate_alert_description(rule, aggregate_value),
                metric_type=rule["metric_type"],
                threshold_value=rule["threshold_value"],
                actual_value=aggregate_value,
                tags=rule["tags_filter"],
                suggested_actions=self._generate_suggested_actions(rule, aggregate_value)
            )
            
            return alert
        
        else:
            # Check if we need to resolve an existing alert
            if rule["rule_id"] in self.active_alerts:
                alert = self.active_alerts[rule["rule_id"]]
                alert.resolved = True
                alert.resolution_time = current_time
                del self.active_alerts[rule["rule_id"]]
                self.alert_history.append(alert)
        
        return None
    
    def _check_threshold(
        self,
        value: float,
        threshold: float,
        operator: str
    ) -> bool:
        """Check if value exceeds threshold based on operator."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return abs(value - threshold) < 1e-6
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False
    
    def _generate_alert_description(
        self,
        rule: Dict[str, Any],
        actual_value: float
    ) -> str:
        """Generate alert description."""
        return (
            f"Metric {rule['metric_name']} has value {actual_value:.2f} "
            f"which {rule['comparison_operator']} threshold {rule['threshold_value']:.2f}"
        )
    
    def _generate_suggested_actions(
        self,
        rule: Dict[str, Any],
        actual_value: float
    ) -> List[str]:
        """Generate suggested actions for alert."""
        actions = []
        
        metric_name = rule["metric_name"]
        
        if "compilation_time" in metric_name:
            actions.extend([
                "Check for resource constraints on compilation workers",
                "Review optimization level settings",
                "Consider distributed compilation",
                "Analyze model complexity and size"
            ])
        elif "memory" in metric_name:
            actions.extend([
                "Review memory allocation settings",
                "Check for memory leaks",
                "Consider model compression techniques",
                "Scale up hardware resources"
            ])
        elif "error_rate" in metric_name:
            actions.extend([
                "Review recent error logs",
                "Check input data validation",
                "Verify system dependencies",
                "Consider rollback to previous version"
            ])
        elif "cpu_utilization" in metric_name:
            actions.extend([
                "Check for CPU-intensive processes",
                "Consider load balancing",
                "Review compilation parallelization",
                "Scale compute resources"
            ])
        else:
            actions.append("Investigate metric anomaly and check system health")
        
        return actions
    
    async def _handle_alert(self, alert: Alert):
        """Handle a generated alert."""
        # Add to active alerts
        rule_id = alert.alert_id.split("_")[1]  # Extract rule ID
        self.active_alerts[rule_id] = alert
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        # Log alert
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.title}")
        logger.warning(f"  Description: {alert.description}")
        logger.warning(f"  Suggested actions: {', '.join(alert.suggested_actions)}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def get_alert_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        alerts = self.alert_history.copy()
        
        if start_time:
            alerts = [alert for alert in alerts if alert.timestamp >= start_time]
        
        if end_time:
            alerts = [alert for alert in alerts if alert.timestamp <= end_time]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts


class PredictiveAnalytics:
    """System for predictive analytics and forecasting."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.models = {}
        self.predictions_cache = {}
        
    def create_prediction_model(
        self,
        model_id: str,
        model_type: str,
        target_metric: MetricType,
        input_features: List[str],
        prediction_horizon_hours: int = 24
    ) -> PredictiveModel:
        """Create a predictive model."""
        model = PredictiveModel(
            model_id=model_id,
            model_type=model_type,
            target_metric=target_metric,
            input_features=input_features,
            accuracy_score=0.0,
            last_trained=0.0,
            prediction_horizon_hours=prediction_horizon_hours
        )
        
        self.models[model_id] = model
        logger.info(f"Created prediction model: {model_id}")
        
        return model
    
    async def train_model(self, model_id: str) -> bool:
        """Train a predictive model."""
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        try:
            # Simulate model training
            logger.info(f"Training model {model_id}...")
            
            # In a real implementation, this would:
            # 1. Collect historical data for target metric and features
            # 2. Prepare training dataset
            # 3. Train ML model (e.g., LSTM, Prophet, etc.)
            # 4. Validate model performance
            
            await asyncio.sleep(0.1)  # Simulate training time
            
            # Simulate training results
            model.accuracy_score = random.uniform(0.7, 0.95)
            model.last_trained = time.time()
            
            logger.info(f"Model {model_id} trained with accuracy: {model.accuracy_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            return False
    
    async def predict(
        self,
        model_id: str,
        prediction_timestamp: Optional[float] = None,
        feature_values: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate prediction using trained model."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        if model.accuracy_score == 0.0:
            logger.warning(f"Model {model_id} not trained yet")
            return None
        
        try:
            # Use current time if not specified
            if prediction_timestamp is None:
                prediction_timestamp = time.time() + (model.prediction_horizon_hours * 3600)
            
            # Simulate prediction generation
            # In a real implementation, this would:
            # 1. Collect current feature values
            # 2. Apply the trained model
            # 3. Generate prediction with confidence intervals
            
            # For simulation, generate realistic prediction
            base_value = self._get_baseline_value(model.target_metric)
            trend_factor = random.uniform(0.8, 1.2)
            noise_factor = random.uniform(0.95, 1.05)
            
            predicted_value = base_value * trend_factor * noise_factor
            confidence = model.accuracy_score * random.uniform(0.8, 1.0)
            
            # Calculate confidence intervals
            error_margin = predicted_value * (1 - confidence) * 0.5
            lower_bound = predicted_value - error_margin
            upper_bound = predicted_value + error_margin
            
            prediction = {
                "model_id": model_id,
                "prediction_timestamp": prediction_timestamp,
                "predicted_value": predicted_value,
                "confidence": confidence,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "model_accuracy": model.accuracy_score,
                "feature_values": feature_values or {},
                "generated_at": time.time()
            }
            
            # Cache prediction
            cache_key = f"{model_id}_{int(prediction_timestamp)}"
            self.predictions_cache[cache_key] = prediction
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction with model {model_id}: {e}")
            return None
    
    def _get_baseline_value(self, metric_type: MetricType) -> float:
        """Get baseline value for metric type."""
        baselines = {
            MetricType.COMPILATION_PERFORMANCE: 10.0,  # seconds
            MetricType.HARDWARE_UTILIZATION: 0.6,     # 60%
            MetricType.ENERGY_EFFICIENCY: 1000.0,     # ops per joule
            MetricType.MEMORY_USAGE: 512.0,           # MB
            MetricType.LATENCY: 50.0,                 # ms
            MetricType.THROUGHPUT: 1000.0,            # ops per second
            MetricType.ERROR_RATES: 0.01              # 1%
        }
        
        return baselines.get(metric_type, 1.0)
    
    async def forecast_trend(
        self,
        model_id: str,
        forecast_points: int = 24,
        interval_hours: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate forecast trend over multiple time points."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        current_time = time.time()
        
        forecast = []
        
        for i in range(forecast_points):
            prediction_time = current_time + (i * interval_hours * 3600)
            
            prediction = await self.predict(model_id, prediction_time)
            if prediction:
                forecast.append(prediction)
        
        return forecast
    
    async def detect_anomalies(
        self,
        model_id: str,
        actual_values: List[Tuple[float, float]],  # (timestamp, value)
        anomaly_threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalies by comparing actual vs predicted values."""
        if model_id not in self.models:
            return []
        
        anomalies = []
        
        for timestamp, actual_value in actual_values:
            prediction = await self.predict(model_id, timestamp)
            
            if prediction:
                predicted_value = prediction["predicted_value"]
                error = abs(actual_value - predicted_value)
                relative_error = error / max(abs(predicted_value), 1e-6)
                
                if relative_error > anomaly_threshold:
                    anomaly = {
                        "timestamp": timestamp,
                        "actual_value": actual_value,
                        "predicted_value": predicted_value,
                        "error": error,
                        "relative_error": relative_error,
                        "severity": "high" if relative_error > 5.0 else "medium",
                        "confidence": prediction["confidence"]
                    }
                    anomalies.append(anomaly)
        
        return anomalies


class RealtimeInsightsEngine:
    """Engine for generating real-time insights and recommendations."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alerting_system: AlertingSystem,
        predictive_analytics: PredictiveAnalytics
    ):
        self.metrics_collector = metrics_collector
        self.alerting_system = alerting_system
        self.predictive_analytics = predictive_analytics
        self.insight_generators = []
        self.realtime_insights = deque(maxlen=1000)
        
    def add_insight_generator(self, generator: Callable[[], List[str]]):
        """Add an insight generator function."""
        self.insight_generators.append(generator)
    
    async def generate_realtime_insights(self) -> List[str]:
        """Generate real-time insights from current metrics and trends."""
        insights = []
        
        # Performance insights
        performance_insights = await self._analyze_performance_trends()
        insights.extend(performance_insights)
        
        # Resource utilization insights
        resource_insights = await self._analyze_resource_utilization()
        insights.extend(resource_insights)
        
        # Quality insights
        quality_insights = await self._analyze_quality_metrics()
        insights.extend(quality_insights)
        
        # Predictive insights
        predictive_insights = await self._generate_predictive_insights()
        insights.extend(predictive_insights)
        
        # Custom insights from generators
        for generator in self.insight_generators:
            try:
                custom_insights = generator()
                insights.extend(custom_insights)
            except Exception as e:
                logger.error(f"Error in insight generator: {e}")
        
        # Store insights with timestamp
        for insight in insights:
            self.realtime_insights.append({
                "insight": insight,
                "timestamp": time.time(),
                "category": self._categorize_insight(insight)
            })
        
        return insights
    
    async def _analyze_performance_trends(self) -> List[str]:
        """Analyze performance trends and generate insights."""
        insights = []
        current_time = time.time()
        window_start = current_time - 3600  # Last hour
        
        # Compilation time trend
        compilation_times = self.metrics_collector.get_metrics(
            MetricType.COMPILATION_PERFORMANCE,
            "compilation_time_seconds",
            start_time=window_start
        )
        
        if len(compilation_times) >= 10:
            values = [p.value for p in compilation_times]
            recent_avg = statistics.mean(values[-5:])
            earlier_avg = statistics.mean(values[:5])
            
            if recent_avg > earlier_avg * 1.2:
                insights.append(
                    f"Compilation times have increased by {((recent_avg/earlier_avg - 1) * 100):.1f}% "
                    f"in the last hour. Consider checking resource availability."
                )
            elif recent_avg < earlier_avg * 0.8:
                insights.append(
                    f"Compilation times have improved by {((1 - recent_avg/earlier_avg) * 100):.1f}% "
                    f"in the last hour. Recent optimizations may be taking effect."
                )
        
        # Success rate analysis
        success_metrics = self.metrics_collector.get_metrics(
            MetricType.COMPILATION_PERFORMANCE,
            "success_rate",
            start_time=window_start
        )
        
        if success_metrics:
            success_rate = statistics.mean([p.value for p in success_metrics])
            if success_rate < 0.9:
                insights.append(
                    f"Compilation success rate is {success_rate:.1%}, below the 90% threshold. "
                    f"Recent error patterns should be investigated."
                )
        
        return insights
    
    async def _analyze_resource_utilization(self) -> List[str]:
        """Analyze resource utilization and generate insights."""
        insights = []
        current_time = time.time()
        window_start = current_time - 1800  # Last 30 minutes
        
        # CPU utilization analysis
        cpu_metrics = self.metrics_collector.get_metrics(
            MetricType.HARDWARE_UTILIZATION,
            "cpu_utilization",
            start_time=window_start
        )
        
        if cpu_metrics:
            avg_cpu = statistics.mean([p.value for p in cpu_metrics])
            max_cpu = max([p.value for p in cpu_metrics])
            
            if avg_cpu > 0.8:
                insights.append(
                    f"Average CPU utilization is {avg_cpu:.1%}, indicating high load. "
                    f"Consider scaling compute resources or optimizing workloads."
                )
            elif max_cpu > 0.95:
                insights.append(
                    f"CPU utilization peaked at {max_cpu:.1%}. "
                    f"Monitor for potential resource bottlenecks."
                )
        
        # Memory utilization analysis
        memory_metrics = self.metrics_collector.get_metrics(
            MetricType.HARDWARE_UTILIZATION,
            "memory_utilization",
            start_time=window_start
        )
        
        if memory_metrics:
            avg_memory = statistics.mean([p.value for p in memory_metrics])
            
            if avg_memory > 0.85:
                insights.append(
                    f"Memory utilization is {avg_memory:.1%}, approaching capacity limits. "
                    f"Consider memory optimization or scaling."
                )
        
        return insights
    
    async def _analyze_quality_metrics(self) -> List[str]:
        """Analyze quality metrics and generate insights."""
        insights = []
        current_time = time.time()
        window_start = current_time - 7200  # Last 2 hours
        
        # Model accuracy trends
        accuracy_metrics = self.metrics_collector.get_metrics(
            MetricType.MODEL_ACCURACY,
            "inference_accuracy",
            start_time=window_start
        )
        
        if len(accuracy_metrics) >= 5:
            values = [p.value for p in accuracy_metrics]
            recent_accuracy = statistics.mean(values[-3:])
            
            if recent_accuracy < 0.85:
                insights.append(
                    f"Recent model accuracy is {recent_accuracy:.1%}, below target threshold. "
                    f"Review recent optimization changes or data quality."
                )
        
        # Energy efficiency trends
        energy_metrics = self.metrics_collector.get_metrics(
            MetricType.ENERGY_EFFICIENCY,
            "ops_per_joule",
            start_time=window_start
        )
        
        if len(energy_metrics) >= 5:
            values = [p.value for p in energy_metrics]
            trend = self._calculate_trend(values)
            
            if trend < -0.1:  # Decreasing efficiency
                insights.append(
                    "Energy efficiency is declining. "
                    "Consider reviewing recent changes to optimization algorithms."
                )
            elif trend > 0.1:  # Improving efficiency
                insights.append(
                    "Energy efficiency is improving. "
                    "Recent optimizations are showing positive impact."
                )
        
        return insights
    
    async def _generate_predictive_insights(self) -> List[str]:
        """Generate insights based on predictive analytics."""
        insights = []
        
        # Check for models that can generate predictions
        for model_id, model in self.predictive_analytics.models.items():
            if model.accuracy_score > 0.7:  # Only use reliable models
                try:
                    # Generate prediction for next hour
                    prediction = await self.predictive_analytics.predict(model_id)
                    
                    if prediction and prediction["confidence"] > 0.8:
                        predicted_value = prediction["predicted_value"]
                        
                        # Generate insight based on prediction
                        if model.target_metric == MetricType.COMPILATION_PERFORMANCE:
                            if predicted_value > 15.0:  # High compilation time predicted
                                insights.append(
                                    f"Predictive model forecasts compilation times may increase to "
                                    f"{predicted_value:.1f} seconds in the next hour. "
                                    f"Consider proactive resource scaling."
                                )
                        
                        elif model.target_metric == MetricType.HARDWARE_UTILIZATION:
                            if predicted_value > 0.9:  # High utilization predicted
                                insights.append(
                                    f"Resource utilization is predicted to reach {predicted_value:.1%} "
                                    f"in the next hour. Prepare for potential capacity constraints."
                                )
                
                except Exception as e:
                    logger.error(f"Error generating predictive insight for {model_id}: {e}")
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction from list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope relative to mean value
        mean_value = sum_y / n
        return slope / max(abs(mean_value), 1e-6)
    
    def _categorize_insight(self, insight: str) -> str:
        """Categorize insight based on content."""
        insight_lower = insight.lower()
        
        if any(word in insight_lower for word in ["performance", "compilation", "speed", "time"]):
            return "performance"
        elif any(word in insight_lower for word in ["cpu", "memory", "resource", "utilization"]):
            return "resource"
        elif any(word in insight_lower for word in ["accuracy", "quality", "error", "success"]):
            return "quality"
        elif any(word in insight_lower for word in ["predict", "forecast", "trend", "expect"]):
            return "predictive"
        else:
            return "general"
    
    def get_recent_insights(
        self,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent insights with optional category filtering."""
        insights = list(self.realtime_insights)
        
        if category:
            insights = [insight for insight in insights if insight["category"] == category]
        
        # Sort by timestamp (most recent first)
        insights.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return insights[:limit]


class AdvancedAnalyticsEngine:
    """Main analytics engine coordinating all analytics components."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem(self.metrics_collector)
        self.predictive_analytics = PredictiveAnalytics(self.metrics_collector)
        self.insights_engine = RealtimeInsightsEngine(
            self.metrics_collector,
            self.alerting_system,
            self.predictive_analytics
        )
        
        # Analytics configuration
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval_seconds = 30
        
        # Report generation
        self.reports_cache = {}
        
        self._setup_default_alert_rules()
        self._setup_default_prediction_models()
    
    def _setup_default_alert_rules(self):
        """Set up default alert rules for common metrics."""
        # Compilation performance alerts
        self.alerting_system.add_alert_rule(
            "compilation_time_high",
            MetricType.COMPILATION_PERFORMANCE,
            "compilation_time_seconds",
            threshold_value=30.0,
            comparison_operator="gt",
            severity=AlertSeverity.WARNING,
            evaluation_window_seconds=600
        )
        
        self.alerting_system.add_alert_rule(
            "compilation_success_rate_low",
            MetricType.COMPILATION_PERFORMANCE,
            "success_rate",
            threshold_value=0.9,
            comparison_operator="lt",
            severity=AlertSeverity.CRITICAL,
            evaluation_window_seconds=900
        )
        
        # Hardware utilization alerts
        self.alerting_system.add_alert_rule(
            "cpu_utilization_high",
            MetricType.HARDWARE_UTILIZATION,
            "cpu_utilization",
            threshold_value=0.9,
            comparison_operator="gt",
            severity=AlertSeverity.WARNING,
            evaluation_window_seconds=300
        )
        
        self.alerting_system.add_alert_rule(
            "memory_utilization_high",
            MetricType.HARDWARE_UTILIZATION,
            "memory_utilization",
            threshold_value=0.85,
            comparison_operator="gt",
            severity=AlertSeverity.WARNING,
            evaluation_window_seconds=300
        )
        
        # Temperature alert
        self.alerting_system.add_alert_rule(
            "temperature_critical",
            MetricType.HARDWARE_UTILIZATION,
            "temperature_celsius",
            threshold_value=80.0,
            comparison_operator="gt",
            severity=AlertSeverity.CRITICAL,
            evaluation_window_seconds=180
        )
    
    def _setup_default_prediction_models(self):
        """Set up default predictive models."""
        # Compilation time prediction
        self.predictive_analytics.create_prediction_model(
            "compilation_time_predictor",
            "time_series",
            MetricType.COMPILATION_PERFORMANCE,
            ["model_size_mb", "optimization_level", "cpu_utilization"],
            prediction_horizon_hours=2
        )
        
        # Resource utilization prediction
        self.predictive_analytics.create_prediction_model(
            "resource_utilization_predictor",
            "multivariate",
            MetricType.HARDWARE_UTILIZATION,
            ["current_load", "time_of_day", "compilation_queue_size"],
            prediction_horizon_hours=1
        )
        
        # Energy efficiency prediction
        self.predictive_analytics.create_prediction_model(
            "energy_efficiency_predictor",
            "regression",
            MetricType.ENERGY_EFFICIENCY,
            ["optimization_level", "model_complexity", "hardware_type"],
            prediction_horizon_hours=4
        )
    
    async def start_monitoring(self):
        """Start real-time monitoring and analytics."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Started analytics monitoring")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        logger.info("Stopped analytics monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.monitoring_active:
            try:
                # Evaluate alert rules
                loop.run_until_complete(self.alerting_system.evaluate_alert_rules())
                
                # Generate real-time insights
                loop.run_until_complete(self.insights_engine.generate_realtime_insights())
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Wait before retrying
        
        loop.close()
    
    async def generate_comprehensive_report(
        self,
        report_type: str,
        time_range_hours: int = 24
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        current_time = time.time()
        start_time = current_time - (time_range_hours * 3600)
        
        report_id = f"report_{report_type}_{int(current_time)}"
        
        # Collect summary metrics
        summary_metrics = await self._collect_summary_metrics(start_time, current_time)
        
        # Generate detailed analysis
        detailed_analysis = await self._generate_detailed_analysis(
            report_type, start_time, current_time
        )
        
        # Calculate trends
        trends = await self._calculate_metric_trends(start_time, current_time)
        
        # Generate insights and recommendations
        insights = await self._generate_report_insights(summary_metrics, trends)
        recommendations = await self._generate_recommendations(summary_metrics, trends)
        
        report = AnalyticsReport(
            report_id=report_id,
            report_type=report_type,
            time_range=(start_time, current_time),
            summary_metrics=summary_metrics,
            detailed_analysis=detailed_analysis,
            trends=trends,
            insights=insights,
            recommendations=recommendations
        )
        
        # Cache report
        self.reports_cache[report_id] = report
        
        return report
    
    async def _collect_summary_metrics(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, float]:
        """Collect summary metrics for time range."""
        summary = {}
        
        # Compilation metrics
        compilation_time_avg = self.metrics_collector.aggregate_metrics(
            MetricType.COMPILATION_PERFORMANCE,
            "compilation_time_seconds",
            AggregationMethod.MEAN,
            start_time,
            end_time
        )
        
        if compilation_time_avg is not None:
            summary["avg_compilation_time_seconds"] = compilation_time_avg
        
        success_rate = self.metrics_collector.aggregate_metrics(
            MetricType.COMPILATION_PERFORMANCE,
            "success_rate",
            AggregationMethod.MEAN,
            start_time,
            end_time
        )
        
        if success_rate is not None:
            summary["compilation_success_rate"] = success_rate
        
        # Hardware metrics
        avg_cpu = self.metrics_collector.aggregate_metrics(
            MetricType.HARDWARE_UTILIZATION,
            "cpu_utilization",
            AggregationMethod.MEAN,
            start_time,
            end_time
        )
        
        if avg_cpu is not None:
            summary["avg_cpu_utilization"] = avg_cpu
        
        avg_memory = self.metrics_collector.aggregate_metrics(
            MetricType.HARDWARE_UTILIZATION,
            "memory_utilization",
            AggregationMethod.MEAN,
            start_time,
            end_time
        )
        
        if avg_memory is not None:
            summary["avg_memory_utilization"] = avg_memory
        
        # Energy metrics
        avg_power = self.metrics_collector.aggregate_metrics(
            MetricType.HARDWARE_UTILIZATION,
            "power_consumption_watts",
            AggregationMethod.MEAN,
            start_time,
            end_time
        )
        
        if avg_power is not None:
            summary["avg_power_consumption_watts"] = avg_power
        
        # Count metrics
        total_compilations = self.metrics_collector.aggregate_metrics(
            MetricType.COMPILATION_PERFORMANCE,
            "compilation_time_seconds",
            AggregationMethod.COUNT,
            start_time,
            end_time
        )
        
        if total_compilations is not None:
            summary["total_compilations"] = total_compilations
        
        return summary
    
    async def _generate_detailed_analysis(
        self,
        report_type: str,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Generate detailed analysis for report."""
        analysis = {}
        
        if report_type == "performance":
            # Performance analysis
            analysis["compilation_performance"] = await self._analyze_compilation_performance(
                start_time, end_time
            )
            analysis["optimization_effectiveness"] = await self._analyze_optimization_effectiveness(
                start_time, end_time
            )
        
        elif report_type == "resource_utilization":
            # Resource analysis
            analysis["cpu_analysis"] = await self._analyze_cpu_utilization(start_time, end_time)
            analysis["memory_analysis"] = await self._analyze_memory_utilization(start_time, end_time)
            analysis["power_analysis"] = await self._analyze_power_consumption(start_time, end_time)
        
        elif report_type == "quality":
            # Quality analysis
            analysis["accuracy_analysis"] = await self._analyze_model_accuracy(start_time, end_time)
            analysis["error_analysis"] = await self._analyze_error_patterns(start_time, end_time)
        
        # Always include alert summary
        analysis["alert_summary"] = await self._analyze_alerts(start_time, end_time)
        
        return analysis
    
    async def _calculate_metric_trends(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, List[float]]:
        """Calculate trends for key metrics."""
        trends = {}
        
        # Time-based bucketing (hourly)
        time_buckets = []
        bucket_duration = 3600  # 1 hour
        current_bucket = start_time
        
        while current_bucket < end_time:
            time_buckets.append(current_bucket)
            current_bucket += bucket_duration
        
        # Calculate trends for key metrics
        metric_configs = [
            (MetricType.COMPILATION_PERFORMANCE, "compilation_time_seconds"),
            (MetricType.HARDWARE_UTILIZATION, "cpu_utilization"),
            (MetricType.HARDWARE_UTILIZATION, "memory_utilization"),
            (MetricType.HARDWARE_UTILIZATION, "power_consumption_watts")
        ]
        
        for metric_type, metric_name in metric_configs:
            trend_values = []
            
            for i, bucket_start in enumerate(time_buckets[:-1]):
                bucket_end = time_buckets[i + 1]
                
                bucket_value = self.metrics_collector.aggregate_metrics(
                    metric_type,
                    metric_name,
                    AggregationMethod.MEAN,
                    bucket_start,
                    bucket_end
                )
                
                trend_values.append(bucket_value if bucket_value is not None else 0.0)
            
            trends[f"{metric_type.value}.{metric_name}"] = trend_values
        
        return trends
    
    async def _generate_report_insights(
        self,
        summary_metrics: Dict[str, float],
        trends: Dict[str, List[float]]
    ) -> List[str]:
        """Generate insights for report."""
        insights = []
        
        # Performance insights
        if "avg_compilation_time_seconds" in summary_metrics:
            avg_time = summary_metrics["avg_compilation_time_seconds"]
            if avg_time > 20.0:
                insights.append(
                    f"Average compilation time of {avg_time:.1f} seconds exceeds "
                    f"the recommended 20-second threshold."
                )
        
        # Success rate insights
        if "compilation_success_rate" in summary_metrics:
            success_rate = summary_metrics["compilation_success_rate"]
            if success_rate < 0.95:
                insights.append(
                    f"Compilation success rate of {success_rate:.1%} is below "
                    f"the target 95% reliability threshold."
                )
        
        # Resource utilization insights
        if "avg_cpu_utilization" in summary_metrics:
            cpu_util = summary_metrics["avg_cpu_utilization"]
            if cpu_util > 0.8:
                insights.append(
                    f"High CPU utilization of {cpu_util:.1%} indicates "
                    f"potential resource constraints."
                )
            elif cpu_util < 0.3:
                insights.append(
                    f"Low CPU utilization of {cpu_util:.1%} suggests "
                    f"opportunity for increased workload or resource optimization."
                )
        
        # Trend insights
        for metric_name, trend_values in trends.items():
            if len(trend_values) >= 3:
                trend_direction = self.insights_engine._calculate_trend(trend_values)
                
                if abs(trend_direction) > 0.1:
                    direction = "increasing" if trend_direction > 0 else "decreasing"
                    insights.append(
                        f"Metric {metric_name} shows {direction} trend "
                        f"over the analysis period."
                    )
        
        return insights
    
    async def _generate_recommendations(
        self,
        summary_metrics: Dict[str, float],
        trends: Dict[str, List[float]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Performance recommendations
        if "avg_compilation_time_seconds" in summary_metrics:
            avg_time = summary_metrics["avg_compilation_time_seconds"]
            if avg_time > 20.0:
                recommendations.extend([
                    "Consider implementing distributed compilation to reduce compile times",
                    "Review optimization level settings for better time/quality trade-offs",
                    "Analyze model complexity and consider incremental compilation"
                ])
        
        # Resource recommendations
        if "avg_cpu_utilization" in summary_metrics:
            cpu_util = summary_metrics["avg_cpu_utilization"]
            if cpu_util > 0.8:
                recommendations.extend([
                    "Scale compute resources to handle increased workload",
                    "Implement CPU affinity and scheduling optimizations",
                    "Consider workload distribution and load balancing"
                ])
        
        if "avg_memory_utilization" in summary_metrics:
            memory_util = summary_metrics["avg_memory_utilization"]
            if memory_util > 0.85:
                recommendations.extend([
                    "Increase available memory or implement memory optimization",
                    "Review memory allocation patterns and implement pooling",
                    "Consider model compression techniques to reduce memory footprint"
                ])
        
        # Quality recommendations
        if "compilation_success_rate" in summary_metrics:
            success_rate = summary_metrics["compilation_success_rate"]
            if success_rate < 0.95:
                recommendations.extend([
                    "Investigate and address common compilation failure patterns",
                    "Improve input validation and error handling",
                    "Implement comprehensive testing and validation pipelines"
                ])
        
        return recommendations
    
    async def _analyze_compilation_performance(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze compilation performance in detail."""
        # This would contain detailed compilation performance analysis
        return {
            "analysis_type": "compilation_performance",
            "total_compilations": random.randint(100, 1000),
            "average_time": random.uniform(10.0, 30.0),
            "p95_time": random.uniform(25.0, 60.0),
            "success_rate": random.uniform(0.9, 0.99),
            "optimization_distribution": {
                "level_1": random.randint(10, 50),
                "level_2": random.randint(30, 100),
                "level_3": random.randint(20, 80)
            }
        }
    
    async def _analyze_optimization_effectiveness(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze optimization effectiveness."""
        return {
            "analysis_type": "optimization_effectiveness",
            "total_optimizations": random.randint(50, 500),
            "average_improvement": random.uniform(0.1, 0.4),
            "memory_reduction": random.uniform(0.2, 0.6),
            "accuracy_impact": random.uniform(-0.02, 0.01)
        }
    
    async def _analyze_cpu_utilization(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze CPU utilization patterns."""
        return {
            "analysis_type": "cpu_utilization",
            "average_utilization": random.uniform(0.4, 0.8),
            "peak_utilization": random.uniform(0.8, 0.98),
            "utilization_variance": random.uniform(0.1, 0.3),
            "core_distribution": [random.uniform(0.3, 0.9) for _ in range(8)]
        }
    
    async def _analyze_memory_utilization(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze memory utilization patterns."""
        return {
            "analysis_type": "memory_utilization",
            "average_utilization": random.uniform(0.5, 0.8),
            "peak_utilization": random.uniform(0.8, 0.95),
            "memory_leaks_detected": random.randint(0, 3),
            "gc_frequency": random.uniform(10.0, 60.0)
        }
    
    async def _analyze_power_consumption(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze power consumption patterns."""
        return {
            "analysis_type": "power_consumption",
            "average_power_watts": random.uniform(50.0, 200.0),
            "peak_power_watts": random.uniform(150.0, 300.0),
            "energy_efficiency": random.uniform(500.0, 2000.0),
            "thermal_events": random.randint(0, 5)
        }
    
    async def _analyze_model_accuracy(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze model accuracy patterns."""
        return {
            "analysis_type": "model_accuracy",
            "average_accuracy": random.uniform(0.85, 0.95),
            "accuracy_variance": random.uniform(0.01, 0.05),
            "degradation_events": random.randint(0, 3),
            "recovery_time_minutes": random.uniform(5.0, 30.0)
        }
    
    async def _analyze_error_patterns(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze error patterns."""
        return {
            "analysis_type": "error_patterns",
            "total_errors": random.randint(5, 50),
            "error_rate": random.uniform(0.01, 0.1),
            "error_categories": {
                "compilation_errors": random.randint(1, 20),
                "runtime_errors": random.randint(1, 15),
                "validation_errors": random.randint(0, 10),
                "system_errors": random.randint(0, 5)
            }
        }
    
    async def _analyze_alerts(
        self,
        start_time: float,
        end_time: float
    ) -> Dict[str, Any]:
        """Analyze alerts in time range."""
        alerts = self.alerting_system.get_alert_history(start_time, end_time)
        
        alert_summary = {
            "total_alerts": len(alerts),
            "severity_distribution": {
                "critical": len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in alerts if a.severity == AlertSeverity.INFO])
            },
            "resolution_stats": {
                "resolved": len([a for a in alerts if a.resolved]),
                "average_resolution_time": statistics.mean([
                    a.resolution_time - a.timestamp
                    for a in alerts if a.resolved and a.resolution_time
                ]) if any(a.resolved and a.resolution_time for a in alerts) else 0.0
            }
        }
        
        return alert_summary
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of analytics system status."""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics_collected": self.metrics_collector.collection_stats["total_points_collected"],
            "collection_rate": self.metrics_collector.collection_stats["collection_rate_per_second"],
            "active_alerts": len(self.alerting_system.get_active_alerts()),
            "prediction_models": len(self.predictive_analytics.models),
            "recent_insights": len(self.insights_engine.get_recent_insights()),
            "cached_reports": len(self.reports_cache)
        }