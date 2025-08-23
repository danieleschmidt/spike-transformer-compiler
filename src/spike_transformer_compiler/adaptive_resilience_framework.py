"""Adaptive Resilience Framework for Neuromorphic Computing Systems.

Advanced resilience framework providing self-healing, fault tolerance,
chaos engineering, and adaptive recovery capabilities for large-scale
neuromorphic computing deployments.
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import numpy as np
import logging
from pathlib import Path

from .compiler import SpikeCompiler
from .monitoring import SystemHealthMonitor
from .security import SecurityMetrics


class FailureType(Enum):
    """Types of system failures."""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_BUG = "software_bug"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_BREACH = "security_breach"
    DATA_CORRUPTION = "data_corruption"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    EXTERNAL_DEPENDENCY = "external_dependency"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RESTART_SERVICE = "restart_service"
    FAILOVER_REPLICA = "failover_replica"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ROLLBACK_VERSION = "rollback_version"
    SCALE_RESOURCES = "scale_resources"
    ISOLATE_COMPONENT = "isolate_component"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class FailureScenario:
    """Definition of a failure scenario for testing."""
    scenario_id: str
    name: str
    description: str
    failure_type: FailureType
    impact_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    probability: float
    mttr_target: float  # Mean Time To Recovery (seconds)
    affected_components: List[str]
    chaos_parameters: Dict[str, Any]
    recovery_strategies: List[RecoveryStrategy]


@dataclass
class ResilienceEvent:
    """Record of a resilience event."""
    event_id: str
    timestamp: float
    event_type: str
    component: str
    failure_type: Optional[FailureType]
    recovery_strategy: Optional[RecoveryStrategy]
    status: str  # DETECTED, RECOVERING, RECOVERED, FAILED
    impact: Dict[str, Any]
    recovery_time: Optional[float] = None
    lessons_learned: List[str] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_id: str
    health_score: float  # 0.0 to 1.0
    availability: float
    performance_metrics: Dict[str, float]
    last_failure: Optional[float]
    failure_count: int
    recovery_count: int
    mean_recovery_time: float


class CircuitBreaker:
    """Circuit breaker pattern implementation with adaptive behavior."""
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State management
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        
        # Adaptive parameters
        self.success_rate_window = []
        self.adaptive_threshold = failure_threshold
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function call through circuit breaker."""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        if self.state == "HALF_OPEN" and self.half_open_calls >= self.half_open_max_calls:
            raise Exception(f"Circuit breaker {self.name} is HALF_OPEN with max calls exceeded")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success handling
            await self._on_success()
            return result
            
        except Exception as e:
            # Failure handling
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            self.half_open_calls += 1
            
            if self.success_count >= 3:  # Recovery threshold
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
        else:
            # Reset failure count on successful calls
            self.failure_count = max(0, self.failure_count - 1)
        
        # Update success rate for adaptive behavior
        self.success_rate_window.append(1.0)
        if len(self.success_rate_window) > 100:
            self.success_rate_window.pop(0)
        
        await self._adapt_threshold()
    
    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
        elif self.failure_count >= self.adaptive_threshold:
            self.state = "OPEN"
        
        # Update success rate for adaptive behavior
        self.success_rate_window.append(0.0)
        if len(self.success_rate_window) > 100:
            self.success_rate_window.pop(0)
        
        await self._adapt_threshold()
    
    async def _adapt_threshold(self):
        """Adapt failure threshold based on historical success rate."""
        if len(self.success_rate_window) < 20:
            return
        
        success_rate = np.mean(self.success_rate_window)
        
        if success_rate > 0.9:
            # High success rate, can be more tolerant
            self.adaptive_threshold = min(10, self.failure_threshold + 2)
        elif success_rate < 0.7:
            # Low success rate, be more aggressive
            self.adaptive_threshold = max(2, self.failure_threshold - 1)
        
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "adaptive_threshold": self.adaptive_threshold,
            "success_rate": np.mean(self.success_rate_window) if self.success_rate_window else 0.0,
            "last_failure": self.last_failure_time
        }


class ChaosEngineer:
    """Chaos engineering implementation for resilience testing."""
    
    def __init__(self, system_components: List[str]):
        self.system_components = system_components
        self.active_experiments = {}
        self.experiment_history = []
        
        # Failure injection capabilities
        self.failure_injectors = {
            FailureType.HARDWARE_FAILURE: self._inject_hardware_failure,
            FailureType.SOFTWARE_BUG: self._inject_software_bug,
            FailureType.NETWORK_PARTITION: self._inject_network_partition,
            FailureType.RESOURCE_EXHAUSTION: self._inject_resource_exhaustion,
            FailureType.PERFORMANCE_DEGRADATION: self._inject_performance_degradation
        }
    
    async def run_chaos_experiment(self, scenario: FailureScenario) -> Dict[str, Any]:
        """Run a chaos engineering experiment."""
        
        print(f"🔥 Starting Chaos Experiment: {scenario.name}")
        
        experiment_id = f"chaos_{int(time.time())}_{scenario.scenario_id}"
        experiment_start = time.time()
        
        # Record baseline metrics
        baseline_metrics = await self._collect_baseline_metrics()
        
        # Inject failure
        failure_injected = await self._inject_failure(scenario)
        
        if not failure_injected:
            print(f"⚠️  Failed to inject failure for scenario: {scenario.name}")
            return {"status": "failed", "reason": "injection_failed"}
        
        # Monitor system response
        response_data = await self._monitor_chaos_response(scenario, experiment_start)
        
        # Collect recovery metrics  
        recovery_metrics = await self._collect_recovery_metrics(baseline_metrics)
        
        # Clean up injected failures
        await self._cleanup_chaos_experiment(scenario)
        
        experiment_result = {
            "experiment_id": experiment_id,
            "scenario": asdict(scenario),
            "baseline_metrics": baseline_metrics,
            "response_data": response_data,
            "recovery_metrics": recovery_metrics,
            "duration": time.time() - experiment_start,
            "lessons_learned": await self._analyze_experiment_results(response_data, recovery_metrics)
        }
        
        self.experiment_history.append(experiment_result)
        print(f"✅ Chaos Experiment Complete: {scenario.name}")
        
        return experiment_result
    
    async def _inject_failure(self, scenario: FailureScenario) -> bool:
        """Inject failure based on scenario."""
        
        injector = self.failure_injectors.get(scenario.failure_type)
        if not injector:
            return False
        
        try:
            await injector(scenario)
            return True
        except Exception as e:
            print(f"⚠️  Failure injection error: {e}")
            return False
    
    async def _inject_hardware_failure(self, scenario: FailureScenario):
        """Simulate hardware failure."""
        # Simulate by degrading performance or making components unavailable
        for component in scenario.affected_components:
            await self._degrade_component(component, scenario.chaos_parameters.get("degradation", 0.5))
    
    async def _inject_software_bug(self, scenario: FailureScenario):
        """Simulate software bug."""
        # Introduce random exceptions or incorrect behavior
        bug_probability = scenario.chaos_parameters.get("bug_probability", 0.1)
        self._register_bug_injection(scenario.affected_components, bug_probability)
    
    async def _inject_network_partition(self, scenario: FailureScenario):
        """Simulate network partition."""
        # Simulate network delays or dropped connections
        partition_duration = scenario.chaos_parameters.get("duration", 60.0)
        await self._create_network_partition(scenario.affected_components, partition_duration)
    
    async def _inject_resource_exhaustion(self, scenario: FailureScenario):
        """Simulate resource exhaustion."""
        # Consume CPU, memory, or disk resources
        resource_type = scenario.chaos_parameters.get("resource", "memory")
        consumption_level = scenario.chaos_parameters.get("level", 0.8)
        await self._exhaust_resources(resource_type, consumption_level)
    
    async def _inject_performance_degradation(self, scenario: FailureScenario):
        """Simulate performance degradation."""
        # Add artificial delays
        delay_ms = scenario.chaos_parameters.get("delay_ms", 100)
        await self._add_artificial_delays(scenario.affected_components, delay_ms)
    
    async def _monitor_chaos_response(
        self, 
        scenario: FailureScenario, 
        start_time: float
    ) -> Dict[str, Any]:
        """Monitor system response during chaos experiment."""
        
        response_data = {
            "detection_time": None,
            "recovery_initiated": None,
            "recovery_completed": None,
            "impact_metrics": [],
            "recovery_actions": []
        }
        
        monitoring_duration = scenario.chaos_parameters.get("monitoring_duration", 300)  # 5 minutes
        end_time = start_time + monitoring_duration
        
        while time.time() < end_time:
            current_metrics = await self._collect_current_metrics()
            response_data["impact_metrics"].append({
                "timestamp": time.time(),
                "metrics": current_metrics
            })
            
            # Check if system has detected the failure
            if not response_data["detection_time"] and await self._failure_detected():
                response_data["detection_time"] = time.time() - start_time
            
            # Check if recovery has been initiated
            if not response_data["recovery_initiated"] and await self._recovery_initiated():
                response_data["recovery_initiated"] = time.time() - start_time
            
            # Check if recovery is complete
            if not response_data["recovery_completed"] and await self._recovery_completed():
                response_data["recovery_completed"] = time.time() - start_time
                break
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        return response_data
    
    async def _collect_baseline_metrics(self) -> Dict[str, Any]:
        """Collect baseline system metrics before chaos injection."""
        return {
            "cpu_usage": 25.0,
            "memory_usage": 60.0,
            "network_throughput": 1000.0,
            "response_time": 0.1,
            "error_rate": 0.001,
            "availability": 1.0
        }
    
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        # Simulate degraded metrics during chaos
        return {
            "cpu_usage": 45.0 + random.uniform(-10, 10),
            "memory_usage": 75.0 + random.uniform(-5, 15),
            "network_throughput": 800.0 + random.uniform(-200, 100),
            "response_time": 0.2 + random.uniform(0, 0.1),
            "error_rate": 0.01 + random.uniform(0, 0.02),
            "availability": 0.85 + random.uniform(0, 0.1)
        }
    
    async def _collect_recovery_metrics(self, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics after recovery."""
        # Simulate recovered metrics (not quite back to baseline)
        return {
            "cpu_usage": baseline["cpu_usage"] * 1.1,
            "memory_usage": baseline["memory_usage"] * 1.05,
            "network_throughput": baseline["network_throughput"] * 0.95,
            "response_time": baseline["response_time"] * 1.2,
            "error_rate": baseline["error_rate"] * 2.0,
            "availability": baseline["availability"] * 0.98
        }


class SelfHealingSystem:
    """Self-healing system with automated recovery capabilities."""
    
    def __init__(self, components: List[str]):
        self.components = components
        self.component_health = {}
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        
        # Initialize health tracking
        for component in components:
            self.component_health[component] = ComponentHealth(
                component_id=component,
                health_score=1.0,
                availability=1.0,
                performance_metrics={},
                last_failure=None,
                failure_count=0,
                recovery_count=0,
                mean_recovery_time=0.0
            )
            
            # Initialize circuit breakers
            self.circuit_breakers[component] = CircuitBreaker(
                name=f"{component}_breaker"
            )
        
        # Configure recovery strategies
        self._configure_recovery_strategies()
        
        # Healing parameters
        self.healing_threshold = 0.5  # Health score below which healing is triggered
        self.healing_active = True
        
    def _configure_recovery_strategies(self):
        """Configure recovery strategies for different failure types."""
        self.recovery_strategies = {
            FailureType.HARDWARE_FAILURE: [
                RecoveryStrategy.FAILOVER_REPLICA,
                RecoveryStrategy.SCALE_RESOURCES
            ],
            FailureType.SOFTWARE_BUG: [
                RecoveryStrategy.RESTART_SERVICE,
                RecoveryStrategy.ROLLBACK_VERSION
            ],
            FailureType.NETWORK_PARTITION: [
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.SCALE_RESOURCES,
                RecoveryStrategy.ISOLATE_COMPONENT
            ],
            FailureType.PERFORMANCE_DEGRADATION: [
                RecoveryStrategy.SCALE_RESOURCES,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ]
        }
    
    async def monitor_and_heal(self):
        """Continuous monitoring and healing loop."""
        
        while self.healing_active:
            try:
                # Check health of all components
                for component in self.components:
                    health = await self._assess_component_health(component)
                    self.component_health[component] = health
                    
                    # Trigger healing if health is below threshold
                    if health.health_score < self.healing_threshold:
                        await self._initiate_healing(component, health)
                
                # Update circuit breaker states
                await self._update_circuit_breakers()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"⚠️  Self-healing monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _assess_component_health(self, component: str) -> ComponentHealth:
        """Assess health of a system component."""
        
        current_health = self.component_health[component]
        
        # Simulate health assessment
        performance_metrics = await self._collect_component_metrics(component)
        
        # Calculate health score based on metrics
        health_score = self._calculate_health_score(performance_metrics)
        
        # Update availability based on recent failures
        availability = self._calculate_availability(current_health)
        
        return ComponentHealth(
            component_id=component,
            health_score=health_score,
            availability=availability,
            performance_metrics=performance_metrics,
            last_failure=current_health.last_failure,
            failure_count=current_health.failure_count,
            recovery_count=current_health.recovery_count,
            mean_recovery_time=current_health.mean_recovery_time
        )
    
    async def _collect_component_metrics(self, component: str) -> Dict[str, float]:
        """Collect performance metrics for a component."""
        # Simulate component metrics
        base_metrics = {
            "response_time": 0.1,
            "throughput": 1000.0,
            "error_rate": 0.001,
            "cpu_usage": 30.0,
            "memory_usage": 50.0
        }
        
        # Add some noise
        for key in base_metrics:
            base_metrics[key] *= (1.0 + random.uniform(-0.1, 0.1))
        
        return base_metrics
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall health score from metrics."""
        
        # Weighted health score calculation
        weights = {
            "response_time": -0.3,  # Lower is better
            "throughput": 0.2,      # Higher is better  
            "error_rate": -0.4,     # Lower is better
            "cpu_usage": -0.05,     # Lower is better
            "memory_usage": -0.05   # Lower is better
        }
        
        score = 1.0
        for metric, value in metrics.items():
            if metric in weights:
                if weights[metric] > 0:  # Higher is better
                    score += weights[metric] * min(1.0, value / 1000.0)
                else:  # Lower is better
                    score += weights[metric] * min(1.0, value / 100.0)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_availability(self, health: ComponentHealth) -> float:
        """Calculate component availability."""
        
        # Simple availability calculation based on failure history
        if health.failure_count == 0:
            return 1.0
        
        # Decay availability based on failure frequency
        time_since_last_failure = time.time() - (health.last_failure or 0)
        failure_impact = health.failure_count / max(1, time_since_last_failure / 3600)  # failures per hour
        
        availability = max(0.0, 1.0 - failure_impact * 0.1)
        return availability
    
    async def _initiate_healing(self, component: str, health: ComponentHealth):
        """Initiate healing process for unhealthy component."""
        
        print(f"🔧 Initiating healing for {component} (health: {health.health_score:.3f})")
        
        # Determine primary failure type
        failure_type = await self._diagnose_failure_type(component, health)
        
        # Select recovery strategies
        strategies = self.recovery_strategies.get(failure_type, [RecoveryStrategy.RESTART_SERVICE])
        
        # Execute recovery strategies in order
        for strategy in strategies:
            success = await self._execute_recovery_strategy(component, strategy)
            if success:
                print(f"✅ Healing successful for {component} using {strategy.value}")
                
                # Update recovery statistics
                self.component_health[component].recovery_count += 1
                break
        else:
            print(f"❌ Healing failed for {component}")
    
    async def _diagnose_failure_type(self, component: str, health: ComponentHealth) -> FailureType:
        """Diagnose the type of failure affecting the component."""
        
        metrics = health.performance_metrics
        
        # Simple heuristic-based diagnosis
        if metrics.get("error_rate", 0) > 0.1:
            return FailureType.SOFTWARE_BUG
        elif metrics.get("response_time", 0) > 1.0:
            return FailureType.PERFORMANCE_DEGRADATION
        elif metrics.get("cpu_usage", 0) > 90 or metrics.get("memory_usage", 0) > 90:
            return FailureType.RESOURCE_EXHAUSTION
        else:
            return FailureType.HARDWARE_FAILURE
    
    async def _execute_recovery_strategy(self, component: str, strategy: RecoveryStrategy) -> bool:
        """Execute specific recovery strategy."""
        
        try:
            if strategy == RecoveryStrategy.RESTART_SERVICE:
                await self._restart_service(component)
                
            elif strategy == RecoveryStrategy.FAILOVER_REPLICA:
                await self._failover_to_replica(component)
                
            elif strategy == RecoveryStrategy.SCALE_RESOURCES:
                await self._scale_resources(component)
                
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                await self._enable_graceful_degradation(component)
                
            elif strategy == RecoveryStrategy.ROLLBACK_VERSION:
                await self._rollback_version(component)
                
            elif strategy == RecoveryStrategy.ISOLATE_COMPONENT:
                await self._isolate_component(component)
                
            return True
            
        except Exception as e:
            print(f"⚠️  Recovery strategy {strategy.value} failed for {component}: {e}")
            return False
    
    async def _restart_service(self, component: str):
        """Restart service component."""
        print(f"🔄 Restarting service: {component}")
        await asyncio.sleep(2)  # Simulate restart time
        
    async def _failover_to_replica(self, component: str):
        """Failover to replica."""
        print(f"🔀 Failing over to replica: {component}")
        await asyncio.sleep(1)  # Simulate failover time
        
    async def _scale_resources(self, component: str):
        """Scale resources for component."""
        print(f"📈 Scaling resources: {component}")
        await asyncio.sleep(3)  # Simulate scaling time
        
    async def _enable_graceful_degradation(self, component: str):
        """Enable graceful degradation mode."""
        print(f"⚡ Enabling graceful degradation: {component}")
        
    async def _rollback_version(self, component: str):
        """Rollback to previous version."""
        print(f"⏪ Rolling back version: {component}")
        await asyncio.sleep(5)  # Simulate rollback time
        
    async def _isolate_component(self, component: str):
        """Isolate problematic component."""
        print(f"🔒 Isolating component: {component}")


class AdaptiveResilienceFramework:
    """Main adaptive resilience framework."""
    
    def __init__(
        self,
        system_components: List[str],
        enable_chaos_engineering: bool = True,
        enable_self_healing: bool = True,
        resilience_storage_path: str = "resilience_data"
    ):
        self.system_components = system_components
        self.storage_path = Path(resilience_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.chaos_engineer = ChaosEngineer(system_components) if enable_chaos_engineering else None
        self.self_healing = SelfHealingSystem(system_components) if enable_self_healing else None
        
        # Configuration
        self.enable_chaos_engineering = enable_chaos_engineering
        self.enable_self_healing = enable_self_healing
        
        # State tracking
        self.resilience_events: List[ResilienceEvent] = []
        self.failure_scenarios = self._initialize_failure_scenarios()
        
        # Monitoring
        self.monitoring_active = False
        
    def _initialize_failure_scenarios(self) -> Dict[str, FailureScenario]:
        """Initialize predefined failure scenarios."""
        
        scenarios = {
            "compiler_crash": FailureScenario(
                scenario_id="compiler_crash_001",
                name="Compiler Service Crash",
                description="Simulate sudden crash of the compilation service",
                failure_type=FailureType.SOFTWARE_BUG,
                impact_level="HIGH",
                probability=0.05,
                mttr_target=60.0,
                affected_components=["compiler"],
                chaos_parameters={"crash_probability": 1.0, "recovery_delay": 10},
                recovery_strategies=[RecoveryStrategy.RESTART_SERVICE, RecoveryStrategy.FAILOVER_REPLICA]
            ),
            "memory_exhaustion": FailureScenario(
                scenario_id="memory_exhaustion_001", 
                name="Memory Exhaustion Attack",
                description="Exhaust available memory resources",
                failure_type=FailureType.RESOURCE_EXHAUSTION,
                impact_level="CRITICAL",
                probability=0.02,
                mttr_target=120.0,
                affected_components=["runtime", "compiler"],
                chaos_parameters={"resource": "memory", "level": 0.95, "duration": 180},
                recovery_strategies=[RecoveryStrategy.SCALE_RESOURCES, RecoveryStrategy.ISOLATE_COMPONENT]
            ),
            "network_partition": FailureScenario(
                scenario_id="network_partition_001",
                name="Network Partition",
                description="Simulate network partition between distributed components",
                failure_type=FailureType.NETWORK_PARTITION,
                impact_level="MEDIUM", 
                probability=0.03,
                mttr_target=300.0,
                affected_components=["distributed_compiler", "edge_nodes"],
                chaos_parameters={"duration": 120, "packet_loss": 0.5},
                recovery_strategies=[RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.GRACEFUL_DEGRADATION]
            )
        }
        
        return scenarios
    
    async def start_resilience_framework(self):
        """Start the adaptive resilience framework."""
        
        print("🛡️  Starting Adaptive Resilience Framework...")
        self.monitoring_active = True
        
        # Start framework components
        tasks = []
        
        if self.self_healing:
            tasks.append(asyncio.create_task(self.self_healing.monitor_and_heal()))
        
        if self.chaos_engineer:
            tasks.append(asyncio.create_task(self._chaos_engineering_scheduler()))
        
        # Start resilience monitoring
        tasks.append(asyncio.create_task(self._resilience_monitoring()))
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _chaos_engineering_scheduler(self):
        """Schedule and run chaos engineering experiments."""
        
        while self.monitoring_active:
            try:
                # Run chaos experiments periodically
                for scenario_id, scenario in self.failure_scenarios.items():
                    # Probabilistic experiment execution
                    if random.random() < scenario.probability / 100:  # Reduce frequency
                        print(f"🎯 Scheduled chaos experiment: {scenario.name}")
                        
                        experiment_result = await self.chaos_engineer.run_chaos_experiment(scenario)
                        await self._process_experiment_results(experiment_result)
                
                # Wait between experiment cycles (run daily)
                await asyncio.sleep(86400)
                
            except Exception as e:
                print(f"⚠️  Chaos engineering scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _resilience_monitoring(self):
        """Monitor overall system resilience."""
        
        while self.monitoring_active:
            try:
                # Collect resilience metrics
                resilience_metrics = await self._collect_resilience_metrics()
                
                # Analyze resilience posture
                posture = await self._analyze_resilience_posture(resilience_metrics)
                
                # Take action if resilience is degraded
                if posture["overall_score"] < 0.7:
                    await self._improve_resilience(posture)
                
                # Log resilience event
                await self._log_resilience_event("resilience_assessment", posture)
                
                await asyncio.sleep(1800)  # Monitor every 30 minutes
                
            except Exception as e:
                print(f"⚠️  Resilience monitoring error: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_resilience_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive resilience metrics."""
        
        metrics = {
            "component_health": {},
            "failure_rates": {},
            "recovery_times": {},
            "availability": {},
            "chaos_experiment_results": []
        }
        
        # Collect component health
        if self.self_healing:
            for component, health in self.self_healing.component_health.items():
                metrics["component_health"][component] = {
                    "health_score": health.health_score,
                    "availability": health.availability,
                    "failure_count": health.failure_count,
                    "recovery_count": health.recovery_count,
                    "mean_recovery_time": health.mean_recovery_time
                }
        
        # Collect chaos experiment results
        if self.chaos_engineer:
            metrics["chaos_experiment_results"] = self.chaos_engineer.experiment_history[-10:]
        
        return metrics
    
    async def _analyze_resilience_posture(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system resilience posture."""
        
        posture = {
            "overall_score": 0.0,
            "component_scores": {},
            "weak_points": [],
            "recommendations": [],
            "trends": {}
        }
        
        # Analyze component health
        component_scores = []
        for component, health_data in metrics["component_health"].items():
            score = health_data["health_score"] * health_data["availability"]
            component_scores.append(score)
            posture["component_scores"][component] = score
            
            if score < 0.8:
                posture["weak_points"].append(component)
        
        # Calculate overall score
        posture["overall_score"] = np.mean(component_scores) if component_scores else 0.0
        
        # Generate recommendations
        if posture["overall_score"] < 0.8:
            posture["recommendations"].append("Increase monitoring frequency")
            posture["recommendations"].append("Consider additional redundancy")
        
        if len(posture["weak_points"]) > 2:
            posture["recommendations"].append("Focus on weak component recovery")
        
        return posture
    
    async def _improve_resilience(self, posture: Dict[str, Any]):
        """Take actions to improve system resilience."""
        
        print(f"📈 Improving resilience (current score: {posture['overall_score']:.3f})")
        
        # Focus on weak points
        for weak_component in posture["weak_points"]:
            if self.self_healing:
                health = self.self_healing.component_health.get(weak_component)
                if health:
                    await self.self_healing._initiate_healing(weak_component, health)
        
        # Increase chaos testing for weak areas
        if self.chaos_engineer:
            for scenario in self.failure_scenarios.values():
                if any(comp in posture["weak_points"] for comp in scenario.affected_components):
                    print(f"🔥 Running targeted chaos experiment: {scenario.name}")
                    await self.chaos_engineer.run_chaos_experiment(scenario)
    
    async def _process_experiment_results(self, experiment_result: Dict[str, Any]):
        """Process chaos experiment results."""
        
        lessons = experiment_result.get("lessons_learned", [])
        
        # Update failure scenarios based on results
        scenario_id = experiment_result["scenario"]["scenario_id"]
        if scenario_id in self.failure_scenarios:
            scenario = self.failure_scenarios[scenario_id]
            
            # Adjust probability based on results
            if experiment_result["recovery_metrics"]:
                # Success - slightly decrease probability
                scenario.probability *= 0.95
            else:
                # Failure - increase probability
                scenario.probability *= 1.1
        
        # Log experiment results
        await self._log_resilience_event("chaos_experiment", experiment_result)
    
    async def _log_resilience_event(self, event_type: str, event_data: Any):
        """Log resilience event."""
        
        event = ResilienceEvent(
            event_id=f"res_{int(time.time())}",
            timestamp=time.time(),
            event_type=event_type,
            component="framework",
            failure_type=None,
            recovery_strategy=None,
            status="LOGGED",
            impact=event_data if isinstance(event_data, dict) else {"data": event_data}
        )
        
        self.resilience_events.append(event)
        
        # Persist to storage
        event_file = self.storage_path / "resilience_events.jsonl"
        with open(event_file, 'a') as f:
            f.write(json.dumps(asdict(event)) + "\n")
    
    def get_resilience_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive resilience dashboard."""
        
        dashboard = {
            "framework_status": "ACTIVE" if self.monitoring_active else "INACTIVE",
            "total_events": len(self.resilience_events),
            "chaos_engineering_enabled": self.enable_chaos_engineering,
            "self_healing_enabled": self.enable_self_healing,
            "failure_scenarios": len(self.failure_scenarios)
        }
        
        # Component health summary
        if self.self_healing:
            health_scores = [h.health_score for h in self.self_healing.component_health.values()]
            dashboard["average_health_score"] = np.mean(health_scores) if health_scores else 0.0
            dashboard["unhealthy_components"] = len([s for s in health_scores if s < 0.8])
        
        # Chaos experiment summary
        if self.chaos_engineer:
            dashboard["chaos_experiments_run"] = len(self.chaos_engineer.experiment_history)
            dashboard["last_experiment"] = self.chaos_engineer.experiment_history[-1]["experiment_id"] if self.chaos_engineer.experiment_history else "none"
        
        # Recent events
        dashboard["recent_events"] = len([e for e in self.resilience_events if time.time() - e.timestamp < 3600])
        
        return dashboard
    
    async def stop_resilience_framework(self):
        """Stop the resilience framework."""
        
        print("🛑 Stopping Adaptive Resilience Framework...")
        self.monitoring_active = False
        
        if self.self_healing:
            self.self_healing.healing_active = False
        
        # Final resilience assessment
        final_metrics = await self._collect_resilience_metrics()
        await self._log_resilience_event("framework_stopped", final_metrics)
        
        print("✅ Resilience framework stopped.")