"""Distributed compilation cluster management."""

import asyncio
import time
import uuid
import hashlib
import socket
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import Future
import weakref

from ..logging_config import compiler_logger
from ..performance import PerformanceProfiler
from ..security import SecurityValidator


class NodeStatus(Enum):
    """Node status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class NodeCapabilities:
    """Describes node computational capabilities."""
    cpu_cores: int
    memory_gb: float
    disk_gb: float
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    specialized_accelerators: List[str] = field(default_factory=list)
    supported_backends: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 4
    network_bandwidth_mbps: float = 1000.0


@dataclass
class NodeMetrics:
    """Real-time node performance metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    disk_utilization: float
    network_io_mbps: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_task_duration: float
    queue_length: int


@dataclass
class CompilationTask:
    """Distributed compilation task."""
    task_id: str
    model_hash: str
    input_shape: Tuple[int, ...]
    optimization_level: int
    target_backend: str
    priority: int = 0
    requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class ClusterNode:
    """Individual node in the compilation cluster."""
    
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        capabilities: NodeCapabilities,
        heartbeat_interval: float = 30.0
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.capabilities = capabilities
        self.heartbeat_interval = heartbeat_interval
        
        # State tracking
        self.status = NodeStatus.INITIALIZING
        self.last_heartbeat = 0.0
        self.current_metrics: Optional[NodeMetrics] = None
        
        # Task management
        self.active_tasks: Dict[str, CompilationTask] = {}
        self.task_history: List[str] = []
        
        # Performance tracking
        self.performance_profiler = PerformanceProfiler()
        self.security_validator = SecurityValidator()
        
        # Communication
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        compiler_logger.logger.info(f"Cluster node {node_id} initialized at {host}:{port}")
    
    def start(self) -> None:
        """Start the cluster node."""
        self.status = NodeStatus.ACTIVE
        self.last_heartbeat = time.time()
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        
        compiler_logger.logger.info(f"Cluster node {self.node_id} started")
    
    def stop(self) -> None:
        """Stop the cluster node."""
        self.status = NodeStatus.OFFLINE
        self._shutdown_event.set()
        
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            self.cancel_task(task_id)
        
        compiler_logger.logger.info(f"Cluster node {self.node_id} stopped")
    
    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for node health monitoring."""
        while not self._shutdown_event.wait(self.heartbeat_interval):
            try:
                self._update_metrics()
                self.last_heartbeat = time.time()
                
                # Update status based on load
                if self.current_metrics:
                    if self.current_metrics.cpu_utilization > 90 or len(self.active_tasks) >= self.capabilities.max_concurrent_tasks:
                        self.status = NodeStatus.OVERLOADED
                    elif self.current_metrics.cpu_utilization > 75:
                        self.status = NodeStatus.BUSY
                    else:
                        self.status = NodeStatus.ACTIVE
                        
            except Exception as e:
                compiler_logger.logger.error(f"Heartbeat error on node {self.node_id}: {e}")
                self.status = NodeStatus.ERROR
    
    def _update_metrics(self) -> None:
        """Update node performance metrics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            # Calculate network I/O (simplified)
            net_io = psutil.net_io_counters()
            network_io = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
            
            # Task statistics
            completed_count = len([t for t in self.task_history if t])
            failed_count = sum(1 for task in self.active_tasks.values() if task.error)
            
            # Average task duration (simplified)
            avg_duration = 30.0  # Placeholder
            
            self.current_metrics = NodeMetrics(
                timestamp=time.time(),
                cpu_utilization=cpu_percent,
                memory_utilization=memory_info.percent,
                disk_utilization=disk_info.percent,
                network_io_mbps=network_io,
                active_tasks=len(self.active_tasks),
                completed_tasks=completed_count,
                failed_tasks=failed_count,
                average_task_duration=avg_duration,
                queue_length=0  # Simplified
            )
            
        except ImportError:
            # Fallback metrics if psutil not available
            self.current_metrics = NodeMetrics(
                timestamp=time.time(),
                cpu_utilization=50.0,
                memory_utilization=60.0,
                disk_utilization=30.0,
                network_io_mbps=100.0,
                active_tasks=len(self.active_tasks),
                completed_tasks=len(self.task_history),
                failed_tasks=0,
                average_task_duration=30.0,
                queue_length=0
            )
        except Exception as e:
            compiler_logger.logger.warning(f"Failed to update metrics for node {self.node_id}: {e}")
    
    def can_accept_task(self, task: CompilationTask) -> bool:
        """Check if node can accept a new task."""
        if self.status not in [NodeStatus.ACTIVE, NodeStatus.BUSY]:
            return False
        
        if len(self.active_tasks) >= self.capabilities.max_concurrent_tasks:
            return False
        
        # Check backend compatibility
        if task.target_backend not in self.capabilities.supported_backends:
            return False
        
        # Check resource requirements
        if task.requirements:
            required_memory = task.requirements.get('memory_gb', 0)
            if required_memory > self.capabilities.memory_gb:
                return False
        
        return True
    
    def assign_task(self, task: CompilationTask) -> bool:
        """Assign a task to this node."""
        if not self.can_accept_task(task):
            return False
        
        task.assigned_node = self.node_id
        task.started_at = time.time()
        self.active_tasks[task.task_id] = task
        
        # Start task execution asynchronously
        threading.Thread(
            target=self._execute_task,
            args=(task,),
            daemon=True
        ).start()
        
        compiler_logger.logger.info(f"Task {task.task_id} assigned to node {self.node_id}")
        return True
    
    def _execute_task(self, task: CompilationTask) -> None:
        """Execute compilation task."""
        try:
            # Security validation
            self.security_validator.validate_compilation_task(task)
            
            # Performance tracking
            with self.performance_profiler.profile_stage(f"task_{task.task_id}"):
                # Simulate compilation process
                result = self._simulate_compilation(task)
                
            task.result = result
            task.completed_at = time.time()
            
            # Move to history
            self.task_history.append(task.task_id)
            del self.active_tasks[task.task_id]
            
            compiler_logger.logger.info(f"Task {task.task_id} completed on node {self.node_id}")
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = time.time()
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
                
            compiler_logger.logger.error(f"Task {task.task_id} failed on node {self.node_id}: {e}")
    
    def _simulate_compilation(self, task: CompilationTask) -> Dict[str, Any]:
        """Simulate compilation process (placeholder)."""
        # This would interface with the actual spike compiler
        import time
        import random
        
        # Simulate compilation time based on optimization level
        base_time = 10.0  # 10 seconds base
        complexity_factor = task.optimization_level * 2.0
        model_size_factor = sum(task.input_shape) / 100.0
        
        compilation_time = base_time + complexity_factor + model_size_factor
        compilation_time += random.uniform(-2.0, 2.0)  # Add some variance
        
        time.sleep(min(compilation_time, 60.0))  # Cap at 60 seconds for simulation
        
        return {
            'compiled_model_size': random.randint(1024, 10240),  # KB
            'optimization_applied': True,
            'energy_estimate': random.uniform(0.1, 10.0),  # nJ
            'compilation_time': compilation_time,
            'node_id': self.node_id
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.error = "Task cancelled"
            task.completed_at = time.time()
            del self.active_tasks[task_id]
            
            compiler_logger.logger.info(f"Task {task_id} cancelled on node {self.node_id}")
            return True
        return False
    
    def get_utilization_score(self) -> float:
        """Get node utilization score for load balancing."""
        if not self.current_metrics or self.status == NodeStatus.OFFLINE:
            return 1.0  # Fully utilized (unavailable)
        
        cpu_score = self.current_metrics.cpu_utilization / 100.0
        memory_score = self.current_metrics.memory_utilization / 100.0
        task_score = len(self.active_tasks) / self.capabilities.max_concurrent_tasks
        
        # Weighted average
        return (cpu_score * 0.4 + memory_score * 0.3 + task_score * 0.3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'status': self.status.value,
            'capabilities': {
                'cpu_cores': self.capabilities.cpu_cores,
                'memory_gb': self.capabilities.memory_gb,
                'disk_gb': self.capabilities.disk_gb,
                'max_concurrent_tasks': self.capabilities.max_concurrent_tasks,
                'supported_backends': self.capabilities.supported_backends
            },
            'current_metrics': {
                'cpu_utilization': self.current_metrics.cpu_utilization if self.current_metrics else 0,
                'memory_utilization': self.current_metrics.memory_utilization if self.current_metrics else 0,
                'active_tasks': len(self.active_tasks),
                'utilization_score': self.get_utilization_score()
            } if self.current_metrics else None,
            'last_heartbeat': self.last_heartbeat
        }


class CompilationCluster:
    """Manages a cluster of compilation nodes."""
    
    def __init__(self, cluster_name: str = "spike_compiler_cluster"):
        self.cluster_name = cluster_name
        self.nodes: Dict[str, ClusterNode] = {}
        
        # Cluster management
        self.cluster_lock = threading.Lock()
        self.task_queue: List[CompilationTask] = []
        self.completed_tasks: Dict[str, CompilationTask] = {}
        
        # Health monitoring
        self.health_monitor_active = False
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Load balancing
        self.load_balancer = ClusterLoadBalancer()
        
        compiler_logger.logger.info(f"Compilation cluster '{cluster_name}' initialized")
    
    def add_node(self, node: ClusterNode) -> None:
        """Add a node to the cluster."""
        with self.cluster_lock:
            if node.node_id in self.nodes:
                raise ValueError(f"Node {node.node_id} already exists in cluster")
            
            self.nodes[node.node_id] = node
            node.start()
            
            compiler_logger.logger.info(f"Node {node.node_id} added to cluster {self.cluster_name}")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster."""
        with self.cluster_lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            node.stop()
            del self.nodes[node_id]
            
            compiler_logger.logger.info(f"Node {node_id} removed from cluster {self.cluster_name}")
            return True
    
    def submit_task(self, task: CompilationTask) -> str:
        """Submit a compilation task to the cluster."""
        task.task_id = task.task_id or str(uuid.uuid4())
        
        with self.cluster_lock:
            # Try immediate assignment
            assigned_node = self.load_balancer.select_best_node(self.nodes, task)
            if assigned_node and assigned_node.assign_task(task):
                compiler_logger.logger.info(f"Task {task.task_id} immediately assigned to node {assigned_node.node_id}")
            else:
                # Queue for later assignment
                self.task_queue.append(task)
                compiler_logger.logger.info(f"Task {task.task_id} queued for assignment")
        
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[CompilationTask]:
        """Get status of a task."""
        # Check completed tasks first
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check active tasks on nodes
        for node in self.nodes.values():
            if task_id in node.active_tasks:
                return node.active_tasks[task_id]
        
        # Check queued tasks
        for task in self.task_queue:
            if task.task_id == task_id:
                return task
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self.cluster_lock:
            # Try to cancel from nodes
            for node in self.nodes.values():
                if node.cancel_task(task_id):
                    return True
            
            # Try to remove from queue
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    self.task_queue.pop(i)
                    compiler_logger.logger.info(f"Task {task_id} cancelled from queue")
                    return True
        
        return False
    
    def start_cluster(self) -> None:
        """Start cluster operations."""
        self.health_monitor_active = True
        
        def cluster_management_loop():
            while not self.shutdown_event.wait(10):  # Check every 10 seconds
                try:
                    self._process_task_queue()
                    self._collect_completed_tasks()
                    self._monitor_node_health()
                except Exception as e:
                    compiler_logger.logger.error(f"Cluster management error: {e}")
        
        self.health_monitor_thread = threading.Thread(target=cluster_management_loop, daemon=True)
        self.health_monitor_thread.start()
        
        compiler_logger.logger.info(f"Cluster {self.cluster_name} started")
    
    def stop_cluster(self) -> None:
        """Stop cluster operations."""
        self.health_monitor_active = False
        self.shutdown_event.set()
        
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        
        # Stop all nodes
        for node in list(self.nodes.values()):
            node.stop()
        
        compiler_logger.logger.info(f"Cluster {self.cluster_name} stopped")
    
    def _process_task_queue(self) -> None:
        """Process queued tasks."""
        with self.cluster_lock:
            remaining_tasks = []
            
            for task in self.task_queue:
                assigned_node = self.load_balancer.select_best_node(self.nodes, task)
                if assigned_node and assigned_node.assign_task(task):
                    compiler_logger.logger.info(f"Queued task {task.task_id} assigned to node {assigned_node.node_id}")
                else:
                    remaining_tasks.append(task)
            
            self.task_queue = remaining_tasks
    
    def _collect_completed_tasks(self) -> None:
        """Collect completed tasks from nodes."""
        for node in self.nodes.values():
            completed_task_ids = []
            
            for task_id, task in node.active_tasks.items():
                if task.completed_at is not None:
                    self.completed_tasks[task_id] = task
                    completed_task_ids.append(task_id)
            
            # Remove from node
            for task_id in completed_task_ids:
                if task_id in node.active_tasks:
                    del node.active_tasks[task_id]
    
    def _monitor_node_health(self) -> None:
        """Monitor node health and handle failures."""
        current_time = time.time()
        unhealthy_nodes = []
        
        for node_id, node in self.nodes.items():
            # Check heartbeat timeout
            if current_time - node.last_heartbeat > node.heartbeat_interval * 3:
                node.status = NodeStatus.OFFLINE
                unhealthy_nodes.append(node_id)
                compiler_logger.logger.warning(f"Node {node_id} marked offline due to heartbeat timeout")
        
        # Handle unhealthy nodes (could implement failover logic here)
        for node_id in unhealthy_nodes:
            # For now, just log the issue
            pass
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        active_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.ACTIVE)
        busy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.BUSY)
        offline_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.OFFLINE)
        
        total_active_tasks = sum(len(n.active_tasks) for n in self.nodes.values())
        total_capacity = sum(n.capabilities.max_concurrent_tasks for n in self.nodes.values())
        
        return {
            'cluster_name': self.cluster_name,
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'busy_nodes': busy_nodes,
            'offline_nodes': offline_nodes,
            'queued_tasks': len(self.task_queue),
            'active_tasks': total_active_tasks,
            'completed_tasks': len(self.completed_tasks),
            'total_capacity': total_capacity,
            'utilization': total_active_tasks / max(1, total_capacity),
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }


class ClusterLoadBalancer:
    """Intelligent load balancer for the compilation cluster."""
    
    def __init__(self):
        self.assignment_history = {}
        
    def select_best_node(self, nodes: Dict[str, ClusterNode], task: CompilationTask) -> Optional[ClusterNode]:
        """Select the best node for a task."""
        if not nodes:
            return None
        
        # Filter available nodes
        available_nodes = [
            node for node in nodes.values()
            if node.can_accept_task(task)
        ]
        
        if not available_nodes:
            return None
        
        # Score nodes based on multiple criteria
        best_node = None
        best_score = float('inf')
        
        for node in available_nodes:
            score = self._calculate_node_score(node, task)
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _calculate_node_score(self, node: ClusterNode, task: CompilationTask) -> float:
        """Calculate node suitability score (lower is better)."""
        # Base utilization score
        utilization_score = node.get_utilization_score()
        
        # Backend preference (lower score for exact match)
        backend_score = 0.0 if task.target_backend in node.capabilities.supported_backends else 1.0
        
        # Capacity score (prefer nodes with more available capacity)
        available_capacity = node.capabilities.max_concurrent_tasks - len(node.active_tasks)
        capacity_score = 1.0 / max(1, available_capacity)
        
        # Network locality (simplified - could use actual network metrics)
        locality_score = 0.1  # Placeholder
        
        # Weighted combination
        total_score = (
            utilization_score * 0.4 +
            backend_score * 0.3 +
            capacity_score * 0.2 +
            locality_score * 0.1
        )
        
        return total_score