"""Distributed compilation coordinator for managing large-scale compilation workloads."""

import asyncio
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from concurrent.futures import Future, as_completed
import weakref

from ..logging_config import compiler_logger
from ..performance import PerformanceProfiler
from .compilation_cluster import CompilationCluster, CompilationTask, ClusterNode
from .task_scheduler import IntelligentTaskScheduler
from .result_aggregator import DistributedResultAggregator


@dataclass
class BatchCompilationRequest:
    """Request for batch compilation across multiple models."""
    request_id: str
    model_configs: List[Dict[str, Any]]
    shared_optimization_level: int = 2
    target_backends: List[str] = None
    priority: int = 0
    callback: Optional[Callable] = None
    timeout_seconds: float = 3600.0  # 1 hour default
    parallelism_level: int = 4
    
    
@dataclass
class CompilationPipeline:
    """Represents a compilation pipeline with dependencies."""
    pipeline_id: str
    stages: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    execution_order: List[str]
    shared_state: Dict[str, Any]


class DistributedCompilationCoordinator:
    """Coordinates distributed compilation across multiple clusters."""
    
    def __init__(self, name: str = "distributed_coordinator"):
        self.name = name
        self.clusters: Dict[str, CompilationCluster] = {}
        self.task_scheduler = IntelligentTaskScheduler()
        self.result_aggregator = DistributedResultAggregator()
        
        # Request management
        self.active_requests: Dict[str, BatchCompilationRequest] = {}
        self.request_futures: Dict[str, Future] = {}
        self.pipelines: Dict[str, CompilationPipeline] = {}
        
        # Performance tracking
        self.performance_profiler = PerformanceProfiler()
        self.coordinator_stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'average_completion_time': 0.0,
            'peak_concurrent_requests': 0
        }
        
        # Background services
        self.coordination_thread: Optional[threading.Thread] = None
        self.active = False
        self.shutdown_event = threading.Event()
        
        compiler_logger.logger.info(f"Distributed coordinator '{name}' initialized")
    
    def register_cluster(self, cluster_name: str, cluster: CompilationCluster) -> None:
        """Register a compilation cluster."""
        self.clusters[cluster_name] = cluster
        
        # Start cluster if not already running
        if not cluster.health_monitor_active:
            cluster.start_cluster()
        
        compiler_logger.logger.info(f"Cluster '{cluster_name}' registered with coordinator")
    
    def unregister_cluster(self, cluster_name: str) -> bool:
        """Unregister a compilation cluster."""
        if cluster_name not in self.clusters:
            return False
        
        cluster = self.clusters.pop(cluster_name)
        cluster.stop_cluster()
        
        compiler_logger.logger.info(f"Cluster '{cluster_name}' unregistered from coordinator")
        return True
    
    def start_coordination(self) -> None:
        """Start coordination services."""
        if self.active:
            return
        
        self.active = True
        
        def coordination_loop():
            while not self.shutdown_event.wait(5):  # Check every 5 seconds
                try:
                    self._monitor_requests()
                    self._rebalance_workload()
                    self._collect_results()
                    self._update_statistics()
                except Exception as e:
                    compiler_logger.logger.error(f"Coordination loop error: {e}")
        
        self.coordination_thread = threading.Thread(target=coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        compiler_logger.logger.info("Distributed coordination started")
    
    def stop_coordination(self) -> None:
        """Stop coordination services."""
        self.active = False
        self.shutdown_event.set()
        
        if self.coordination_thread:
            self.coordination_thread.join(timeout=10.0)
        
        # Stop all clusters
        for cluster in self.clusters.values():
            cluster.stop_cluster()
        
        compiler_logger.logger.info("Distributed coordination stopped")
    
    async def submit_batch_request(self, request: BatchCompilationRequest) -> str:
        """Submit a batch compilation request."""
        self.coordinator_stats['total_requests'] += 1
        
        # Validate request
        if not request.model_configs:
            raise ValueError("Batch request must contain at least one model configuration")
        
        if not self.clusters:
            raise RuntimeError("No clusters registered with coordinator")
        
        self.active_requests[request.request_id] = request
        
        # Create compilation tasks
        tasks = []
        for i, model_config in enumerate(request.model_configs):
            task = CompilationTask(
                task_id=f"{request.request_id}_task_{i}",
                model_hash=self._calculate_model_hash(model_config),
                input_shape=tuple(model_config.get('input_shape', (1, 3, 224, 224))),
                optimization_level=request.shared_optimization_level,
                target_backend=model_config.get('backend', 'simulation'),
                priority=request.priority,
                requirements=model_config.get('requirements', {})
            )
            tasks.append(task)
        
        # Schedule tasks across clusters
        scheduled_tasks = await self._schedule_batch_tasks(tasks, request)
        
        # Create future for tracking completion
        future = asyncio.create_task(self._wait_for_batch_completion(request.request_id, scheduled_tasks))
        self.request_futures[request.request_id] = future
        
        compiler_logger.logger.info(f"Batch request {request.request_id} submitted with {len(tasks)} tasks")
        return request.request_id
    
    async def _schedule_batch_tasks(
        self, 
        tasks: List[CompilationTask], 
        request: BatchCompilationRequest
    ) -> Dict[str, List[str]]:
        """Schedule batch tasks across clusters."""
        scheduled_tasks = {}
        
        # Use intelligent scheduler to determine optimal placement
        cluster_assignments = self.task_scheduler.schedule_batch_tasks(
            tasks, 
            list(self.clusters.values()),
            parallelism_level=request.parallelism_level
        )
        
        # Submit tasks to assigned clusters
        for cluster_name, task_ids in cluster_assignments.items():
            if cluster_name in self.clusters:
                cluster = self.clusters[cluster_name]
                scheduled_task_ids = []
                
                for task_id in task_ids:
                    task = next(t for t in tasks if t.task_id == task_id)
                    submitted_id = cluster.submit_task(task)
                    scheduled_task_ids.append(submitted_id)
                
                scheduled_tasks[cluster_name] = scheduled_task_ids
        
        return scheduled_tasks
    
    async def _wait_for_batch_completion(
        self, 
        request_id: str, 
        scheduled_tasks: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Wait for batch completion and aggregate results."""
        request = self.active_requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")
        
        start_time = time.time()
        completed_tasks = {}
        failed_tasks = {}
        
        # Monitor task completion
        while time.time() - start_time < request.timeout_seconds:
            all_completed = True
            
            for cluster_name, task_ids in scheduled_tasks.items():
                cluster = self.clusters.get(cluster_name)
                if not cluster:
                    continue
                
                for task_id in task_ids:
                    if task_id in completed_tasks or task_id in failed_tasks:
                        continue
                    
                    task_status = cluster.get_task_status(task_id)
                    if task_status:
                        if task_status.completed_at:
                            if task_status.error:
                                failed_tasks[task_id] = task_status
                            else:
                                completed_tasks[task_id] = task_status
                        else:
                            all_completed = False
            
            if all_completed:
                break
                
            await asyncio.sleep(1)  # Check every second
        
        # Aggregate results
        aggregated_result = self.result_aggregator.aggregate_batch_results(
            completed_tasks, failed_tasks, request
        )
        
        # Update statistics
        completion_time = time.time() - start_time
        if failed_tasks:
            self.coordinator_stats['failed_requests'] += 1
        else:
            self.coordinator_stats['completed_requests'] += 1
        
        # Update average completion time
        total_completed = self.coordinator_stats['completed_requests'] + self.coordinator_stats['failed_requests']
        if total_completed > 1:
            self.coordinator_stats['average_completion_time'] = (
                self.coordinator_stats['average_completion_time'] * (total_completed - 1) + completion_time
            ) / total_completed
        else:
            self.coordinator_stats['average_completion_time'] = completion_time
        
        # Cleanup
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        if request_id in self.request_futures:
            del self.request_futures[request_id]
        
        # Call callback if provided
        if request.callback:
            try:
                request.callback(request_id, aggregated_result)
            except Exception as e:
                compiler_logger.logger.error(f"Callback error for request {request_id}: {e}")
        
        compiler_logger.logger.info(f"Batch request {request_id} completed in {completion_time:.2f}s")
        return aggregated_result
    
    def submit_pipeline(self, pipeline: CompilationPipeline) -> str:
        """Submit a compilation pipeline with dependencies."""
        self.pipelines[pipeline.pipeline_id] = pipeline
        
        # Execute pipeline stages in dependency order
        threading.Thread(
            target=self._execute_pipeline,
            args=(pipeline,),
            daemon=True
        ).start()
        
        compiler_logger.logger.info(f"Compilation pipeline {pipeline.pipeline_id} submitted")
        return pipeline.pipeline_id
    
    def _execute_pipeline(self, pipeline: CompilationPipeline) -> None:
        """Execute compilation pipeline."""
        try:
            completed_stages = set()
            
            for stage_id in pipeline.execution_order:
                # Wait for dependencies
                stage_config = next(s for s in pipeline.stages if s['id'] == stage_id)
                dependencies = pipeline.dependencies.get(stage_id, [])
                
                # Wait for dependencies to complete
                while not all(dep in completed_stages for dep in dependencies):
                    time.sleep(1)
                
                # Execute stage
                self._execute_pipeline_stage(pipeline, stage_id, stage_config)
                completed_stages.add(stage_id)
                
                compiler_logger.logger.info(f"Pipeline {pipeline.pipeline_id} stage {stage_id} completed")
            
            compiler_logger.logger.info(f"Pipeline {pipeline.pipeline_id} completed successfully")
            
        except Exception as e:
            compiler_logger.logger.error(f"Pipeline {pipeline.pipeline_id} failed: {e}")
        finally:
            if pipeline.pipeline_id in self.pipelines:
                del self.pipelines[pipeline.pipeline_id]
    
    def _execute_pipeline_stage(
        self, 
        pipeline: CompilationPipeline, 
        stage_id: str, 
        stage_config: Dict[str, Any]
    ) -> None:
        """Execute a single pipeline stage."""
        # This would integrate with the actual compilation system
        # For now, simulate stage execution
        stage_type = stage_config.get('type', 'compilation')
        duration = stage_config.get('duration', 30.0)
        
        time.sleep(min(duration, 60.0))  # Cap simulation time
        
        # Update shared state
        pipeline.shared_state[f'{stage_id}_result'] = {
            'completed_at': time.time(),
            'status': 'success'
        }
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch compilation request."""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            
            # Count progress across clusters
            total_tasks = len(request.model_configs)
            completed_tasks = 0
            failed_tasks = 0
            
            for cluster in self.clusters.values():
                cluster_stats = cluster.get_cluster_stats()
                # This would need more detailed tracking in a real implementation
                completed_tasks += cluster_stats['completed_tasks']
                
            return {
                'request_id': request_id,
                'status': 'in_progress',
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'progress': completed_tasks / max(1, total_tasks)
            }
        
        return None
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a batch compilation request."""
        if request_id not in self.active_requests:
            return False
        
        # Cancel tasks across clusters
        cancelled_count = 0
        for cluster in self.clusters.values():
            cluster_stats = cluster.get_cluster_stats()
            for task_id in cluster_stats.get('active_tasks', []):
                if task_id.startswith(request_id):
                    if cluster.cancel_task(task_id):
                        cancelled_count += 1
        
        # Cancel future
        if request_id in self.request_futures:
            future = self.request_futures[request_id]
            future.cancel()
            del self.request_futures[request_id]
        
        # Remove from active requests
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        compiler_logger.logger.info(f"Request {request_id} cancelled ({cancelled_count} tasks)")
        return True
    
    def _monitor_requests(self) -> None:
        """Monitor active requests for timeouts and failures."""
        current_time = time.time()
        expired_requests = []
        
        for request_id, request in self.active_requests.items():
            # Check for timeout
            if hasattr(request, 'created_at'):
                if current_time - request.created_at > request.timeout_seconds:
                    expired_requests.append(request_id)
        
        # Handle expired requests
        for request_id in expired_requests:
            compiler_logger.logger.warning(f"Request {request_id} timed out")
            self.cancel_request(request_id)
    
    def _rebalance_workload(self) -> None:
        """Rebalance workload across clusters."""
        # Simple rebalancing: move queued tasks from overloaded clusters
        cluster_stats = {name: cluster.get_cluster_stats() for name, cluster in self.clusters.items()}
        
        overloaded_clusters = [
            name for name, stats in cluster_stats.items()
            if stats['utilization'] > 0.9 and stats['queued_tasks'] > 0
        ]
        
        underutilized_clusters = [
            name for name, stats in cluster_stats.items()
            if stats['utilization'] < 0.5
        ]
        
        # Move tasks (simplified implementation)
        for overloaded in overloaded_clusters:
            if not underutilized_clusters:
                break
                
            # This would implement actual task migration logic
            compiler_logger.logger.debug(f"Would rebalance tasks from {overloaded} to underutilized clusters")
    
    def _collect_results(self) -> None:
        """Collect and cache results from completed requests."""
        # This would implement result collection and caching
        pass
    
    def _update_statistics(self) -> None:
        """Update coordinator statistics."""
        # Update peak concurrent requests
        current_concurrent = len(self.active_requests)
        if current_concurrent > self.coordinator_stats['peak_concurrent_requests']:
            self.coordinator_stats['peak_concurrent_requests'] = current_concurrent
    
    def _calculate_model_hash(self, model_config: Dict[str, Any]) -> str:
        """Calculate hash for model configuration."""
        import hashlib
        import json
        
        # Create deterministic hash from model configuration
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        cluster_stats = {name: cluster.get_cluster_stats() for name, cluster in self.clusters.items()}
        
        total_nodes = sum(stats['total_nodes'] for stats in cluster_stats.values())
        total_active_tasks = sum(stats['active_tasks'] for stats in cluster_stats.values())
        total_capacity = sum(stats['total_capacity'] for stats in cluster_stats.values())
        
        return {
            'coordinator_name': self.name,
            'active': self.active,
            'registered_clusters': len(self.clusters),
            'total_nodes': total_nodes,
            'active_requests': len(self.active_requests),
            'active_pipelines': len(self.pipelines),
            'total_active_tasks': total_active_tasks,
            'total_capacity': total_capacity,
            'overall_utilization': total_active_tasks / max(1, total_capacity),
            'performance_stats': self.coordinator_stats,
            'cluster_details': cluster_stats
        }