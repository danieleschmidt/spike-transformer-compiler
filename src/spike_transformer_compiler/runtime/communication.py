"""Multi-chip communication system for distributed neuromorphic execution."""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from ..logging_config import runtime_logger


class MessageType(Enum):
    """Types of inter-chip messages."""
    SPIKE_DATA = "spike_data"
    CONTROL = "control"
    SYNC = "sync"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class ChipStatus(Enum):
    """Status of neuromorphic chips."""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class InterChipMessage:
    """Message structure for inter-chip communication."""
    
    def __init__(
        self,
        msg_id: str,
        msg_type: MessageType,
        source_chip: str,
        target_chip: str,
        payload: Any,
        priority: int = 1,
        timestamp: Optional[float] = None
    ):
        self.msg_id = msg_id
        self.msg_type = msg_type
        self.source_chip = source_chip
        self.target_chip = target_chip
        self.payload = payload
        self.priority = priority
        self.timestamp = timestamp or time.time()
        self.retry_count = 0
        self.max_retries = 3
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "source_chip": self.source_chip,
            "target_chip": self.target_chip,
            "payload": self.payload,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterChipMessage':
        """Create message from dictionary."""
        msg = cls(
            msg_id=data["msg_id"],
            msg_type=MessageType(data["msg_type"]),
            source_chip=data["source_chip"],
            target_chip=data["target_chip"],
            payload=data["payload"],
            priority=data["priority"],
            timestamp=data["timestamp"]
        )
        msg.retry_count = data["retry_count"]
        return msg


class SpikeRouter:
    """Routes spike messages between chips based on connectivity."""
    
    def __init__(self, topology: str = "mesh"):
        self.topology = topology
        self.chip_connections = {}
        self.routing_table = {}
        self.congestion_stats = {}
        
        runtime_logger.info(f"SpikeRouter initialized with {topology} topology")
        
    def set_chip_connections(self, connections: Dict[str, List[str]]) -> None:
        """Set direct connections between chips."""
        self.chip_connections = connections
        self._build_routing_table()
        
    def _build_routing_table(self) -> None:
        """Build routing table for message forwarding."""
        # Floyd-Warshall algorithm for shortest paths
        chips = list(self.chip_connections.keys())
        n = len(chips)
        
        # Initialize distance matrix
        dist = {}
        next_hop = {}
        
        for i, chip1 in enumerate(chips):
            for j, chip2 in enumerate(chips):
                if i == j:
                    dist[(chip1, chip2)] = 0
                    next_hop[(chip1, chip2)] = chip2
                elif chip2 in self.chip_connections.get(chip1, []):
                    dist[(chip1, chip2)] = 1
                    next_hop[(chip1, chip2)] = chip2
                else:
                    dist[(chip1, chip2)] = float('inf')
                    next_hop[(chip1, chip2)] = None
                    
        # Floyd-Warshall
        for k in chips:
            for i in chips:
                for j in chips:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
                        next_hop[(i, j)] = next_hop[(i, k)]
                        
        self.routing_table = next_hop
        runtime_logger.info(f"Built routing table for {n} chips")
        
    def route_message(self, message: InterChipMessage) -> Optional[str]:
        """Determine next hop for message routing."""
        source = message.source_chip
        target = message.target_chip
        
        if (source, target) in self.routing_table:
            next_hop = self.routing_table[(source, target)]
            
            # Check congestion
            if self._is_congested(source, next_hop):
                return self._find_alternative_route(source, target)
                
            return next_hop
        else:
            runtime_logger.warning(f"No route from {source} to {target}")
            return None
            
    def _is_congested(self, source: str, target: str) -> bool:
        """Check if link between chips is congested."""
        link_key = f"{source}->{target}"
        congestion = self.congestion_stats.get(link_key, 0)
        return congestion > 0.8  # 80% congestion threshold
        
    def _find_alternative_route(self, source: str, target: str) -> Optional[str]:
        """Find alternative route to avoid congestion."""
        # Simple alternative: try different next hop
        alternatives = self.chip_connections.get(source, [])
        for alt in alternatives:
            link_key = f"{source}->{alt}"
            if self.congestion_stats.get(link_key, 0) < 0.5:
                return alt
        return None
        
    def update_congestion_stats(self, source: str, target: str, utilization: float) -> None:
        """Update link congestion statistics."""
        link_key = f"{source}->{target}"
        self.congestion_stats[link_key] = utilization
        
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        return {
            "topology": self.topology,
            "total_chips": len(self.chip_connections),
            "total_links": sum(len(conns) for conns in self.chip_connections.values()),
            "congestion_stats": self.congestion_stats.copy(),
            "average_congestion": np.mean(list(self.congestion_stats.values())) if self.congestion_stats else 0
        }


class MultiChipCommunicator:
    """Manages communication between multiple neuromorphic chips."""
    
    def __init__(
        self,
        chip_ids: Optional[List[str]] = None,
        interconnect: str = "mesh",
        max_message_size: int = 1024 * 1024,  # 1MB
        buffer_size: int = 10000
    ):
        self.chip_ids = chip_ids or ["chip_0"]
        self.interconnect = interconnect
        self.max_message_size = max_message_size
        self.buffer_size = buffer_size
        
        # Communication infrastructure
        self.message_queues = {}
        self.chip_status = {}
        self.router = SpikeRouter(interconnect)
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.chip_ids))
        
        # Statistics and monitoring
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "dropped": 0,
            "retries": 0,
            "errors": 0
        }
        
        self.latency_stats = []
        self.throughput_stats = []
        
        # Communication locks
        self._lock = threading.Lock()
        self._running = False
        
        self._initialize_communication()
        
    def _initialize_communication(self) -> None:
        """Initialize communication infrastructure."""
        # Create message queues for each chip
        for chip_id in self.chip_ids:
            self.message_queues[chip_id] = queue.PriorityQueue(maxsize=self.buffer_size)
            self.chip_status[chip_id] = ChipStatus.IDLE
            
        # Set up chip connections based on topology
        self._setup_topology()
        
        runtime_logger.info(
            f"MultiChipCommunicator initialized for {len(self.chip_ids)} chips "
            f"with {self.interconnect} topology"
        )
        
    def _setup_topology(self) -> None:
        """Set up chip connections based on topology."""
        connections = {}
        
        if self.interconnect == "mesh":
            # 2D mesh topology
            n = len(self.chip_ids)
            grid_size = int(np.ceil(np.sqrt(n)))
            
            for i, chip_id in enumerate(self.chip_ids):
                row, col = i // grid_size, i % grid_size
                connections[chip_id] = []
                
                # Connect to neighbors
                neighbors = [
                    (row-1, col), (row+1, col),  # vertical neighbors
                    (row, col-1), (row, col+1)   # horizontal neighbors
                ]
                
                for nr, nc in neighbors:
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        neighbor_idx = nr * grid_size + nc
                        if neighbor_idx < len(self.chip_ids):
                            connections[chip_id].append(self.chip_ids[neighbor_idx])
                            
        elif self.interconnect == "torus":
            # Torus topology (wrap-around mesh)
            n = len(self.chip_ids)
            grid_size = int(np.ceil(np.sqrt(n)))
            
            for i, chip_id in enumerate(self.chip_ids):
                row, col = i // grid_size, i % grid_size
                connections[chip_id] = []
                
                # Connect with wrap-around
                neighbors = [
                    ((row-1) % grid_size, col), ((row+1) % grid_size, col),
                    (row, (col-1) % grid_size), (row, (col+1) % grid_size)
                ]
                
                for nr, nc in neighbors:
                    neighbor_idx = nr * grid_size + nc
                    if neighbor_idx < len(self.chip_ids) and neighbor_idx != i:
                        connections[chip_id].append(self.chip_ids[neighbor_idx])
                        
        elif self.interconnect == "hierarchical":
            # Hierarchical topology
            for i, chip_id in enumerate(self.chip_ids):
                connections[chip_id] = []
                
                # Connect to adjacent chips and hub
                if i > 0:
                    connections[chip_id].append(self.chip_ids[i-1])
                if i < len(self.chip_ids) - 1:
                    connections[chip_id].append(self.chip_ids[i+1])
                    
                # Connect to hub (first chip)
                if i != 0:
                    connections[chip_id].append(self.chip_ids[0])
                    
        else:  # default: fully connected
            for chip_id in self.chip_ids:
                connections[chip_id] = [c for c in self.chip_ids if c != chip_id]
                
        self.router.set_chip_connections(connections)
        
    def start_communication(self) -> None:
        """Start communication threads."""
        with self._lock:
            if self._running:
                return
                
            self._running = True
            
            # Start message processing threads
            for chip_id in self.chip_ids:
                self.thread_pool.submit(self._process_messages, chip_id)
                
            runtime_logger.info("Communication system started")
            
    def stop_communication(self) -> None:
        """Stop communication system."""
        with self._lock:
            self._running = False
            
        # Wait for threads to finish
        self.thread_pool.shutdown(wait=True)
        
        runtime_logger.info("Communication system stopped")
        
    def send_message(
        self,
        source_chip: str,
        target_chip: str,
        msg_type: MessageType,
        payload: Any,
        priority: int = 1
    ) -> bool:
        """Send message between chips."""
        try:
            # Validate message size
            if self._get_message_size(payload) > self.max_message_size:
                runtime_logger.error("Message size exceeds limit")
                self.message_stats["dropped"] += 1
                return False
                
            # Create message
            msg_id = f"msg_{int(time.time() * 1000000)}"
            message = InterChipMessage(
                msg_id=msg_id,
                msg_type=msg_type,
                source_chip=source_chip,
                target_chip=target_chip,
                payload=payload,
                priority=priority
            )
            
            # Route message
            next_hop = self.router.route_message(message)
            if next_hop is None:
                runtime_logger.error(f"No route from {source_chip} to {target_chip}")
                self.message_stats["dropped"] += 1
                return False
                
            # Queue message
            if next_hop in self.message_queues:
                try:
                    # Use negative priority for priority queue (lower number = higher priority)
                    self.message_queues[next_hop].put((-priority, time.time(), message), timeout=1.0)
                    self.message_stats["sent"] += 1
                    
                    runtime_logger.debug(f"Queued message {msg_id} to {next_hop}")
                    return True
                    
                except queue.Full:
                    runtime_logger.warning(f"Message queue full for chip {next_hop}")
                    self.message_stats["dropped"] += 1
                    return False
            else:
                runtime_logger.error(f"No message queue for chip {next_hop}")
                self.message_stats["dropped"] += 1
                return False
                
        except Exception as e:
            runtime_logger.error(f"Failed to send message: {str(e)}")
            self.message_stats["errors"] += 1
            return False
            
    def broadcast_message(
        self,
        source_chip: str,
        msg_type: MessageType,
        payload: Any,
        exclude_chips: Optional[Set[str]] = None
    ) -> int:
        """Broadcast message to all chips."""
        exclude_chips = exclude_chips or set()
        exclude_chips.add(source_chip)  # Don't send to self
        
        successful_sends = 0
        
        for chip_id in self.chip_ids:
            if chip_id not in exclude_chips:
                if self.send_message(source_chip, chip_id, msg_type, payload):
                    successful_sends += 1
                    
        runtime_logger.info(f"Broadcast from {source_chip}: {successful_sends}/{len(self.chip_ids)-len(exclude_chips)} successful")
        
        return successful_sends
        
    def _process_messages(self, chip_id: str) -> None:
        """Process messages for a specific chip."""
        runtime_logger.debug(f"Started message processing for {chip_id}")
        
        while self._running:
            try:
                # Get message from queue with timeout
                priority, timestamp, message = self.message_queues[chip_id].get(timeout=1.0)
                
                # Update statistics
                self.message_stats["received"] += 1
                message_latency = time.time() - message.timestamp
                self.latency_stats.append(message_latency)
                
                # Process message based on type
                self._handle_message(chip_id, message)
                
                # Update chip status
                self.chip_status[chip_id] = ChipStatus.ACTIVE
                
            except queue.Empty:
                # Timeout - update status to idle
                self.chip_status[chip_id] = ChipStatus.IDLE
                continue
            except Exception as e:
                runtime_logger.error(f"Error processing message on {chip_id}: {str(e)}")
                self.chip_status[chip_id] = ChipStatus.ERROR
                self.message_stats["errors"] += 1
                
        runtime_logger.debug(f"Stopped message processing for {chip_id}")
        
    def _handle_message(self, chip_id: str, message: InterChipMessage) -> None:
        """Handle received message."""
        if message.msg_type == MessageType.SPIKE_DATA:
            self._handle_spike_data(chip_id, message)
        elif message.msg_type == MessageType.CONTROL:
            self._handle_control_message(chip_id, message)
        elif message.msg_type == MessageType.SYNC:
            self._handle_sync_message(chip_id, message)
        elif message.msg_type == MessageType.HEARTBEAT:
            self._handle_heartbeat(chip_id, message)
        elif message.msg_type == MessageType.ERROR:
            self._handle_error_message(chip_id, message)
        else:
            runtime_logger.warning(f"Unknown message type: {message.msg_type}")
            
    def _handle_spike_data(self, chip_id: str, message: InterChipMessage) -> None:
        """Handle spike data message."""
        spike_data = message.payload
        runtime_logger.debug(f"Received spike data on {chip_id}: {len(spike_data)} spikes")
        
        # Forward if not final destination
        if message.target_chip != chip_id:
            next_hop = self.router.route_message(message)
            if next_hop and next_hop in self.message_queues:
                try:
                    self.message_queues[next_hop].put((-message.priority, time.time(), message), timeout=0.1)
                except queue.Full:
                    runtime_logger.warning(f"Failed to forward spike data to {next_hop}")
                    
    def _handle_control_message(self, chip_id: str, message: InterChipMessage) -> None:
        """Handle control message."""
        control_data = message.payload
        runtime_logger.info(f"Control message on {chip_id}: {control_data}")
        
    def _handle_sync_message(self, chip_id: str, message: InterChipMessage) -> None:
        """Handle synchronization message."""
        sync_data = message.payload
        runtime_logger.debug(f"Sync message on {chip_id}: {sync_data}")
        
    def _handle_heartbeat(self, chip_id: str, message: InterChipMessage) -> None:
        """Handle heartbeat message."""
        # Update chip status based on heartbeat
        self.chip_status[message.source_chip] = ChipStatus.ACTIVE
        
    def _handle_error_message(self, chip_id: str, message: InterChipMessage) -> None:
        """Handle error message."""
        error_info = message.payload
        runtime_logger.error(f"Error reported by {message.source_chip}: {error_info}")
        self.chip_status[message.source_chip] = ChipStatus.ERROR
        
    def _get_message_size(self, payload: Any) -> int:
        """Estimate message payload size in bytes."""
        if isinstance(payload, np.ndarray):
            return payload.nbytes
        elif isinstance(payload, (list, tuple)):
            return len(str(payload))
        elif isinstance(payload, dict):
            return len(str(payload))
        else:
            return len(str(payload))
            
    def synchronize_chips(self, timeout: float = 5.0) -> bool:
        """Synchronize all chips."""
        sync_id = f"sync_{int(time.time() * 1000)}"
        
        # Send sync message to all chips
        successful_syncs = self.broadcast_message(
            source_chip="master",
            msg_type=MessageType.SYNC,
            payload={"sync_id": sync_id, "timestamp": time.time()}
        )
        
        if successful_syncs == len(self.chip_ids) - 1:  # Exclude master
            runtime_logger.info("All chips synchronized successfully")
            return True
        else:
            runtime_logger.warning(f"Only {successful_syncs} chips synchronized")
            return False
            
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        # Calculate throughput
        current_time = time.time()
        if hasattr(self, '_last_stats_time'):
            time_diff = current_time - self._last_stats_time
            throughput = self.message_stats["sent"] / max(time_diff, 1.0)
        else:
            throughput = 0.0
            
        self._last_stats_time = current_time
        
        return {
            "message_stats": self.message_stats.copy(),
            "chip_status": {chip: status.value for chip, status in self.chip_status.items()},
            "average_latency_ms": np.mean(self.latency_stats) * 1000 if self.latency_stats else 0,
            "throughput_msgs_per_sec": throughput,
            "active_chips": len([s for s in self.chip_status.values() if s == ChipStatus.ACTIVE]),
            "error_chips": len([s for s in self.chip_status.values() if s == ChipStatus.ERROR]),
            "router_stats": self.router.get_routing_stats()
        }
        
    def cleanup(self) -> None:
        """Cleanup communication resources."""
        self.stop_communication()
        
        # Clear queues
        for queue_obj in self.message_queues.values():
            while not queue_obj.empty():
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    break
                    
        runtime_logger.info("Communication system cleanup completed")