"""Adaptive spike encoding for neuromorphic models."""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from enum import Enum


class EncodingType(Enum):
    """Supported spike encoding methods."""
    RATE = "rate"
    TEMPORAL = "temporal" 
    PHASE = "phase"
    DELTA = "delta"
    HYBRID = "hybrid"


class SpikeEncoder:
    """Base class for spike encoding methods."""
    
    def __init__(self, encoding_type: EncodingType, time_steps: int = 4):
        self.encoding_type = encoding_type
        self.time_steps = time_steps
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode input data to spike trains."""
        raise NotImplementedError
        
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Decode spike trains back to continuous values."""
        raise NotImplementedError


class RateEncoder(SpikeEncoder):
    """Rate-based spike encoding."""
    
    def __init__(self, time_steps: int = 4, max_rate: float = 1.0):
        super().__init__(EncodingType.RATE, time_steps)
        self.max_rate = max_rate
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using rate coding."""
        # Normalize data to [0, 1] range
        data_normalized = np.clip(data, 0, 1)
        
        # Generate Poisson spike trains
        spike_probs = data_normalized * self.max_rate
        
        # Generate spikes for each time step
        spikes = []
        for t in range(self.time_steps):
            time_step_spikes = np.random.random(data.shape) < spike_probs
            spikes.append(time_step_spikes.astype(np.float32))
            
        return np.stack(spikes, axis=0)
        
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Decode rate-coded spikes."""
        # Sum spikes across time and normalize
        spike_counts = np.sum(spikes, axis=0)
        decoded = spike_counts / (self.time_steps * self.max_rate)
        return decoded


class TemporalEncoder(SpikeEncoder):
    """Temporal spike encoding."""
    
    def __init__(self, time_steps: int = 4, temporal_window: float = 1.0):
        super().__init__(EncodingType.TEMPORAL, time_steps)
        self.temporal_window = temporal_window
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using temporal coding."""
        # Normalize data to [0, 1]
        data_normalized = np.clip(data, 0, 1)
        
        # Convert to spike times (higher values spike earlier)
        spike_times = (1.0 - data_normalized) * (self.time_steps - 1)
        
        # Generate temporal spike trains
        spikes = np.zeros((self.time_steps,) + data.shape, dtype=np.float32)
        
        for t in range(self.time_steps):
            # Spike if current time matches encoded time
            time_mask = np.abs(spike_times - t) < 0.5
            spikes[t] = time_mask.astype(np.float32)
            
        return spikes
        
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Decode temporal-coded spikes."""
        # Find first spike time for each location
        spike_times = np.argmax(spikes, axis=0)
        
        # Handle locations with no spikes
        no_spike_mask = np.sum(spikes, axis=0) == 0
        spike_times = spike_times.astype(np.float32)
        spike_times[no_spike_mask] = self.time_steps - 1
        
        # Convert back to normalized values
        decoded = 1.0 - (spike_times / (self.time_steps - 1))
        decoded[no_spike_mask] = 0.0
        
        return decoded


class PhaseEncoder(SpikeEncoder):
    """Phase-based spike encoding."""
    
    def __init__(self, time_steps: int = 8, frequency: float = 1.0):
        super().__init__(EncodingType.PHASE, time_steps)
        self.frequency = frequency
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using phase coding."""
        # Normalize data to [0, 2π] phase range
        data_normalized = np.clip(data, 0, 1)
        phases = data_normalized * 2 * np.pi
        
        # Generate oscillatory spikes
        spikes = np.zeros((self.time_steps,) + data.shape, dtype=np.float32)
        
        for t in range(self.time_steps):
            time_phase = 2 * np.pi * self.frequency * t / self.time_steps
            oscillation = np.sin(time_phase - phases)
            
            # Generate spikes at oscillation peaks
            spike_threshold = 0.8
            spikes[t] = (oscillation > spike_threshold).astype(np.float32)
            
        return spikes
        
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Decode phase-coded spikes."""
        # Estimate phase from spike patterns
        time_weighted_spikes = np.zeros(spikes.shape[1:])
        
        for t in range(self.time_steps):
            phase_contribution = 2 * np.pi * t / self.time_steps
            time_weighted_spikes += spikes[t] * phase_contribution
            
        # Normalize to [0, 1] range
        total_spikes = np.sum(spikes, axis=0)
        total_spikes = np.where(total_spikes == 0, 1, total_spikes)  # Avoid division by zero
        
        decoded_phases = time_weighted_spikes / total_spikes
        decoded = decoded_phases / (2 * np.pi)
        
        return np.clip(decoded, 0, 1)


class DeltaEncoder(SpikeEncoder):
    """Delta spike encoding for temporal differences."""
    
    def __init__(self, time_steps: int = 4, threshold: float = 0.1):
        super().__init__(EncodingType.DELTA, time_steps)
        self.threshold = threshold
        self.previous_values = None
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using delta coding."""
        if self.previous_values is None:
            self.previous_values = np.zeros_like(data)
            
        # Calculate differences
        differences = data - self.previous_values
        self.previous_values = data.copy()
        
        # Generate spikes for positive and negative changes
        spikes = np.zeros((self.time_steps,) + data.shape, dtype=np.float32)
        
        # Positive changes
        pos_spikes = differences > self.threshold
        # Negative changes (represented as different timing)
        neg_spikes = differences < -self.threshold
        
        # Encode in first half of time steps for positive, second half for negative
        mid_point = self.time_steps // 2
        
        for t in range(mid_point):
            spikes[t] = pos_spikes.astype(np.float32)
            
        for t in range(mid_point, self.time_steps):
            spikes[t] = neg_spikes.astype(np.float32)
            
        return spikes
        
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Decode delta-coded spikes."""
        mid_point = self.time_steps // 2
        
        # Sum positive and negative contributions
        pos_spikes = np.sum(spikes[:mid_point], axis=0)
        neg_spikes = np.sum(spikes[mid_point:], axis=0)
        
        # Reconstruct differences
        differences = (pos_spikes - neg_spikes) * self.threshold
        
        return differences


class AdaptiveEncoder:
    """Adaptive spike encoder that learns optimal encoding parameters."""
    
    def __init__(
        self,
        encoding_type: str = "hybrid",
        time_steps: int = 4,
        adaptation_rate: float = 0.01,
        target_sparsity: float = 0.05
    ):
        self.encoding_type = encoding_type
        self.time_steps = time_steps
        self.adaptation_rate = adaptation_rate
        self.target_sparsity = target_sparsity
        
        # Initialize encoders
        self.rate_encoder = RateEncoder(time_steps)
        self.temporal_encoder = TemporalEncoder(time_steps)
        self.phase_encoder = PhaseEncoder(time_steps)
        
        # Adaptive parameters
        self.encoding_weights = {
            "rate": 0.33,
            "temporal": 0.33,
            "phase": 0.34
        }
        
        # Performance tracking
        self.sparsity_history = []
        self.accuracy_history = []
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Adaptive encoding using weighted combination."""
        if self.encoding_type == "hybrid":
            return self._hybrid_encode(data)
        elif self.encoding_type == "rate":
            return self.rate_encoder.encode(data)
        elif self.encoding_type == "temporal":
            return self.temporal_encoder.encode(data)
        elif self.encoding_type == "phase":
            return self.phase_encoder.encode(data)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            
    def _hybrid_encode(self, data: np.ndarray) -> np.ndarray:
        """Combine multiple encoding methods."""
        # Get encodings from each method
        rate_spikes = self.rate_encoder.encode(data)
        temporal_spikes = self.temporal_encoder.encode(data)
        phase_spikes = self.phase_encoder.encode(data)
        
        # Weighted combination
        combined_spikes = (
            rate_spikes * self.encoding_weights["rate"] +
            temporal_spikes * self.encoding_weights["temporal"] +
            phase_spikes * self.encoding_weights["phase"]
        )
        
        # Binarize the result
        threshold = 0.5
        binary_spikes = (combined_spikes > threshold).astype(np.float32)
        
        return binary_spikes
        
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """Decode using the current encoding method."""
        if self.encoding_type == "hybrid":
            # Use weighted decoding
            rate_decoded = self.rate_encoder.decode(spikes)
            temporal_decoded = self.temporal_encoder.decode(spikes)
            phase_decoded = self.phase_encoder.decode(spikes)
            
            combined = (
                rate_decoded * self.encoding_weights["rate"] +
                temporal_decoded * self.encoding_weights["temporal"] +
                phase_decoded * self.encoding_weights["phase"]
            )
            
            return combined
        elif self.encoding_type == "rate":
            return self.rate_encoder.decode(spikes)
        elif self.encoding_type == "temporal":
            return self.temporal_encoder.decode(spikes)
        elif self.encoding_type == "phase":
            return self.phase_encoder.decode(spikes)
            
    def adapt(
        self,
        input_distribution: np.ndarray,
        target_sparsity: Optional[float] = None,
        epochs: int = 50
    ) -> None:
        """Adapt encoding parameters based on input distribution."""
        if target_sparsity is not None:
            self.target_sparsity = target_sparsity
            
        print(f"Adapting encoder for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Sample from input distribution
            batch_idx = np.random.choice(len(input_distribution), size=32, replace=True)
            batch_data = input_distribution[batch_idx]
            
            # Encode and measure sparsity
            spikes = self.encode(batch_data)
            current_sparsity = np.mean(spikes)
            
            # Decode and measure reconstruction error
            reconstructed = self.decode(spikes)
            reconstruction_error = np.mean((batch_data - reconstructed) ** 2)
            
            # Adapt weights based on sparsity and accuracy
            sparsity_error = current_sparsity - self.target_sparsity
            
            if self.encoding_type == "hybrid":
                self._adapt_hybrid_weights(sparsity_error, reconstruction_error)
                
            # Track performance
            self.sparsity_history.append(current_sparsity)
            self.accuracy_history.append(1.0 / (1.0 + reconstruction_error))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Sparsity={current_sparsity:.4f}, "
                      f"Error={reconstruction_error:.4f}")
                
        print("Adaptation complete!")
        print(f"Final encoding weights: {self.encoding_weights}")
        
    def _adapt_hybrid_weights(self, sparsity_error: float, reconstruction_error: float) -> None:
        """Adapt hybrid encoding weights."""
        # Adjust weights based on performance
        weight_adjustment = self.adaptation_rate * sparsity_error
        
        # Increase rate coding if sparsity too high
        if sparsity_error > 0:
            self.encoding_weights["rate"] += weight_adjustment
            self.encoding_weights["temporal"] -= weight_adjustment / 2
            self.encoding_weights["phase"] -= weight_adjustment / 2
        else:
            self.encoding_weights["temporal"] += weight_adjustment / 2
            self.encoding_weights["phase"] += weight_adjustment / 2
            self.encoding_weights["rate"] -= weight_adjustment
            
        # Normalize weights
        total_weight = sum(self.encoding_weights.values())
        for key in self.encoding_weights:
            self.encoding_weights[key] /= total_weight
            
        # Ensure non-negative weights
        for key in self.encoding_weights:
            self.encoding_weights[key] = max(0.01, self.encoding_weights[key])
            
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get encoding performance statistics."""
        return {
            "encoding_type": self.encoding_type,
            "time_steps": self.time_steps,
            "target_sparsity": self.target_sparsity,
            "current_weights": self.encoding_weights.copy(),
            "avg_sparsity": np.mean(self.sparsity_history) if self.sparsity_history else 0.0,
            "avg_accuracy": np.mean(self.accuracy_history) if self.accuracy_history else 0.0,
            "adaptation_epochs": len(self.sparsity_history)
        }
        
    def save_encoder_state(self, filepath: str) -> None:
        """Save encoder state to file."""
        state = {
            "encoding_type": self.encoding_type,
            "time_steps": self.time_steps,
            "adaptation_rate": self.adaptation_rate,
            "target_sparsity": self.target_sparsity,
            "encoding_weights": self.encoding_weights,
            "sparsity_history": self.sparsity_history,
            "accuracy_history": self.accuracy_history
        }
        
        np.save(filepath, state)
        
    def load_encoder_state(self, filepath: str) -> None:
        """Load encoder state from file."""
        state = np.load(filepath, allow_pickle=True).item()
        
        self.encoding_type = state["encoding_type"]
        self.time_steps = state["time_steps"] 
        self.adaptation_rate = state["adaptation_rate"]
        self.target_sparsity = state["target_sparsity"]
        self.encoding_weights = state["encoding_weights"]
        self.sparsity_history = state["sparsity_history"]
        self.accuracy_history = state["accuracy_history"]