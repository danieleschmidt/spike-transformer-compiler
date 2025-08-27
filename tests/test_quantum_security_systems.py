"""Comprehensive test suite for quantum security and performance systems."""

import pytest
import numpy as np
import asyncio
import time
import threading
import hashlib
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('../src')

from spike_transformer_compiler.quantum_secure_compilation import (
    SecureCompilationOrchestrator, QuantumResistantCrypto, AdvancedThreatDetector,
    SecurityContext, ThreatIntelligence, ThreatLevel, SecurityEvent
)
from spike_transformer_compiler.adaptive_resilience_advanced import (
    AdvancedResilienceOrchestrator, CircuitBreaker, BulkheadIsolation,
    CheckpointManager, FailureEvent, FailureType, RecoveryStrategy
)
from spike_transformer_compiler.quantum_neuromorphic_acceleration import (
    QuantumNeuromorphicProcessor, QuantumSpikingNeuralNetwork, QuantumGate,
    QuantumState, QuantumCircuit
)


class TestQuantumResistantCrypto:
    """Test quantum-resistant cryptographic operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.crypto = QuantumResistantCrypto()
    
    def test_initialization(self):
        """Test crypto system initialization."""
        assert self.crypto.key_size == 32
        assert self.crypto.nonce_size == 24
        assert len(self.crypto.hash_algorithms) >= 2
        assert 'dimension' in self.crypto.lattice_parameters
    
    def test_generate_quantum_resistant_key(self):
        """Test quantum-resistant key generation."""
        key1 = self.crypto.generate_quantum_resistant_key()
        key2 = self.crypto.generate_quantum_resistant_key()
        
        assert len(key1) == self.crypto.key_size
        assert len(key2) == self.crypto.key_size
        assert key1 != key2  # Keys should be different
        assert isinstance(key1, bytes)
    
    def test_derive_key(self):
        """Test key derivation."""
        master_key = self.crypto.generate_quantum_resistant_key()
        context = "test_context"
        
        derived_key1 = self.crypto.derive_key(master_key, context)
        derived_key2 = self.crypto.derive_key(master_key, context)
        
        assert len(derived_key1) == self.crypto.key_size
        assert derived_key1 == derived_key2  # Same context should give same key
        
        # Different context should give different key
        derived_key3 = self.crypto.derive_key(master_key, "different_context")
        assert derived_key1 != derived_key3
    
    def test_encrypt_decrypt_model_weights(self):
        """Test model weight encryption and decryption."""
        weights = np.random.randn(10, 5).astype(np.float32)
        key = self.crypto.generate_quantum_resistant_key()
        
        encrypted_data, nonce = self.crypto.encrypt_model_weights(weights, key)
        decrypted_weights = self.crypto.decrypt_model_weights(
            encrypted_data, key, nonce, weights.shape, 'float32'
        )
        
        assert isinstance(encrypted_data, bytes)
        assert isinstance(nonce, bytes)
        assert len(nonce) == self.crypto.nonce_size
        np.testing.assert_array_almost_equal(weights, decrypted_weights)
    
    def test_encrypt_decrypt_integrity_check(self):
        """Test encryption integrity checking."""
        weights = np.random.randn(5, 3).astype(np.float32)
        key = self.crypto.generate_quantum_resistant_key()
        
        encrypted_data, nonce = self.crypto.encrypt_model_weights(weights, key)
        
        # Tamper with encrypted data
        tampered_data = encrypted_data[:-1] + b'X'
        
        with pytest.raises(ValueError, match="integrity check failed"):
            self.crypto.decrypt_model_weights(
                tampered_data, key, nonce, weights.shape, 'float32'
            )
    
    def test_secure_hash(self):
        """Test secure hash function."""
        data = b"test_data"
        context = "test_context"
        
        hash1 = self.crypto.secure_hash(data, context)
        hash2 = self.crypto.secure_hash(data, context)
        
        assert hash1 == hash2  # Same input should give same hash
        assert len(hash1) == 64  # SHA3-256 hex output
        
        # Different data should give different hash
        hash3 = self.crypto.secure_hash(b"different_data", context)
        assert hash1 != hash3
    
    def test_expand_key(self):
        """Test key expansion functionality."""
        key = self.crypto.generate_quantum_resistant_key()
        expanded = self.crypto._expand_key(key, 100)
        
        assert len(expanded) == 100
        assert isinstance(expanded, bytes)
        
        # Same key and length should give same expansion
        expanded2 = self.crypto._expand_key(key, 100)
        assert expanded == expanded2


class TestAdvancedThreatDetector:
    """Test advanced threat detection system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.detector = AdvancedThreatDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert len(self.detector.evaluation_metrics) > 0
        assert len(self.detector.threat_patterns) > 0
        assert 'model_poisoning' in self.detector.threat_patterns
        assert 'backdoor_injection' in self.detector.threat_patterns
    
    def test_analyze_model_security(self):
        """Test comprehensive model security analysis."""
        model_data = {
            'weights': {
                'layer1': np.random.randn(100, 50),
                'layer2': np.random.randn(50, 10)
            },
            'architecture': {
                'layers': [
                    {'size': 100, 'type': 'input'},
                    {'size': 50, 'type': 'hidden'},
                    {'size': 10, 'type': 'output'}
                ]
            }
        }
        
        context = SecurityContext(user_id="test_user")
        threats = self.detector.analyze_model_security(model_data, context)
        
        assert isinstance(threats, list)
        for threat in threats:
            assert isinstance(threat, ThreatIntelligence)
            assert threat.threat_type in SecurityEvent
            assert threat.severity in ThreatLevel
    
    def test_detect_weight_anomalies(self):
        """Test weight anomaly detection."""
        # Create model with anomalous weights
        model_data = {
            'weights': {
                'normal_layer': np.random.randn(50, 30),
                'anomalous_layer': np.ones((30, 10)) * 1000  # Extreme values
            }
        }
        
        threats = self.detector._detect_weight_anomalies(model_data)
        
        assert len(threats) > 0
        extreme_weight_threats = [t for t in threats if 'extreme_weights' in t.threat_id]
        assert len(extreme_weight_threats) > 0
    
    def test_detect_architectural_anomalies(self):
        """Test architectural anomaly detection."""
        # Create model with architectural anomalies
        model_data = {
            'architecture': {
                'layers': [{'size': 200000} for _ in range(1500)]  # Too many large layers
            }
        }
        
        threats = self.detector._detect_architectural_anomalies(model_data)
        
        assert len(threats) > 0
        resource_threats = [t for t in threats 
                           if t.threat_type == SecurityEvent.RESOURCE_EXHAUSTION]
        assert len(resource_threats) > 0
    
    def test_calculate_skewness_kurtosis(self):
        """Test statistical calculations."""
        # Test with normal distribution
        normal_data = np.random.normal(0, 1, 1000)
        skewness = self.detector._calculate_skewness(normal_data)
        kurtosis = self.detector._calculate_kurtosis(normal_data)
        
        assert abs(skewness) < 0.5  # Should be close to 0 for normal distribution
        assert abs(kurtosis) < 1.0  # Should be close to 0 for normal distribution
        
        # Test with skewed distribution
        skewed_data = np.random.exponential(2, 1000)
        skewed_skewness = self.detector._calculate_skewness(skewed_data)
        
        assert skewed_skewness > 0.5  # Exponential is right-skewed


class TestSecureCompilationOrchestrator:
    """Test secure compilation orchestrator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = SecureCompilationOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.crypto is not None
        assert self.orchestrator.threat_detector is not None
        assert len(self.orchestrator.active_sessions) == 0
        assert len(self.orchestrator.security_policies) > 0
    
    def test_create_secure_session(self):
        """Test secure session creation."""
        context = self.orchestrator.create_secure_session("test_user", "standard")
        
        assert isinstance(context, SecurityContext)
        assert context.user_id == "test_user"
        assert context.access_level == "standard"
        assert len(context.quantum_keys) > 0
        assert context.session_id in self.orchestrator.active_sessions
    
    def test_session_limits(self):
        """Test session creation limits."""
        # Create multiple sessions for same user
        sessions = []
        for i in range(3):  # Max 3 sessions per user
            session = self.orchestrator.create_secure_session(f"user_{i}", "standard")
            sessions.append(session)
        
        # Should fail when creating too many sessions for same user
        with pytest.raises(ValueError, match="Maximum sessions exceeded"):
            for i in range(4):
                self.orchestrator.create_secure_session("same_user", "standard")
    
    def test_secure_compile(self):
        """Test secure compilation process."""
        context = self.orchestrator.create_secure_session("test_user", "standard")
        
        model_data = {
            'architecture': {
                'layers': [
                    {'type': 'input', 'size': 10},
                    {'type': 'hidden', 'size': 20},
                    {'type': 'output', 'size': 5}
                ]
            },
            'weights': {
                'layer1': np.random.randn(10, 20) * 0.1,
                'layer2': np.random.randn(20, 5) * 0.1
            }
        }
        
        compilation_options = {'optimization_level': 2, 'target': 'simulation'}
        
        result = self.orchestrator.secure_compile(
            model_data, context, compilation_options
        )
        
        assert isinstance(result, dict)
        assert 'compiled_model' in result or 'encrypted_compilation_result' in result
    
    def test_sanitize_model_input(self):
        """Test model input sanitization."""
        context = SecurityContext()
        model_data = {
            'safe_field': 'safe_value',
            '__dict__': 'dangerous',
            'eval': 'also_dangerous',
            'nested': {
                'exec': 'nested_dangerous',
                'safe': 'nested_safe'
            }
        }
        
        sanitized = self.orchestrator._sanitize_model_input(model_data, context)
        
        assert 'safe_field' in sanitized
        assert '__dict__' not in sanitized
        assert 'eval' not in sanitized
        assert 'nested' in sanitized
        assert 'exec' not in sanitized['nested']
        assert 'safe' in sanitized['nested']
    
    def test_encrypt_decrypt_compilation_result(self):
        """Test compilation result encryption and decryption."""
        context = self.orchestrator.create_secure_session("test_user", "standard")
        
        result = {
            'compiled_model': {'test': 'data'},
            'metadata': {'size': 1000}
        }
        
        encrypted_result = self.orchestrator._encrypt_compilation_result(result, context)
        decrypted_result = self.orchestrator.decrypt_compilation_result(encrypted_result, context)
        
        assert encrypted_result != result
        assert 'encrypted_compilation_result' in encrypted_result
        assert 'integrity_hash' in encrypted_result
        assert decrypted_result == result
    
    def test_session_expiration(self):
        """Test session expiration handling."""
        context = SecurityContext()
        context.creation_time = time.time() - 7200  # 2 hours ago
        
        assert context.is_expired(max_age=3600)  # 1 hour limit
        assert not context.is_expired(max_age=10800)  # 3 hour limit
    
    def test_get_security_report(self):
        """Test security report generation."""
        context = self.orchestrator.create_secure_session("test_user", "standard")
        
        # Session-specific report
        session_report = self.orchestrator.get_security_report(context.session_id)
        assert 'session_id' in session_report
        assert 'user_id' in session_report
        assert 'security_events' in session_report
        
        # Global report
        global_report = self.orchestrator.get_security_report()
        assert 'summary' in global_report
        assert 'system_status' in global_report


class TestCircuitBreaker:
    """Test circuit breaker implementation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    def test_initialization(self):
        """Test circuit breaker initialization."""
        assert self.circuit_breaker.failure_threshold == 3
        assert self.circuit_breaker.recovery_timeout == 1.0
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0
    
    def test_successful_execution(self):
        """Test successful function execution."""
        @self.circuit_breaker
        def success_function():
            return "success"
        
        result = success_function()
        assert result == "success"
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0
    
    def test_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        @self.circuit_breaker
        def failing_function():
            raise Exception("Test failure")
        
        # Cause failures to exceed threshold
        for i in range(3):
            with pytest.raises(Exception):
                failing_function()
        
        assert self.circuit_breaker.state == "OPEN"
        assert self.circuit_breaker.failure_count == 3
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker behavior in OPEN state."""
        # Force circuit breaker to OPEN state
        self.circuit_breaker.state = "OPEN"
        self.circuit_breaker.last_failure_time = time.time()
        
        @self.circuit_breaker
        def any_function():
            return "should not execute"
        
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            any_function()
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        # Force to OPEN state with old failure time
        self.circuit_breaker.state = "OPEN"
        self.circuit_breaker.last_failure_time = time.time() - 2.0  # 2 seconds ago
        
        @self.circuit_breaker
        def recovery_function():
            return "recovered"
        
        result = recovery_function()
        assert result == "recovered"
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0


class TestBulkheadIsolation:
    """Test bulkhead isolation pattern."""
    
    def setup_method(self):
        """Set up test environment."""
        self.bulkhead = BulkheadIsolation(max_concurrent=3)
    
    def test_initialization(self):
        """Test bulkhead initialization."""
        assert self.bulkhead.max_concurrent == 3
        assert self.bulkhead.active_tasks == 0
    
    def test_successful_execution_within_limit(self):
        """Test successful execution within concurrency limit."""
        def test_function():
            return "success"
        
        result = self.bulkhead.execute(test_function)
        assert result == "success"
    
    def test_concurrency_limit_enforcement(self):
        """Test enforcement of concurrency limits."""
        # Mock semaphore to simulate full capacity
        self.bulkhead.semaphore = Mock()
        self.bulkhead.semaphore.acquire.return_value = False
        
        def test_function():
            return "should not execute"
        
        with pytest.raises(Exception, match="Bulkhead limit exceeded"):
            self.bulkhead.execute(test_function)
    
    def test_active_task_counting(self):
        """Test active task counting."""
        initial_count = self.bulkhead.active_tasks
        
        def counting_function():
            # Task count should be incremented during execution
            assert self.bulkhead.active_tasks == initial_count + 1
            return "counted"
        
        result = self.bulkhead.execute(counting_function)
        assert result == "counted"
        assert self.bulkhead.active_tasks == initial_count


class TestCheckpointManager:
    """Test checkpoint management system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.checkpoint_manager = CheckpointManager()
    
    def test_initialization(self):
        """Test checkpoint manager initialization."""
        assert self.checkpoint_manager.checkpoint_dir.exists()
        assert len(self.checkpoint_manager.checkpoints) == 0
    
    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        component_id = "test_component"
        state = {"data": [1, 2, 3], "config": {"param": "value"}}
        metadata = {"version": "1.0"}
        
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            component_id, state, metadata
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id in self.checkpoint_manager.checkpoints
        assert len(checkpoint_id) == 16  # SHA256 prefix
        
        checkpoint_info = self.checkpoint_manager.checkpoints[checkpoint_id]
        assert checkpoint_info['component_id'] == component_id
        assert checkpoint_info['metadata'] == metadata
    
    def test_restore_checkpoint(self):
        """Test checkpoint restoration."""
        component_id = "test_component"
        original_state = {"restore_test": True, "values": [4, 5, 6]}
        
        # Create checkpoint
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            component_id, original_state
        )
        
        # Restore checkpoint
        restored_data = self.checkpoint_manager.restore_checkpoint(checkpoint_id)
        
        assert restored_data['component_id'] == component_id
        assert restored_data['state'] == original_state
        assert 'timestamp' in restored_data
    
    def test_list_checkpoints(self):
        """Test checkpoint listing."""
        # Create multiple checkpoints
        for i in range(3):
            self.checkpoint_manager.create_checkpoint(
                f"component_{i}", {"data": i}
            )
        
        all_checkpoints = self.checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) == 3
        
        # Test filtering by component
        component_checkpoints = self.checkpoint_manager.list_checkpoints("component_1")
        assert len(component_checkpoints) == 1
        assert component_checkpoints[0]['component_id'] == "component_1"
    
    def test_checkpoint_not_found(self):
        """Test handling of non-existent checkpoints."""
        with pytest.raises(ValueError, match="Checkpoint not found"):
            self.checkpoint_manager.restore_checkpoint("nonexistent_id")


class TestQuantumNeuromorphicProcessor:
    """Test quantum neuromorphic processor."""
    
    def setup_method(self):
        """Set up test environment."""
        self.processor = QuantumNeuromorphicProcessor(num_qubits=4)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.num_qubits == 4
        assert len(self.processor.gate_implementations) > 0
        assert QuantumGate.HADAMARD in self.processor.gate_implementations
        assert len(self.processor.quantum_memory) == 0
    
    def test_create_quantum_state(self):
        """Test quantum state creation."""
        state_id = "test_state"
        state = self.processor.create_quantum_state(state_id)
        
        assert isinstance(state, QuantumState)
        assert state.num_qubits == 4
        assert len(state.amplitudes) == 2**4  # 16 amplitudes
        assert state_id in self.processor.quantum_memory
        
        # Initial state should be |0000⟩
        assert abs(state.amplitudes[0] - 1.0) < 1e-10
        assert all(abs(amp) < 1e-10 for amp in state.amplitudes[1:])
    
    def test_hadamard_gate(self):
        """Test Hadamard gate implementation."""
        state_id = "hadamard_test"
        state = self.processor.create_quantum_state(state_id)
        
        # Apply Hadamard to first qubit
        initial_amplitudes = state.amplitudes.copy()
        self.processor._hadamard_gate(state, [0], {})
        
        # Should create superposition
        assert abs(state.amplitudes[0] - 1/np.sqrt(2)) < 1e-10
        assert abs(state.amplitudes[1] - 1/np.sqrt(2)) < 1e-10
    
    def test_pauli_x_gate(self):
        """Test Pauli-X gate implementation."""
        state_id = "pauli_x_test"
        state = self.processor.create_quantum_state(state_id)
        
        # Apply Pauli-X to first qubit (should flip |0⟩ to |1⟩)
        self.processor._pauli_x_gate(state, [0], {})
        
        # Check that state is now |0001⟩ (qubit 0 flipped)
        assert abs(state.amplitudes[1] - 1.0) < 1e-10  # |0001⟩ state
        assert abs(state.amplitudes[0]) < 1e-10  # |0000⟩ should be zero
    
    def test_cnot_gate(self):
        """Test CNOT gate implementation."""
        state_id = "cnot_test"
        state = self.processor.create_quantum_state(state_id)
        
        # First put control qubit in superposition
        self.processor._hadamard_gate(state, [0], {})
        
        # Apply CNOT with qubit 0 as control, qubit 1 as target
        self.processor._cnot_gate(state, [0, 1], {})
        
        # Should create entangled state |00⟩ + |11⟩
        expected_amplitude = 1/np.sqrt(2)
        assert abs(state.amplitudes[0] - expected_amplitude) < 1e-10  # |0000⟩
        assert abs(state.amplitudes[3] - expected_amplitude) < 1e-10  # |0011⟩
    
    def test_quantum_circuit_application(self):
        """Test applying quantum circuits."""
        circuit = QuantumCircuit(num_qubits=4)
        circuit.add_gate(QuantumGate.HADAMARD, [0])
        circuit.add_gate(QuantumGate.CNOT, [0, 1])
        circuit.add_gate(QuantumGate.PAULI_X, [2])
        
        state_id = "circuit_test"
        initial_state = self.processor.create_quantum_state(state_id)
        final_state = self.processor.apply_quantum_circuit(state_id, circuit)
        
        assert final_state is initial_state  # Should modify in place
        assert len(circuit.gates) == 3
    
    def test_measurement(self):
        """Test quantum state measurement."""
        state_id = "measurement_test"
        state = self.processor.create_quantum_state(state_id)
        
        # Put qubit in superposition
        self.processor._hadamard_gate(state, [0], {})
        
        # Measure all qubits multiple times
        measurements = []
        for _ in range(100):
            result = self.processor.measure_state(state_id)
            measurements.append(result)
        
        assert all(len(measurement) == 4 for measurement in measurements)
        assert all(all(bit in [0, 1] for bit in measurement) 
                  for measurement in measurements)


class TestQuantumSpikingNeuralNetwork:
    """Test quantum spiking neural network."""
    
    def setup_method(self):
        """Set up test environment."""
        self.qsnn = QuantumSpikingNeuralNetwork(num_neurons=5, num_qubits=8)
    
    def test_initialization(self):
        """Test QSNN initialization."""
        assert self.qsnn.num_neurons == 5
        assert self.qsnn.num_qubits == 8
        assert len(self.qsnn.neuron_qubit_mapping) > 0
    
    def test_encode_spike_train(self):
        """Test spike train encoding."""
        neuron_id = 0
        spike_times = np.array([10.0, 30.0, 50.0])
        
        state_id = self.qsnn.encode_spike_train(neuron_id, spike_times)
        
        assert isinstance(state_id, str)
        assert state_id in self.qsnn.quantum_processor.quantum_memory
        assert state_id in self.qsnn.spike_encoding_circuits
    
    def test_quantum_synaptic_transmission(self):
        """Test quantum synaptic transmission."""
        # First encode spike train
        pre_neuron = 0
        post_neuron = 1
        spike_times = np.array([10.0, 30.0])
        weight = 0.5
        
        self.qsnn.encode_spike_train(pre_neuron, spike_times)
        
        # Test synaptic transmission
        synaptic_current = self.qsnn.quantum_synaptic_transmission(
            pre_neuron, post_neuron, weight
        )
        
        assert isinstance(synaptic_current, float)
        assert synaptic_current >= 0  # Should be non-negative
    
    def test_quantum_forward_pass(self):
        """Test quantum neural network forward pass."""
        input_data = np.random.randn(16)
        architecture = {
            'layers': [
                {'type': 'input', 'size': 16},
                {'type': 'hidden', 'size': 8},
                {'type': 'output', 'size': 4}
            ]
        }
        
        output = self.qsnn.quantum_forward_pass(input_data, architecture)
        
        assert isinstance(output, np.ndarray)
        assert len(output) == self.qsnn.num_qubits
        assert all(0 <= val <= 1 for val in output)  # Measurements should be 0 or 1
    
    def test_create_quantum_neural_circuit(self):
        """Test quantum neural circuit creation."""
        architecture = {
            'layers': [
                {'type': 'dense', 'size': 4},
                {'type': 'conv', 'size': 3, 'kernel_size': 2},
                {'type': 'dense', 'size': 2}
            ]
        }
        
        circuit = self.qsnn.create_quantum_neural_circuit(architecture)
        
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == self.qsnn.num_qubits
        assert len(circuit.gates) > 0
    
    def test_get_quantum_metrics(self):
        """Test quantum performance metrics."""
        metrics = self.qsnn.get_quantum_metrics()
        
        assert 'quantum_speedup_factor' in metrics
        assert 'coherence_fidelity' in metrics
        assert 'entanglement_connectivity' in metrics
        assert 'quantum_efficiency' in metrics
        
        for value in metrics.values():
            assert isinstance(value, (int, float))
            assert value >= 0


# Integration tests
class TestSecuritySystemIntegration:
    """Integration tests for security systems."""
    
    def test_crypto_threat_detection_integration(self):
        """Test integration between crypto and threat detection."""
        crypto = QuantumResistantCrypto()
        detector = AdvancedThreatDetector()
        
        # Create model with encrypted weights
        original_weights = np.random.randn(50, 30)
        key = crypto.generate_quantum_resistant_key()
        encrypted_weights, nonce = crypto.encrypt_model_weights(original_weights, key)
        
        # Create model data for threat analysis
        model_data = {
            'weights': {
                'encrypted_layer': encrypted_weights
            },
            'architecture': {
                'layers': [{'size': 50}, {'size': 30}]
            }
        }
        
        context = SecurityContext()
        threats = detector.analyze_model_security(model_data, context)
        
        # Should not detect threats in properly encrypted data
        assert len(threats) == 0 or all(t.severity != ThreatLevel.CRITICAL for t in threats)
    
    def test_resilience_security_integration(self):
        """Test integration between resilience and security systems."""
        orchestrator = SecureCompilationOrchestrator()
        resilience = AdvancedResilienceOrchestrator()
        
        # Register security component with resilience
        resilience.register_component(
            "secure_compiler",
            circuit_breaker_config={'failure_threshold': 2},
            bulkhead_config={'max_concurrent': 2}
        )
        
        # Create secure session
        context = orchestrator.create_secure_session("integration_test", "standard")
        
        # Define secure compilation function
        def secure_compile_function():
            model_data = {'test': 'data'}
            return orchestrator.secure_compile(model_data, context)
        
        # Execute with resilience protection
        try:
            result = resilience.execute_with_resilience(
                "secure_compiler", "compile", secure_compile_function
            )
            assert result is not None
        except Exception as e:
            # Should handle failures gracefully
            assert "compilation" in str(e).lower() or "secure" in str(e).lower()


# Performance and stress tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests for security systems."""
    
    @pytest.mark.benchmark
    def test_crypto_encryption_performance(self, benchmark):
        """Benchmark cryptographic operations."""
        crypto = QuantumResistantCrypto()
        weights = np.random.randn(1000, 500).astype(np.float32)
        key = crypto.generate_quantum_resistant_key()
        
        def encrypt_operation():
            return crypto.encrypt_model_weights(weights, key)
        
        encrypted_data, nonce = benchmark(encrypt_operation)
        assert isinstance(encrypted_data, bytes)
    
    @pytest.mark.benchmark
    def test_threat_detection_performance(self, benchmark):
        """Benchmark threat detection performance."""
        detector = AdvancedThreatDetector()
        
        # Create large model for performance testing
        model_data = {
            'weights': {f'layer_{i}': np.random.randn(100, 100) for i in range(10)},
            'architecture': {
                'layers': [{'size': 100, 'type': 'dense'} for _ in range(10)]
            }
        }
        
        context = SecurityContext()
        
        threats = benchmark(detector.analyze_model_security, model_data, context)
        assert isinstance(threats, list)
    
    @pytest.mark.benchmark  
    def test_quantum_circuit_performance(self, benchmark):
        """Benchmark quantum circuit execution."""
        processor = QuantumNeuromorphicProcessor(num_qubits=10)
        
        # Create complex circuit
        circuit = QuantumCircuit(num_qubits=10)
        for i in range(10):
            circuit.add_gate(QuantumGate.HADAMARD, [i])
        for i in range(9):
            circuit.add_gate(QuantumGate.CNOT, [i, i+1])
        
        state_id = "benchmark_state"
        processor.create_quantum_state(state_id)
        
        final_state = benchmark(processor.apply_quantum_circuit, state_id, circuit)
        assert isinstance(final_state, QuantumState)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=spike_transformer_compiler.quantum_secure_compilation",
        "--cov=spike_transformer_compiler.adaptive_resilience_advanced", 
        "--cov=spike_transformer_compiler.quantum_neuromorphic_acceleration",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "-m", "not slow"
    ])