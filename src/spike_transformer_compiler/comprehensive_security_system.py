"""Comprehensive Security System: Enterprise-grade security and compliance.

This module implements comprehensive security measures including input validation,
secure compilation, threat detection, and compliance monitoring for the
Spike-Transformer-Compiler system.
"""

import time
import json
import hashlib
import hmac
import secrets
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import threading
import re
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    MALICIOUS_CODE_DETECTED = "malicious_code_detected"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION_ATTEMPT = "data_exfiltration_attempt"
    INJECTION_ATTACK = "injection_attack"
    DENIAL_OF_SERVICE = "denial_of_service"
    TAMPERING_DETECTED = "tampering_detected"


@dataclass
class SecurityIncident:
    """Represents a security incident."""
    incident_id: str
    timestamp: float
    event_type: SecurityEvent
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    affected_component: Optional[str] = None
    mitigation_applied: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """Comprehensive input validation with security focus."""
    
    def __init__(self):
        self.validation_rules = {
            "model_name": r"^[a-zA-Z0-9_\-\.]{1,100}$",
            "file_path": r"^[a-zA-Z0-9_\-\.\/\\]{1,500}$",
            "numeric_range": {"min": -1e10, "max": 1e10},
            "string_length": {"max": 10000},
            "array_size": {"max": 1000000}
        }
        
        self.dangerous_patterns = [
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"subprocess",
            r"os\.system",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"\.\.[\/\\]",  # Path traversal
            r"<script",      # XSS
            r"javascript:",  # JavaScript injection
            r"data:.*base64", # Data URI attacks
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
        
    def validate_model_input(self, model_data: Any) -> Tuple[bool, List[str]]:
        """Validate model input data for security threats."""
        errors = []
        
        try:
            # Check for basic data types
            if not self._validate_data_structure(model_data, errors):
                return False, errors
            
            # Check for malicious patterns in string data
            if not self._scan_for_malicious_patterns(model_data, errors):
                return False, errors
            
            # Validate data sizes
            if not self._validate_data_sizes(model_data, errors):
                return False, errors
            
            # Check for suspicious metadata
            if not self._validate_metadata(model_data, errors):
                return False, errors
            
            return True, []
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def validate_file_path(self, file_path: str) -> Tuple[bool, str]:
        """Validate file path for security issues."""
        if not file_path:
            return False, "Empty file path"
        
        # Check for path traversal
        if ".." in file_path or "~" in file_path:
            return False, "Path traversal detected"
        
        # Check for absolute paths that might be dangerous
        dangerous_paths = ["/etc", "/bin", "/usr/bin", "/sys", "/proc", "C:\\Windows", "C:\\System"]
        for dangerous_path in dangerous_paths:
            if file_path.startswith(dangerous_path):
                return False, f"Access to restricted path: {dangerous_path}"
        
        # Check against regex
        if not re.match(self.validation_rules["file_path"], file_path):
            return False, "Invalid file path format"
        
        return True, ""
    
    def validate_compilation_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate compilation configuration for security."""
        errors = []
        
        # Check for required security settings
        if not config.get("secure_mode", False):
            errors.append("Secure mode not enabled")
        
        # Validate target platform
        allowed_targets = ["loihi3", "simulation", "test"]
        target = config.get("target", "")
        if target not in allowed_targets:
            errors.append(f"Invalid target platform: {target}")
        
        # Check optimization level bounds
        opt_level = config.get("optimization_level", 0)
        if not isinstance(opt_level, int) or opt_level < 0 or opt_level > 3:
            errors.append(f"Invalid optimization level: {opt_level}")
        
        # Validate memory limits
        memory_limit = config.get("memory_limit", 0)
        if memory_limit > 0 and memory_limit < 100:  # Less than 100MB is suspicious
            errors.append(f"Suspiciously low memory limit: {memory_limit}MB")
        
        return len(errors) == 0, errors
    
    def _validate_data_structure(self, data: Any, errors: List[str]) -> bool:
        """Validate basic data structure."""
        try:
            # Check for excessively deep nesting
            max_depth = 10
            if self._get_nesting_depth(data) > max_depth:
                errors.append(f"Data structure too deeply nested (max: {max_depth})")
                return False
            
            # Check for circular references
            if self._has_circular_reference(data):
                errors.append("Circular reference detected in data structure")
                return False
            
            return True
            
        except Exception as e:
            errors.append(f"Data structure validation error: {str(e)}")
            return False
    
    def _scan_for_malicious_patterns(self, data: Any, errors: List[str]) -> bool:
        """Scan data for malicious patterns."""
        try:
            strings_to_check = self._extract_strings(data)
            
            for string_data in strings_to_check:
                for pattern in self.compiled_patterns:
                    if pattern.search(string_data):
                        errors.append(f"Malicious pattern detected: {pattern.pattern}")
                        return False
            
            return True
            
        except Exception as e:
            errors.append(f"Pattern scanning error: {str(e)}")
            return False
    
    def _validate_data_sizes(self, data: Any, errors: List[str]) -> bool:
        """Validate data sizes to prevent DoS attacks."""
        try:
            # Check string lengths
            strings = self._extract_strings(data)
            for string_data in strings:
                if len(string_data) > self.validation_rules["string_length"]["max"]:
                    errors.append(f"String too long: {len(string_data)} characters")
                    return False
            
            # Check array sizes
            arrays = self._extract_arrays(data)
            for array_data in arrays:
                if len(array_data) > self.validation_rules["array_size"]["max"]:
                    errors.append(f"Array too large: {len(array_data)} elements")
                    return False
            
            return True
            
        except Exception as e:
            errors.append(f"Size validation error: {str(e)}")
            return False
    
    def _validate_metadata(self, data: Any, errors: List[str]) -> bool:
        """Validate metadata for suspicious content."""
        if isinstance(data, dict):
            # Check for suspicious keys
            suspicious_keys = ["__class__", "__module__", "__dict__", "__globals__", "func_code"]
            for key in data.keys():
                if isinstance(key, str) and key in suspicious_keys:
                    errors.append(f"Suspicious metadata key: {key}")
                    return False
        
        return True
    
    def _get_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of data structure."""
        if current_depth > 20:  # Prevent infinite recursion
            return current_depth
        
        if isinstance(data, dict):
            return max([self._get_nesting_depth(v, current_depth + 1) for v in data.values()] + [current_depth])
        elif isinstance(data, (list, tuple)):
            return max([self._get_nesting_depth(item, current_depth + 1) for item in data] + [current_depth])
        else:
            return current_depth
    
    def _has_circular_reference(self, data: Any, seen: set = None) -> bool:
        """Check for circular references."""
        if seen is None:
            seen = set()
        
        data_id = id(data)
        if data_id in seen:
            return True
        
        if isinstance(data, (dict, list, tuple)):
            seen.add(data_id)
            
            if isinstance(data, dict):
                for value in data.values():
                    if self._has_circular_reference(value, seen.copy()):
                        return True
            elif isinstance(data, (list, tuple)):
                for item in data:
                    if self._has_circular_reference(item, seen.copy()):
                        return True
        
        return False
    
    def _extract_strings(self, data: Any) -> List[str]:
        """Extract all strings from data structure."""
        strings = []
        
        if isinstance(data, str):
            strings.append(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str):
                    strings.append(key)
                strings.extend(self._extract_strings(value))
        elif isinstance(data, (list, tuple)):
            for item in data:
                strings.extend(self._extract_strings(item))
        
        return strings
    
    def _extract_arrays(self, data: Any) -> List[List]:
        """Extract all arrays from data structure."""
        arrays = []
        
        if isinstance(data, (list, tuple)):
            arrays.append(data)
            for item in data:
                arrays.extend(self._extract_arrays(item))
        elif isinstance(data, dict):
            for value in data.values():
                arrays.extend(self._extract_arrays(value))
        
        return arrays


class CryptographicManager:
    """Manages cryptographic operations for secure compilation."""
    
    def __init__(self):
        self.symmetric_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.symmetric_key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Hash context for integrity checks
        self.integrity_hashes = {}
        
    def encrypt_sensitive_data(self, data: bytes) -> Tuple[bytes, str]:
        """Encrypt sensitive data and return encrypted data with key ID."""
        encrypted_data = self.cipher_suite.encrypt(data)
        key_id = hashlib.sha256(self.symmetric_key).hexdigest()[:16]
        return encrypted_data, key_id
    
    def decrypt_sensitive_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt sensitive data using key ID verification."""
        # Verify key ID
        expected_key_id = hashlib.sha256(self.symmetric_key).hexdigest()[:16]
        if key_id != expected_key_id:
            raise ValueError("Invalid key ID for decryption")
        
        return self.cipher_suite.decrypt(encrypted_data)
    
    def sign_compilation_artifact(self, artifact_data: bytes) -> bytes:
        """Sign compilation artifact for integrity verification."""
        signature = self.private_key.sign(
            artifact_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_compilation_artifact(self, artifact_data: bytes, signature: bytes) -> bool:
        """Verify compilation artifact signature."""
        try:
            self.public_key.verify(
                signature,
                artifact_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def compute_integrity_hash(self, data: bytes, label: str) -> str:
        """Compute and store integrity hash for data."""
        hash_value = hashlib.sha256(data).hexdigest()
        self.integrity_hashes[label] = {
            "hash": hash_value,
            "timestamp": time.time()
        }
        return hash_value
    
    def verify_integrity(self, data: bytes, label: str) -> bool:
        """Verify data integrity against stored hash."""
        if label not in self.integrity_hashes:
            return False
        
        expected_hash = self.integrity_hashes[label]["hash"]
        actual_hash = hashlib.sha256(data).hexdigest()
        
        return hmac.compare_digest(expected_hash, actual_hash)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def secure_delete(self, data_reference: str) -> bool:
        """Securely delete sensitive data from memory."""
        try:
            # Remove from integrity hashes
            if data_reference in self.integrity_hashes:
                del self.integrity_hashes[data_reference]
            
            # In a real implementation, this would overwrite memory
            # For simulation, we'll just log the action
            logging.info(f"Securely deleted data reference: {data_reference}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to securely delete {data_reference}: {str(e)}")
            return False


class ThreatDetectionEngine:
    """Advanced threat detection and analysis engine."""
    
    def __init__(self):
        self.threat_signatures = {
            "code_injection": [
                r"__import__\s*\(",
                r"eval\s*\(",
                r"exec\s*\(",
                r"compile\s*\(",
            ],
            "file_manipulation": [
                r"open\s*\(",
                r"file\s*\(",
                r"with\s+open",
                r"os\.remove",
                r"os\.unlink",
                r"shutil\.",
            ],
            "network_access": [
                r"urllib",
                r"requests\.",
                r"socket\.",
                r"http\.",
            ],
            "process_manipulation": [
                r"subprocess",
                r"os\.system",
                r"os\.popen",
                r"os\.spawn",
            ]
        }
        
        self.threat_scores = defaultdict(float)
        self.detection_history = deque(maxlen=1000)
        self.anomaly_baseline = {}
        
    def analyze_compilation_request(self, request_data: Dict[str, Any]) -> Tuple[ThreatLevel, List[str]]:
        """Analyze compilation request for security threats."""
        threats = []
        threat_score = 0.0
        
        # Analyze source code patterns
        if "source_code" in request_data:
            source_threats, source_score = self._analyze_source_code(request_data["source_code"])
            threats.extend(source_threats)
            threat_score += source_score
        
        # Analyze configuration parameters
        if "config" in request_data:
            config_threats, config_score = self._analyze_configuration(request_data["config"])
            threats.extend(config_threats)
            threat_score += config_score
        
        # Analyze request metadata
        metadata_threats, metadata_score = self._analyze_request_metadata(request_data)
        threats.extend(metadata_threats)
        threat_score += metadata_score
        
        # Determine threat level
        if threat_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif threat_score >= 0.3:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW
        
        # Record detection
        self.detection_history.append({
            "timestamp": time.time(),
            "threat_level": threat_level,
            "threat_score": threat_score,
            "threats_detected": len(threats)
        })
        
        return threat_level, threats
    
    def detect_anomalous_behavior(self, behavior_metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Detect anomalous behavior using statistical analysis."""
        anomaly_score = 0.0
        anomalies_detected = 0
        
        for metric_name, current_value in behavior_metrics.items():
            if metric_name in self.anomaly_baseline:
                baseline = self.anomaly_baseline[metric_name]
                mean = baseline["mean"]
                std_dev = baseline["std_dev"]
                
                # Calculate z-score
                if std_dev > 0:
                    z_score = abs((current_value - mean) / std_dev)
                    
                    # Anomaly if z-score > 3 (3-sigma rule)
                    if z_score > 3:
                        anomaly_score += min(1.0, z_score / 10)  # Normalize to 0-1
                        anomalies_detected += 1
            else:
                # Initialize baseline for new metric
                self.anomaly_baseline[metric_name] = {
                    "mean": current_value,
                    "std_dev": 0.0,
                    "count": 1,
                    "sum": current_value,
                    "sum_sq": current_value ** 2
                }
        
        # Update baselines
        self._update_anomaly_baselines(behavior_metrics)
        
        # Determine if anomalous
        is_anomalous = anomaly_score > 0.5 or anomalies_detected >= 3
        
        return is_anomalous, anomaly_score
    
    def _analyze_source_code(self, source_code: str) -> Tuple[List[str], float]:
        """Analyze source code for malicious patterns."""
        threats = []
        threat_score = 0.0
        
        for threat_type, patterns in self.threat_signatures.items():
            for pattern in patterns:
                if re.search(pattern, source_code, re.IGNORECASE):
                    threats.append(f"{threat_type}: {pattern}")
                    threat_score += 0.2  # Each detection adds to score
        
        # Check for obfuscation attempts
        if self._detect_obfuscation(source_code):
            threats.append("Code obfuscation detected")
            threat_score += 0.3
        
        return threats, min(1.0, threat_score)
    
    def _analyze_configuration(self, config: Dict[str, Any]) -> Tuple[List[str], float]:
        """Analyze configuration for security issues."""
        threats = []
        threat_score = 0.0
        
        # Check for suspicious settings
        if config.get("debug", False):
            threats.append("Debug mode enabled in production")
            threat_score += 0.1
        
        if config.get("allow_unsafe_operations", False):
            threats.append("Unsafe operations allowed")
            threat_score += 0.4
        
        # Check for excessive resource allocation
        memory_limit = config.get("memory_limit", 0)
        if memory_limit > 10000:  # >10GB
            threats.append(f"Excessive memory allocation: {memory_limit}MB")
            threat_score += 0.2
        
        return threats, min(1.0, threat_score)
    
    def _analyze_request_metadata(self, request_data: Dict[str, Any]) -> Tuple[List[str], float]:
        """Analyze request metadata for anomalies."""
        threats = []
        threat_score = 0.0
        
        # Check request size
        request_size = len(json.dumps(request_data, default=str))
        if request_size > 1000000:  # >1MB
            threats.append(f"Suspiciously large request: {request_size} bytes")
            threat_score += 0.2
        
        # Check for unusual patterns in request structure
        if self._detect_unusual_request_structure(request_data):
            threats.append("Unusual request structure detected")
            threat_score += 0.1
        
        return threats, min(1.0, threat_score)
    
    def _detect_obfuscation(self, code: str) -> bool:
        """Detect code obfuscation techniques."""
        # Check for excessive use of escape sequences
        escape_ratio = code.count("\\") / max(1, len(code))
        if escape_ratio > 0.1:
            return True
        
        # Check for suspicious character patterns
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]{5,}", code):
            return True
        
        # Check for base64-like patterns
        base64_pattern = r"[A-Za-z0-9+/]{20,}={0,2}"
        if len(re.findall(base64_pattern, code)) > 3:
            return True
        
        return False
    
    def _detect_unusual_request_structure(self, request_data: Dict[str, Any]) -> bool:
        """Detect unusual request structure patterns."""
        # Check for excessive nesting
        if self._get_max_nesting_depth(request_data) > 10:
            return True
        
        # Check for too many keys at top level
        if isinstance(request_data, dict) and len(request_data) > 50:
            return True
        
        return False
    
    def _get_max_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Get maximum nesting depth of data structure."""
        if current_depth > 15:  # Prevent stack overflow
            return current_depth
        
        if isinstance(data, dict):
            return max([self._get_max_nesting_depth(v, current_depth + 1) for v in data.values()] + [current_depth])
        elif isinstance(data, list):
            return max([self._get_max_nesting_depth(item, current_depth + 1) for item in data] + [current_depth])
        else:
            return current_depth
    
    def _update_anomaly_baselines(self, metrics: Dict[str, float]) -> None:
        """Update anomaly detection baselines with new data."""
        for metric_name, value in metrics.items():
            if metric_name in self.anomaly_baseline:
                baseline = self.anomaly_baseline[metric_name]
                
                # Update running statistics
                baseline["count"] += 1
                baseline["sum"] += value
                baseline["sum_sq"] += value ** 2
                
                # Recalculate mean and standard deviation
                n = baseline["count"]
                baseline["mean"] = baseline["sum"] / n
                
                if n > 1:
                    variance = (baseline["sum_sq"] - (baseline["sum"] ** 2) / n) / (n - 1)
                    baseline["std_dev"] = max(0, variance) ** 0.5


class SecurityMonitor:
    """Comprehensive security monitoring and incident response."""
    
    def __init__(self):
        self.security_incidents = deque(maxlen=10000)
        self.active_threats = {}
        self.security_metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "security_violations": 0,
            "incidents_resolved": 0
        }
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def log_security_incident(
        self,
        event_type: SecurityEvent,
        threat_level: ThreatLevel,
        description: str,
        **metadata
    ) -> str:
        """Log a security incident."""
        incident_id = secrets.token_urlsafe(16)
        
        incident = SecurityIncident(
            incident_id=incident_id,
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            metadata=metadata
        )
        
        self.security_incidents.append(incident)
        self.security_metrics["security_violations"] += 1
        
        # Auto-escalate critical threats
        if threat_level == ThreatLevel.CRITICAL:
            self._escalate_incident(incident)
        
        logging.warning(
            f"Security incident {incident_id}: {event_type.value} - {description}"
        )
        
        return incident_id
    
    def start_security_monitoring(self) -> None:
        """Start security monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            logging.info("Security monitoring started")
    
    def stop_security_monitoring(self) -> None:
        """Stop security monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            logging.info("Security monitoring stopped")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        recent_incidents = [inc for inc in self.security_incidents 
                          if time.time() - inc.timestamp < 3600]  # Last hour
        
        threat_distribution = defaultdict(int)
        for incident in recent_incidents:
            threat_distribution[incident.threat_level.value] += 1
        
        return {
            "security_metrics": self.security_metrics.copy(),
            "recent_incidents": len(recent_incidents),
            "active_threats": len(self.active_threats),
            "threat_distribution": dict(threat_distribution),
            "security_score": self._calculate_security_score(),
            "monitoring_status": "active" if self.monitoring_active else "inactive"
        }
    
    def _escalate_incident(self, incident: SecurityIncident) -> None:
        """Escalate critical security incident."""
        # Add to active threats
        self.active_threats[incident.incident_id] = incident
        
        # Log escalation
        logging.critical(
            f"SECURITY ESCALATION: {incident.event_type.value} - {incident.description}"
        )
        
        # In a real system, this would trigger alerts, notifications, etc.
    
    def _monitoring_loop(self) -> None:
        """Main security monitoring loop."""
        while self.monitoring_active and not self._stop_event.is_set():
            try:
                # Monitor for patterns in recent incidents
                self._analyze_incident_patterns()
                
                # Check for active threat resolution
                self._check_threat_resolution()
                
                # Update security metrics
                self._update_security_metrics()
                
            except Exception as e:
                logging.error(f"Error in security monitoring loop: {str(e)}")
            
            self._stop_event.wait(10)  # Check every 10 seconds
    
    def _analyze_incident_patterns(self) -> None:
        """Analyze patterns in recent security incidents."""
        recent_incidents = [inc for inc in self.security_incidents 
                          if time.time() - inc.timestamp < 300]  # Last 5 minutes
        
        # Check for incident clustering (potential coordinated attack)
        if len(recent_incidents) > 10:
            self.log_security_incident(
                SecurityEvent.DENIAL_OF_SERVICE,
                ThreatLevel.HIGH,
                f"High incident rate detected: {len(recent_incidents)} incidents in 5 minutes"
            )
    
    def _check_threat_resolution(self) -> None:
        """Check if active threats have been resolved."""
        resolved_threats = []
        
        for threat_id, incident in self.active_threats.items():
            # Auto-resolve threats older than 1 hour if no recent activity
            if time.time() - incident.timestamp > 3600:
                incident.resolved = True
                resolved_threats.append(threat_id)
                self.security_metrics["incidents_resolved"] += 1
        
        # Remove resolved threats
        for threat_id in resolved_threats:
            del self.active_threats[threat_id]
    
    def _update_security_metrics(self) -> None:
        """Update security metrics."""
        # Calculate security score based on recent activity
        self.security_metrics["security_score"] = self._calculate_security_score()
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-1, higher is better)."""
        total_requests = max(1, self.security_metrics["total_requests"])
        block_rate = self.security_metrics["blocked_requests"] / total_requests
        violation_rate = self.security_metrics["security_violations"] / total_requests
        
        # Good security = low violation rate but appropriate block rate
        security_score = 1.0 - violation_rate * 2  # Violations are bad
        security_score = max(0.0, min(1.0, security_score))
        
        return security_score


class ComprehensiveSecuritySystem:
    """Main security system orchestrator."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.crypto_manager = CryptographicManager()
        self.threat_detector = ThreatDetectionEngine()
        self.security_monitor = SecurityMonitor()
        self.security_enabled = True
        
    def secure_compilation_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process compilation request with full security validation."""
        
        if not self.security_enabled:
            return True, {"status": "security_disabled"}
        
        security_result = {
            "allowed": True,
            "threats_detected": [],
            "validation_errors": [],
            "security_actions": [],
            "request_id": self.crypto_manager.generate_secure_token(16)
        }
        
        try:
            self.security_monitor.security_metrics["total_requests"] += 1
            
            # Phase 1: Input validation
            model_valid, validation_errors = self.input_validator.validate_model_input(
                request_data.get("model_data", {})
            )
            
            if not model_valid:
                security_result["allowed"] = False
                security_result["validation_errors"] = validation_errors
                
                self.security_monitor.log_security_incident(
                    SecurityEvent.INPUT_VALIDATION_FAILURE,
                    ThreatLevel.MEDIUM,
                    f"Input validation failed: {', '.join(validation_errors)}"
                )
                
                return False, security_result
            
            # Phase 2: Threat detection
            threat_level, threats = self.threat_detector.analyze_compilation_request(request_data)
            
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                security_result["allowed"] = False
                security_result["threats_detected"] = threats
                
                self.security_monitor.log_security_incident(
                    SecurityEvent.MALICIOUS_CODE_DETECTED,
                    threat_level,
                    f"Threats detected: {', '.join(threats)}",
                    request_id=security_result["request_id"]
                )
                
                self.security_monitor.security_metrics["blocked_requests"] += 1
                return False, security_result
            
            elif threats:
                security_result["threats_detected"] = threats
                security_result["security_actions"].append("Enhanced monitoring enabled")
            
            # Phase 3: Configuration validation
            config = request_data.get("config", {})
            config_valid, config_errors = self.input_validator.validate_compilation_config(config)
            
            if not config_valid:
                security_result["validation_errors"].extend(config_errors)
                
                # For config errors, we might allow with warnings rather than block
                if any("secure_mode" in error for error in config_errors):
                    security_result["allowed"] = False
                    return False, security_result
            
            # Phase 4: Generate security context for compilation
            security_context = self._generate_security_context(request_data, user_context)
            security_result["security_context"] = security_context
            
            return True, security_result
            
        except Exception as e:
            logging.error(f"Security validation error: {str(e)}")
            
            self.security_monitor.log_security_incident(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                ThreatLevel.HIGH,
                f"Security system error: {str(e)}"
            )
            
            return False, {"allowed": False, "error": "Security validation failed"}
    
    def secure_compilation_artifact(
        self,
        artifact_data: bytes,
        metadata: Dict[str, Any]
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Secure compilation artifact with encryption and signing."""
        try:
            # Compute integrity hash
            integrity_hash = self.crypto_manager.compute_integrity_hash(
                artifact_data, metadata.get("artifact_id", "unknown")
            )
            
            # Sign artifact
            signature = self.crypto_manager.sign_compilation_artifact(artifact_data)
            
            # Encrypt if required
            if metadata.get("encrypt", False):
                encrypted_data, key_id = self.crypto_manager.encrypt_sensitive_data(artifact_data)
                artifact_data = encrypted_data
                metadata["encrypted"] = True
                metadata["key_id"] = key_id
            
            # Add security metadata
            security_metadata = {
                "integrity_hash": integrity_hash,
                "signature": base64.b64encode(signature).decode(),
                "security_version": "1.0",
                "timestamp": time.time()
            }
            metadata.update(security_metadata)
            
            return artifact_data, metadata
            
        except Exception as e:
            logging.error(f"Artifact security error: {str(e)}")
            raise
    
    def verify_compilation_artifact(
        self,
        artifact_data: bytes,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, bytes]:
        """Verify compilation artifact security."""
        try:
            # Decrypt if necessary
            if metadata.get("encrypted", False):
                key_id = metadata.get("key_id")
                if not key_id:
                    return False, b""
                
                artifact_data = self.crypto_manager.decrypt_sensitive_data(artifact_data, key_id)
            
            # Verify signature
            if "signature" in metadata:
                signature = base64.b64decode(metadata["signature"])
                if not self.crypto_manager.verify_compilation_artifact(artifact_data, signature):
                    self.security_monitor.log_security_incident(
                        SecurityEvent.TAMPERING_DETECTED,
                        ThreatLevel.CRITICAL,
                        "Artifact signature verification failed"
                    )
                    return False, b""
            
            # Verify integrity
            if "integrity_hash" in metadata:
                artifact_id = metadata.get("artifact_id", "unknown")
                if not self.crypto_manager.verify_integrity(artifact_data, artifact_id):
                    self.security_monitor.log_security_incident(
                        SecurityEvent.TAMPERING_DETECTED,
                        ThreatLevel.CRITICAL,
                        "Artifact integrity verification failed"
                    )
                    return False, b""
            
            return True, artifact_data
            
        except Exception as e:
            logging.error(f"Artifact verification error: {str(e)}")
            return False, b""
    
    def _generate_security_context(self, request_data: Dict, user_context: Dict = None) -> Dict[str, Any]:
        """Generate security context for compilation."""
        return {
            "security_level": "high",
            "sandbox_enabled": True,
            "resource_limits": {
                "max_memory_mb": 4096,
                "max_execution_time_sec": 300,
                "max_file_operations": 100
            },
            "allowed_operations": [
                "compilation",
                "optimization",
                "validation"
            ],
            "blocked_operations": [
                "file_system_access",
                "network_access",
                "process_creation"
            ],
            "monitoring_level": "enhanced",
            "user_permissions": user_context.get("permissions", []) if user_context else []
        }
    
    def enable_security(self) -> None:
        """Enable security system."""
        self.security_enabled = True
        self.security_monitor.start_security_monitoring()
        logging.info("Security system enabled")
    
    def disable_security(self) -> None:
        """Disable security system (for testing only)."""
        self.security_enabled = False
        self.security_monitor.stop_security_monitoring()
        logging.warning("Security system disabled")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "security_enabled": self.security_enabled,
            "security_summary": self.security_monitor.get_security_summary(),
            "threat_detection_active": True,
            "encryption_available": True,
            "monitoring_active": self.security_monitor.monitoring_active
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize security system
    security_system = ComprehensiveSecuritySystem()
    security_system.enable_security()
    
    print("🔒 COMPREHENSIVE SECURITY SYSTEM TESTING")
    
    # Test secure compilation request
    test_request = {
        "model_data": {
            "model_type": "spikeformer",
            "parameters": {"layers": 12, "hidden_size": 768}
        },
        "config": {
            "target": "loihi3",
            "optimization_level": 2,
            "secure_mode": True
        }
    }
    
    allowed, result = security_system.secure_compilation_request(test_request)
    print(f"   ✅ Secure request allowed: {allowed}")
    if result.get("threats_detected"):
        print(f"   ⚠️  Threats detected: {result['threats_detected']}")
    
    # Test malicious request
    malicious_request = {
        "model_data": {
            "model_type": "spikeformer",
            "source_code": "import os; os.system('rm -rf /')"
        },
        "config": {
            "target": "loihi3",
            "allow_unsafe_operations": True
        }
    }
    
    allowed, result = security_system.secure_compilation_request(malicious_request)
    print(f"   ❌ Malicious request blocked: {not allowed}")
    print(f"   🚨 Threats: {result.get('threats_detected', [])}")
    
    # Test artifact security
    test_artifact = b"compiled_model_data_here"
    test_metadata = {"artifact_id": "test_model_001", "encrypt": True}
    
    secured_artifact, secured_metadata = security_system.secure_compilation_artifact(
        test_artifact, test_metadata
    )
    print(f"   🔐 Artifact secured: {len(secured_artifact)} bytes")
    
    # Verify artifact
    verified, decrypted_artifact = security_system.verify_compilation_artifact(
        secured_artifact, secured_metadata
    )
    print(f"   ✅ Artifact verified: {verified}")
    print(f"   🔓 Decrypted matches original: {decrypted_artifact == test_artifact}")
    
    # Get security status
    status = security_system.get_security_status()
    print(f"\n📊 SECURITY STATUS:")
    print(f"   Security Score: {status['security_summary']['security_score']:.3f}")
    print(f"   Total Requests: {status['security_summary']['security_metrics']['total_requests']}")
    print(f"   Blocked Requests: {status['security_summary']['security_metrics']['blocked_requests']}")
    print(f"   Active Threats: {status['security_summary']['active_threats']}")
    
    # Cleanup
    security_system.disable_security()
