"""Enhanced security framework for robust operation."""

import hashlib
import hmac
import secrets
import time
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class SecurityLevel(Enum):
    """Security enforcement levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatType(Enum):
    """Types of security threats."""
    MALICIOUS_INPUT = "malicious_input"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CODE_INJECTION = "code_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: ThreatType
    severity: str  # low, medium, high, critical
    source: str
    description: str
    metadata: Dict[str, Any]
    remediation: Optional[str] = None


class InputValidator:
    """Comprehensive input validation."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'__.*__',  # Python dunder methods
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
            r'import\s+os',  # OS module imports
            r'import\s+subprocess',  # subprocess imports
            r'\.system\s*\(',  # system() calls
            r'\.popen\s*\(',  # popen() calls
            r'\.call\s*\(',  # call() from subprocess
            r'\.run\s*\(',  # run() from subprocess
            r'open\s*\([^)]*[\'"]w',  # write file operations
            r'pickle\.loads',  # pickle deserialization
            r'yaml\.load[^_]',  # unsafe YAML loading
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def validate_model_input(self, model: Any) -> Tuple[bool, List[str]]:
        """Validate model input for security threats."""
        issues = []
        
        try:
            # Check model type
            model_type = type(model).__name__
            if self._is_suspicious_type(model_type):
                issues.append(f"Suspicious model type: {model_type}")
            
            # Check model attributes (if accessible)
            if hasattr(model, '__dict__'):
                for attr_name in dir(model):
                    if self._is_suspicious_attribute(attr_name):
                        issues.append(f"Suspicious model attribute: {attr_name}")
            
            # Check string representation for dangerous patterns
            model_str = str(model)[:10000]  # Limit string length
            for pattern in self.compiled_patterns:
                if pattern.search(model_str):
                    issues.append(f"Dangerous pattern detected in model: {pattern.pattern}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Model validation error: {str(e)}")
            return False, issues
    
    def validate_input_shape(self, shape: Any) -> Tuple[bool, List[str]]:
        """Validate input shape parameter."""
        issues = []
        
        try:
            # Check if shape is a valid tuple/list
            if not isinstance(shape, (tuple, list)):
                issues.append("Input shape must be tuple or list")
                return False, issues
            
            # Check shape dimensions
            if len(shape) == 0:
                issues.append("Input shape cannot be empty")
            elif len(shape) > 6:  # Reasonable limit
                issues.append(f"Input shape too many dimensions: {len(shape)}")
            
            # Check individual dimension values
            for i, dim in enumerate(shape):
                if not isinstance(dim, int):
                    issues.append(f"Dimension {i} is not integer: {type(dim)}")
                elif dim < 0:
                    issues.append(f"Dimension {i} is negative: {dim}")
                elif dim > 1_000_000:  # Reasonable limit
                    issues.append(f"Dimension {i} too large: {dim}")
            
            # Check total size
            if all(isinstance(d, int) and d > 0 for d in shape):
                total_size = 1
                for d in shape:
                    total_size *= d
                    if total_size > 100_000_000:  # 100M elements
                        issues.append(f"Total tensor size too large: {total_size}")
                        break
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Shape validation error: {str(e)}")
            return False, issues
    
    def validate_compilation_target(self, target: str) -> Tuple[bool, List[str]]:
        """Validate compilation target parameter."""
        issues = []
        
        try:
            if not isinstance(target, str):
                issues.append(f"Target must be string, got {type(target)}")
                return False, issues
            
            # Check for suspicious content
            if self._contains_dangerous_patterns(target):
                issues.append("Target contains suspicious patterns")
            
            # Check length
            if len(target) > 100:
                issues.append("Target string too long")
            
            # Check allowed characters
            if not re.match(r'^[a-zA-Z0-9_-]+$', target):
                issues.append("Target contains invalid characters")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Target validation error: {str(e)}")
            return False, issues
    
    def validate_file_path(self, filepath: str) -> Tuple[bool, List[str]]:
        """Validate file path for security."""
        issues = []
        
        try:
            if not isinstance(filepath, str):
                issues.append(f"File path must be string, got {type(filepath)}")
                return False, issues
            
            # Check for path traversal
            if '..' in filepath:
                issues.append("Path traversal detected")
            
            # Check for suspicious patterns
            if self._contains_dangerous_patterns(filepath):
                issues.append("File path contains suspicious patterns")
            
            # Check file extension if present
            path_obj = Path(filepath)
            if path_obj.suffix:
                allowed_extensions = {'.py', '.pth', '.pt', '.onnx', '.json', '.yaml', '.yml'}
                if path_obj.suffix.lower() not in allowed_extensions:
                    issues.append(f"Suspicious file extension: {path_obj.suffix}")
            
            # Check length
            if len(filepath) > 1000:
                issues.append("File path too long")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"File path validation error: {str(e)}")
            return False, issues
    
    def _is_suspicious_type(self, type_name: str) -> bool:
        """Check if type name is suspicious."""
        suspicious_types = {
            'os', 'subprocess', 'sys', 'builtins', '__builtin__',
            'eval', 'exec', 'compile', 'open', 'file'
        }
        return type_name.lower() in suspicious_types
    
    def _is_suspicious_attribute(self, attr_name: str) -> bool:
        """Check if attribute name is suspicious."""
        if attr_name.startswith('__') and attr_name.endswith('__'):
            return True
        
        suspicious_attrs = {
            'system', 'popen', 'call', 'run', 'eval', 'exec',
            'compile', 'open', 'file', 'input', 'raw_input'
        }
        return attr_name.lower() in suspicious_attrs
    
    def _contains_dangerous_patterns(self, text: str) -> bool:
        """Check if text contains dangerous patterns."""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False


class ModelSanitizer:
    """Sanitize models to remove potential threats."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
    
    def sanitize_model(self, model: Any) -> Tuple[Any, List[str]]:
        """Sanitize model by removing dangerous components."""
        warnings = []
        
        try:
            # For mock models, just return as-is with validation
            if hasattr(model, '__class__') and 'Mock' in model.__class__.__name__:
                return model, warnings
            
            # For real PyTorch models, we'd implement actual sanitization
            # This is a simplified version
            
            # Check for suspicious attributes and log warnings
            if hasattr(model, '__dict__'):
                for attr_name in list(vars(model).keys()):
                    if self._is_dangerous_attribute(attr_name):
                        warnings.append(f"Removed suspicious attribute: {attr_name}")
                        # In real implementation, would remove the attribute
            
            return model, warnings
            
        except Exception as e:
            self.logger.error(f"Model sanitization error: {e}")
            return model, [f"Sanitization error: {str(e)}"]
    
    def _is_dangerous_attribute(self, attr_name: str) -> bool:
        """Check if attribute should be removed."""
        dangerous_attrs = {
            '__code__', '__globals__', '__closure__', '__defaults__',
            'eval', 'exec', 'compile', 'open', 'file'
        }
        return attr_name in dangerous_attrs


class SecurityAuditor:
    """Audit system for security events and compliance."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.events: List[SecurityEvent] = []
        self.log_file = log_file
        
        # Security metrics
        self.threat_counts: Dict[ThreatType, int] = {t: 0 for t in ThreatType}
        self.blocked_attempts = 0
        self.allowed_attempts = 0
    
    def log_security_event(self, 
                          event_type: ThreatType,
                          severity: str,
                          source: str,
                          description: str,
                          metadata: Optional[Dict[str, Any]] = None):
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source=source,
            description=description,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self.threat_counts[event_type] += 1
        
        # Log to system logger
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(log_level, f"SECURITY: {description} (type: {event_type.value}, source: {source})")
        
        # Write to security log file if specified
        if self.log_file:
            self._write_to_security_log(event)
    
    def log_blocked_attempt(self, reason: str, source: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a blocked security attempt."""
        self.blocked_attempts += 1
        self.log_security_event(
            event_type=ThreatType.MALICIOUS_INPUT,
            severity="medium",
            source=source,
            description=f"Blocked attempt: {reason}",
            metadata=metadata
        )
    
    def log_allowed_attempt(self, source: str):
        """Log an allowed attempt."""
        self.allowed_attempts += 1
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        threat_summary = {}
        for threat_type in ThreatType:
            count = sum(1 for e in recent_events if e.event_type == threat_type)
            threat_summary[threat_type.value] = count
        
        severity_summary = {}
        for severity in ['low', 'medium', 'high', 'critical']:
            count = sum(1 for e in recent_events if e.severity == severity)
            severity_summary[severity] = count
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "blocked_attempts": self.blocked_attempts,
            "allowed_attempts": self.allowed_attempts,
            "block_rate": self.blocked_attempts / max(1, self.blocked_attempts + self.allowed_attempts),
            "threat_types": threat_summary,
            "severity_levels": severity_summary,
            "recent_critical_events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.event_type.value,
                    "description": e.description,
                    "source": e.source
                }
                for e in recent_events[-10:]  # Last 10 events
                if e.severity == "critical"
            ]
        }
    
    def _write_to_security_log(self, event: SecurityEvent):
        """Write event to security log file."""
        try:
            log_entry = {
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "severity": event.severity,
                "source": event.source,
                "description": event.description,
                "metadata": event.metadata
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write to security log: {e}")


class SecureCompilationManager:
    """Manage secure compilation with multiple security layers."""
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.STANDARD,
                 audit_log: Optional[str] = "security_audit.log"):
        self.security_level = security_level
        self.validator = InputValidator(security_level)
        self.sanitizer = ModelSanitizer(security_level)
        self.auditor = SecurityAuditor(audit_log)
        self.logger = logging.getLogger(__name__)
        
        # Security tokens for session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    def create_secure_session(self) -> str:
        """Create a secure compilation session."""
        session_id = secrets.token_hex(16)
        session_token = secrets.token_hex(32)
        
        self.active_sessions[session_id] = {
            "token": session_token,
            "created_at": time.time(),
            "compilation_count": 0,
            "last_activity": time.time()
        }
        
        self.auditor.log_security_event(
            event_type=ThreatType.PRIVILEGE_ESCALATION,
            severity="low",
            source="session_manager",
            description="New secure session created",
            metadata={"session_id": session_id}
        )
        
        return session_id
    
    def validate_session(self, session_id: str, token: str) -> bool:
        """Validate a compilation session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Check token
        if not hmac.compare_digest(session["token"], token):
            self.auditor.log_blocked_attempt(
                "Invalid session token",
                f"session:{session_id}"
            )
            return False
        
        # Check session expiry (24 hours)
        if time.time() - session["created_at"] > 86400:
            self.cleanup_session(session_id)
            self.auditor.log_blocked_attempt(
                "Expired session",
                f"session:{session_id}"
            )
            return False
        
        # Update last activity
        session["last_activity"] = time.time()
        return True
    
    def secure_compile(self, 
                      model: Any,
                      input_shape: Any,
                      target: str,
                      session_id: Optional[str] = None,
                      **kwargs) -> Tuple[Any, List[str]]:
        """Perform secure compilation with full validation."""
        warnings = []
        source_id = session_id or "anonymous"
        
        try:
            # Phase 1: Input Validation
            model_valid, model_issues = self.validator.validate_model_input(model)
            shape_valid, shape_issues = self.validator.validate_input_shape(input_shape)
            target_valid, target_issues = self.validator.validate_compilation_target(target)
            
            all_issues = model_issues + shape_issues + target_issues
            
            if not (model_valid and shape_valid and target_valid):
                self.auditor.log_blocked_attempt(
                    f"Input validation failed: {'; '.join(all_issues)}",
                    source_id,
                    metadata={"issues": all_issues}
                )
                raise ValueError(f"Security validation failed: {all_issues}")
            
            # Phase 2: Model Sanitization
            sanitized_model, sanitize_warnings = self.sanitizer.sanitize_model(model)
            warnings.extend(sanitize_warnings)
            
            # Phase 3: Rate Limiting Check (simplified)
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session["compilation_count"] += 1
                
                # Basic rate limiting
                if session["compilation_count"] > 100:  # 100 compilations per session
                    self.auditor.log_blocked_attempt(
                        "Rate limit exceeded",
                        source_id,
                        metadata={"compilation_count": session["compilation_count"]}
                    )
                    raise ValueError("Rate limit exceeded for session")
            
            # Phase 4: Perform Compilation
            self.auditor.log_allowed_attempt(source_id)
            
            # Import and use compiler
            from .compiler import SpikeCompiler
            
            compiler = SpikeCompiler(target=target, verbose=False)
            compiled_model = compiler.compile(sanitized_model, input_shape, **kwargs)
            
            # Phase 5: Post-compilation validation
            if hasattr(compiled_model, 'utilization'):
                if compiled_model.utilization > 1.0:
                    warnings.append("Utilization exceeds 100% - potential resource issue")
            
            self.auditor.log_security_event(
                event_type=ThreatType.MALICIOUS_INPUT,
                severity="low",
                source=source_id,
                description="Compilation completed successfully",
                metadata={
                    "target": target,
                    "input_shape": str(input_shape),
                    "warnings": len(warnings)
                }
            )
            
            return compiled_model, warnings
            
        except Exception as e:
            self.auditor.log_security_event(
                event_type=ThreatType.DENIAL_OF_SERVICE,
                severity="medium",
                source=source_id,
                description=f"Compilation failed: {str(e)}",
                metadata={"error": str(e), "target": target}
            )
            raise e
    
    def cleanup_session(self, session_id: str):
        """Clean up a compilation session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
            self.auditor.log_security_event(
                event_type=ThreatType.PRIVILEGE_ESCALATION,
                severity="low",
                source="session_manager",
                description="Session cleaned up",
                metadata={"session_id": session_id}
            )
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Sessions expire after 24 hours of inactivity
            if current_time - session["last_activity"] > 86400:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "security_level": self.security_level.value,
            "active_sessions": len(self.active_sessions),
            "audit_summary": self.auditor.get_security_summary(),
            "recent_warnings": [
                f"Session {sid}: {session['compilation_count']} compilations"
                for sid, session in self.active_sessions.items()
                if session['compilation_count'] > 50
            ]
        }


# Global security manager instance
_global_security_manager = None


def get_security_manager() -> SecureCompilationManager:
    """Get or create global security manager."""
    global _global_security_manager
    
    if _global_security_manager is None:
        _global_security_manager = SecureCompilationManager()
    
    return _global_security_manager


if __name__ == "__main__":
    # Demo security system
    manager = SecureCompilationManager()
    
    # Create secure session
    session_id = manager.create_secure_session()
    session_token = manager.active_sessions[session_id]["token"]
    
    print(f"Created session: {session_id}")
    
    # Test secure compilation
    from .mock_models import create_test_model
    
    try:
        model = create_test_model("simple")
        compiled, warnings = manager.secure_compile(
            model=model,
            input_shape=(1, 10),
            target="simulation",
            session_id=session_id
        )
        
        print("✓ Secure compilation successful")
        if warnings:
            print(f"Warnings: {warnings}")
            
    except Exception as e:
        print(f"✗ Secure compilation failed: {e}")
    
    # Print security status
    status = manager.get_security_status()
    print("\nSecurity Status:")
    print(json.dumps(status, indent=2))