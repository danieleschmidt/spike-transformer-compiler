#!/usr/bin/env python3
"""
Advanced Security Scanner for Progressive Quality Gates
Comprehensive security validation with pattern detection and vulnerability assessment.
"""

import os
import re
import ast
import json
import hashlib
import subprocess
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

from .exceptions import ValidationError


@dataclass
class SecurityFinding:
    """Security finding information."""
    finding_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str  # injection, xss, auth, crypto, etc.
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None
    confidence: float = 0.0  # 0.0-1.0


@dataclass
class SecurityReport:
    """Complete security scan report."""
    scan_id: str
    timestamp: datetime
    total_files_scanned: int
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    risk_score: float
    compliance_status: Dict[str, bool]


class SecurityPatternMatcher:
    """Pattern-based security vulnerability detection."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_scanner.patterns")
        self.patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security vulnerability patterns."""
        return {
            "sql_injection": [
                {
                    "pattern": r"execute\s*\(\s*[\"'].*%.*[\"']\s*%",
                    "severity": "HIGH",
                    "cwe": "CWE-89",
                    "description": "Potential SQL injection via string formatting",
                    "recommendation": "Use parameterized queries or prepared statements"
                },
                {
                    "pattern": r"cursor\.execute\s*\(\s*[\"'].*\+.*[\"']",
                    "severity": "HIGH", 
                    "cwe": "CWE-89",
                    "description": "SQL injection via string concatenation",
                    "recommendation": "Use parameterized queries"
                },
                {
                    "pattern": r"query\s*=.*\.format\s*\(",
                    "severity": "MEDIUM",
                    "cwe": "CWE-89", 
                    "description": "Potential SQL injection via string format",
                    "recommendation": "Use parameterized queries"
                }
            ],
            "command_injection": [
                {
                    "pattern": r"os\.system\s*\(\s*.*\+",
                    "severity": "CRITICAL",
                    "cwe": "CWE-78",
                    "description": "Command injection via os.system with user input",
                    "recommendation": "Use subprocess with shell=False and validate inputs"
                },
                {
                    "pattern": r"subprocess\.(call|run|Popen).*shell\s*=\s*True",
                    "severity": "HIGH",
                    "cwe": "CWE-78",
                    "description": "Potential command injection with shell=True",
                    "recommendation": "Use shell=False and validate inputs"
                },
                {
                    "pattern": r"eval\s*\(",
                    "severity": "CRITICAL",
                    "cwe": "CWE-94",
                    "description": "Code injection via eval()",
                    "recommendation": "Avoid eval() or use ast.literal_eval() for safe evaluation"
                },
                {
                    "pattern": r"exec\s*\(",
                    "severity": "CRITICAL",
                    "cwe": "CWE-94",
                    "description": "Code injection via exec()",
                    "recommendation": "Avoid exec() or validate/sanitize inputs thoroughly"
                }
            ],
            "path_traversal": [
                {
                    "pattern": r"open\s*\(\s*.*\+.*[\"']\.\.\/[\"']",
                    "severity": "HIGH",
                    "cwe": "CWE-22",
                    "description": "Path traversal vulnerability",
                    "recommendation": "Validate and sanitize file paths, use os.path.join()"
                },
                {
                    "pattern": r"file\s*=.*input\(\)",
                    "severity": "MEDIUM",
                    "cwe": "CWE-22",
                    "description": "File access with user input",
                    "recommendation": "Validate file paths and restrict access to safe directories"
                }
            ],
            "weak_crypto": [
                {
                    "pattern": r"hashlib\.md5\s*\(",
                    "severity": "MEDIUM",
                    "cwe": "CWE-327",
                    "description": "Use of weak cryptographic hash MD5",
                    "recommendation": "Use SHA-256 or stronger hash functions"
                },
                {
                    "pattern": r"hashlib\.sha1\s*\(",
                    "severity": "MEDIUM",
                    "cwe": "CWE-327",
                    "description": "Use of weak cryptographic hash SHA-1",
                    "recommendation": "Use SHA-256 or stronger hash functions"
                },
                {
                    "pattern": r"random\.random\s*\(",
                    "severity": "LOW",
                    "cwe": "CWE-338",
                    "description": "Use of weak random number generator",
                    "recommendation": "Use secrets module for cryptographic operations"
                }
            ],
            "hardcoded_secrets": [
                {
                    "pattern": r"password\s*=\s*[\"'][^\"']{3,}[\"']",
                    "severity": "HIGH",
                    "cwe": "CWE-798",
                    "description": "Hardcoded password",
                    "recommendation": "Use environment variables or secure credential storage"
                },
                {
                    "pattern": r"api_key\s*=\s*[\"'][^\"']{10,}[\"']",
                    "severity": "HIGH",
                    "cwe": "CWE-798",
                    "description": "Hardcoded API key",
                    "recommendation": "Use environment variables or secure credential storage"
                },
                {
                    "pattern": r"secret\s*=\s*[\"'][^\"']{5,}[\"']",
                    "severity": "HIGH",
                    "cwe": "CWE-798",
                    "description": "Hardcoded secret",
                    "recommendation": "Use environment variables or secure credential storage"
                }
            ],
            "insecure_transport": [
                {
                    "pattern": r"http://[^\"'\s]+",
                    "severity": "MEDIUM",
                    "cwe": "CWE-319",
                    "description": "Unencrypted HTTP communication",
                    "recommendation": "Use HTTPS for secure communication"
                },
                {
                    "pattern": r"ssl_verify\s*=\s*False",
                    "severity": "HIGH",
                    "cwe": "CWE-295",
                    "description": "SSL verification disabled",
                    "recommendation": "Enable SSL verification for security"
                },
                {
                    "pattern": r"verify\s*=\s*False",
                    "severity": "HIGH", 
                    "cwe": "CWE-295",
                    "description": "Certificate verification disabled",
                    "recommendation": "Enable certificate verification"
                }
            ],
            "input_validation": [
                {
                    "pattern": r"request\.(args|form|json)\[[\"'][^\"']+[\"']\]",
                    "severity": "MEDIUM",
                    "cwe": "CWE-20",
                    "description": "Direct use of user input without validation",
                    "recommendation": "Validate and sanitize all user inputs"
                },
                {
                    "pattern": r"input\(\).*without.*validation",
                    "severity": "LOW",
                    "cwe": "CWE-20",
                    "description": "User input without validation",
                    "recommendation": "Implement input validation and sanitization"
                }
            ]
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security vulnerabilities."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
                
                for category, patterns in self.patterns.items():
                    for pattern_info in patterns:
                        pattern = pattern_info["pattern"]
                        
                        for line_num, line in enumerate(lines, 1):
                            matches = re.finditer(pattern, line, re.IGNORECASE)
                            
                            for match in matches:
                                finding = SecurityFinding(
                                    finding_id=f"{category}_{file_path.name}_{line_num}_{match.start()}",
                                    severity=pattern_info["severity"],
                                    category=category,
                                    title=pattern_info["description"],
                                    description=f"Security issue found: {pattern_info['description']}",
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    code_snippet=line.strip(),
                                    recommendation=pattern_info["recommendation"],
                                    cwe_id=pattern_info.get("cwe"),
                                    confidence=0.8  # Pattern-based confidence
                                )
                                findings.append(finding)
        
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
        
        return findings


class ASTSecurityAnalyzer:
    """AST-based security analysis for deeper code inspection."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_scanner.ast")
        
    def analyze_file(self, file_path: Path) -> List[SecurityFinding]:
        """Analyze file using AST parsing."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Analyze different AST node types
            findings.extend(self._analyze_function_calls(tree, file_path, content))
            findings.extend(self._analyze_imports(tree, file_path))
            findings.extend(self._analyze_assignments(tree, file_path, content))
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path} with AST: {e}")
        
        return findings
    
    def _analyze_function_calls(self, tree: ast.AST, file_path: Path, content: str) -> List[SecurityFinding]:
        """Analyze function calls for security issues."""
        findings = []
        lines = content.splitlines()
        
        class FunctionCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name in ['eval', 'exec']:
                        finding = SecurityFinding(
                            finding_id=f"ast_call_{file_path.name}_{node.lineno}_{func_name}",
                            severity="CRITICAL",
                            category="code_injection",
                            title=f"Dangerous function call: {func_name}",
                            description=f"Use of potentially dangerous function {func_name}()",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                            recommendation=f"Avoid using {func_name}() or ensure proper input validation",
                            cwe_id="CWE-94",
                            confidence=0.9
                        )
                        findings.append(finding)
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for os.system, subprocess.call with shell=True, etc.
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'os' and node.func.attr == 'system'):
                        
                        finding = SecurityFinding(
                            finding_id=f"ast_os_system_{file_path.name}_{node.lineno}",
                            severity="HIGH",
                            category="command_injection",
                            title="Use of os.system()",
                            description="Use of os.system() can lead to command injection",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                            recommendation="Use subprocess with proper argument validation",
                            cwe_id="CWE-78",
                            confidence=0.9
                        )
                        findings.append(finding)
                
                self.generic_visit(node)
        
        visitor = FunctionCallVisitor()
        visitor.visit(tree)
        
        return findings
    
    def _analyze_imports(self, tree: ast.AST, file_path: Path) -> List[SecurityFinding]:
        """Analyze imports for security issues."""
        findings = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name in ['pickle', 'cPickle']:
                        finding = SecurityFinding(
                            finding_id=f"ast_import_{file_path.name}_{node.lineno}_{alias.name}",
                            severity="MEDIUM",
                            category="deserialization",
                            title=f"Potentially unsafe import: {alias.name}",
                            description=f"Use of {alias.name} can be unsafe with untrusted data",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            code_snippet=f"import {alias.name}",
                            recommendation="Be cautious with pickle on untrusted data, consider safer alternatives",
                            cwe_id="CWE-502",
                            confidence=0.6
                        )
                        findings.append(finding)
                
                self.generic_visit(node)
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        return findings
    
    def _analyze_assignments(self, tree: ast.AST, file_path: Path, content: str) -> List[SecurityFinding]:
        """Analyze variable assignments for hardcoded secrets."""
        findings = []
        lines = content.splitlines()
        
        class AssignmentVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                if (isinstance(node.value, ast.Str) and 
                    len(node.value.s) > 10 and 
                    any(isinstance(target, ast.Name) and 
                        any(keyword in target.id.lower() for keyword in ['password', 'secret', 'key', 'token'])
                        for target in node.targets if isinstance(target, ast.Name))):
                    
                    var_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else "unknown"
                    
                    finding = SecurityFinding(
                        finding_id=f"ast_secret_{file_path.name}_{node.lineno}_{var_name}",
                        severity="HIGH",
                        category="hardcoded_secrets",
                        title=f"Potential hardcoded secret: {var_name}",
                        description=f"Variable {var_name} appears to contain a hardcoded secret",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        recommendation="Use environment variables or secure credential storage",
                        cwe_id="CWE-798",
                        confidence=0.7
                    )
                    findings.append(finding)
                
                self.generic_visit(node)
        
        visitor = AssignmentVisitor()
        visitor.visit(tree)
        
        return findings


class DependencySecurityScanner:
    """Scanner for known vulnerabilities in dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_scanner.dependencies")
        
    def scan_dependencies(self, project_root: Path) -> List[SecurityFinding]:
        """Scan project dependencies for known vulnerabilities."""
        findings = []
        
        # Check requirements.txt
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            findings.extend(self._scan_requirements_file(requirements_file))
        
        # Check pyproject.toml
        pyproject_file = project_root / "pyproject.toml"
        if pyproject_file.exists():
            findings.extend(self._scan_pyproject_file(pyproject_file))
        
        return findings
    
    def _scan_requirements_file(self, requirements_file: Path) -> List[SecurityFinding]:
        """Scan requirements.txt for vulnerable packages."""
        findings = []
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.readlines()
            
            # Simple check for potentially problematic packages
            vulnerable_patterns = {
                r'pickle': "Pickle can be unsafe with untrusted data",
                r'exec': "Packages with 'exec' in name may be suspicious",
                r'==.*\d\.\d\.\d': "Pinned versions may have known vulnerabilities"
            }
            
            for line_num, line in enumerate(requirements, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    for pattern, description in vulnerable_patterns.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            finding = SecurityFinding(
                                finding_id=f"dep_req_{requirements_file.name}_{line_num}",
                                severity="LOW",
                                category="dependency",
                                title="Potentially problematic dependency",
                                description=description,
                                file_path=str(requirements_file),
                                line_number=line_num,
                                code_snippet=line,
                                recommendation="Review dependency for security issues",
                                confidence=0.3
                            )
                            findings.append(finding)
                            break
        
        except Exception as e:
            self.logger.error(f"Error scanning requirements file: {e}")
        
        return findings
    
    def _scan_pyproject_file(self, pyproject_file: Path) -> List[SecurityFinding]:
        """Scan pyproject.toml for vulnerable packages."""
        findings = []
        
        try:
            import toml
            with open(pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
            
            # Check dependencies section
            dependencies = pyproject_data.get('project', {}).get('dependencies', [])
            
            for dep in dependencies:
                if any(keyword in dep.lower() for keyword in ['eval', 'exec', 'pickle']):
                    finding = SecurityFinding(
                        finding_id=f"dep_pyproject_{pyproject_file.name}_{hash(dep)}",
                        severity="MEDIUM",
                        category="dependency",
                        title="Potentially unsafe dependency",
                        description=f"Dependency {dep} may have security implications",
                        file_path=str(pyproject_file),
                        line_number=1,  # Can't determine exact line from TOML parsing
                        code_snippet=dep,
                        recommendation="Review dependency for security issues",
                        confidence=0.4
                    )
                    findings.append(finding)
        
        except ImportError:
            self.logger.warning("toml package not available for pyproject.toml scanning")
        except Exception as e:
            self.logger.error(f"Error scanning pyproject file: {e}")
        
        return findings


class ComprehensiveSecurityScanner:
    """Comprehensive security scanner integrating all analysis methods."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger("security_scanner")
        
        # Initialize sub-scanners
        self.pattern_matcher = SecurityPatternMatcher()
        self.ast_analyzer = ASTSecurityAnalyzer()
        self.dependency_scanner = DependencySecurityScanner()
        
        # Configuration
        self.exclude_patterns = [
            "*.pyc", "__pycache__", ".git", ".pytest_cache",
            "node_modules", "venv", "env", ".venv"
        ]
        
    def scan_project(self) -> SecurityReport:
        """Perform comprehensive security scan of the project."""
        self.logger.info(f"Starting security scan of {self.project_root}")
        
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        all_findings = []
        files_scanned = 0
        
        # Scan Python source files
        for py_file in self._get_python_files():
            self.logger.debug(f"Scanning {py_file}")
            
            # Pattern-based scanning
            pattern_findings = self.pattern_matcher.scan_file(py_file)
            all_findings.extend(pattern_findings)
            
            # AST-based scanning
            ast_findings = self.ast_analyzer.analyze_file(py_file)
            all_findings.extend(ast_findings)
            
            files_scanned += 1
        
        # Scan dependencies
        dependency_findings = self.dependency_scanner.scan_dependencies(self.project_root)
        all_findings.extend(dependency_findings)
        
        # Remove duplicates and calculate risk score
        unique_findings = self._deduplicate_findings(all_findings)
        risk_score = self._calculate_risk_score(unique_findings)
        
        # Generate summary
        summary = self._generate_summary(unique_findings)
        
        # Check compliance
        compliance_status = self._check_compliance(unique_findings)
        
        report = SecurityReport(
            scan_id=scan_id,
            timestamp=start_time,
            total_files_scanned=files_scanned,
            findings=unique_findings,
            summary=summary,
            risk_score=risk_score,
            compliance_status=compliance_status
        )
        
        self.logger.info(
            f"Security scan completed: {len(unique_findings)} findings, "
            f"risk score: {risk_score:.1f}"
        )
        
        return report
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files to scan."""
        python_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                re.match(pattern.replace('*', '.*'), d) 
                for pattern in self.exclude_patterns
            )]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
        
        return python_files
    
    def _deduplicate_findings(self, findings: List[SecurityFinding]) -> List[SecurityFinding]:
        """Remove duplicate findings."""
        seen = set()
        unique_findings = []
        
        for finding in findings:
            # Create a hash based on key attributes
            key = (finding.category, finding.file_path, finding.line_number, finding.title)
            key_hash = hashlib.md5(str(key).encode()).hexdigest()
            
            if key_hash not in seen:
                seen.add(key_hash)
                unique_findings.append(finding)
        
        return unique_findings
    
    def _calculate_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall risk score (0-100)."""
        if not findings:
            return 0.0
        
        severity_weights = {
            "CRITICAL": 10.0,
            "HIGH": 7.0,
            "MEDIUM": 4.0,
            "LOW": 1.0
        }
        
        total_score = sum(
            severity_weights.get(finding.severity, 1.0) * finding.confidence
            for finding in findings
        )
        
        # Normalize to 0-100 scale
        max_possible_score = len(findings) * 10.0
        risk_score = min(100.0, (total_score / max_possible_score) * 100.0) if max_possible_score > 0 else 0.0
        
        return risk_score
    
    def _generate_summary(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Generate finding summary by category and severity."""
        summary = {
            "total": len(findings),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for finding in findings:
            severity_key = finding.severity.lower()
            if severity_key in summary:
                summary[severity_key] += 1
        
        # Category breakdown
        categories = defaultdict(int)
        for finding in findings:
            categories[finding.category] += 1
        
        summary["categories"] = dict(categories)
        
        return summary
    
    def _check_compliance(self, findings: List[SecurityFinding]) -> Dict[str, bool]:
        """Check compliance with security standards."""
        compliance = {
            "owasp_top_10": True,
            "cwe_sans_25": True,
            "secure_coding": True,
            "data_protection": True
        }
        
        # Check for specific vulnerability types
        critical_categories = {
            "sql_injection", "command_injection", "code_injection",
            "hardcoded_secrets", "insecure_transport"
        }
        
        high_severity_findings = [f for f in findings if f.severity in ["CRITICAL", "HIGH"]]
        
        if high_severity_findings:
            compliance["owasp_top_10"] = False
            compliance["secure_coding"] = False
        
        if any(f.category in critical_categories for f in findings):
            compliance["cwe_sans_25"] = False
        
        if any(f.category == "hardcoded_secrets" for f in findings):
            compliance["data_protection"] = False
        
        return compliance
    
    def export_report(self, report: SecurityReport, output_path: Path, format: str = "json"):
        """Export security report to file."""
        if format == "json":
            self._export_json_report(report, output_path)
        elif format == "html":
            self._export_html_report(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json_report(self, report: SecurityReport, output_path: Path):
        """Export report as JSON."""
        report_data = asdict(report)
        
        # Convert datetime to string for JSON serialization
        report_data["timestamp"] = report.timestamp.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Security report exported to {output_path}")
    
    def _export_html_report(self, report: SecurityReport, output_path: Path):
        """Export report as HTML."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .finding { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .critical { border-left: 5px solid #d32f2f; }
        .high { border-left: 5px solid #f57c00; }
        .medium { border-left: 5px solid #fbc02d; }
        .low { border-left: 5px solid #388e3c; }
        .code { background-color: #f5f5f5; padding: 5px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p><strong>Scan ID:</strong> {scan_id}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Files Scanned:</strong> {files_scanned}</p>
        <p><strong>Risk Score:</strong> {risk_score:.1f}/100</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Findings:</strong> {total_findings}</p>
        <p><strong>Critical:</strong> {critical} | <strong>High:</strong> {high} | 
           <strong>Medium:</strong> {medium} | <strong>Low:</strong> {low}</p>
    </div>
    
    <div class="findings">
        <h2>Findings</h2>
        {findings_html}
    </div>
</body>
</html>
"""
        
        findings_html = ""
        for finding in report.findings:
            finding_html = f"""
        <div class="finding {finding.severity.lower()}">
            <h3>{finding.title}</h3>
            <p><strong>Severity:</strong> {finding.severity}</p>
            <p><strong>Category:</strong> {finding.category}</p>
            <p><strong>File:</strong> {finding.file_path}:{finding.line_number}</p>
            <p><strong>Description:</strong> {finding.description}</p>
            <div class="code">{finding.code_snippet}</div>
            <p><strong>Recommendation:</strong> {finding.recommendation}</p>
            {f'<p><strong>CWE:</strong> {finding.cwe_id}</p>' if finding.cwe_id else ''}
        </div>
"""
            findings_html += finding_html
        
        html_content = html_template.format(
            scan_id=report.scan_id,
            timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            files_scanned=report.total_files_scanned,
            risk_score=report.risk_score,
            total_findings=report.summary["total"],
            critical=report.summary["critical"],
            high=report.summary["high"],
            medium=report.summary["medium"],
            low=report.summary["low"],
            findings_html=findings_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML security report exported to {output_path}")


def main():
    """Run comprehensive security scan."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔒 COMPREHENSIVE SECURITY SCANNER")
    print("=" * 50)
    
    scanner = ComprehensiveSecurityScanner()
    
    # Run security scan
    report = scanner.scan_project()
    
    # Display results
    print(f"\n📊 Security Scan Results:")
    print(f"Scan ID: {report.scan_id}")
    print(f"Files Scanned: {report.total_files_scanned}")
    print(f"Total Findings: {report.summary['total']}")
    print(f"Risk Score: {report.risk_score:.1f}/100")
    
    print(f"\nFindings by Severity:")
    print(f"  Critical: {report.summary['critical']}")
    print(f"  High: {report.summary['high']}")
    print(f"  Medium: {report.summary['medium']}")
    print(f"  Low: {report.summary['low']}")
    
    if report.findings:
        print(f"\nTop 5 Findings:")
        for i, finding in enumerate(report.findings[:5], 1):
            print(f"  {i}. [{finding.severity}] {finding.title}")
            print(f"     File: {finding.file_path}:{finding.line_number}")
    
    print(f"\nCompliance Status:")
    for standard, status in report.compliance_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {standard.replace('_', ' ').title()}")
    
    # Export reports
    scanner.export_report(report, Path("security_report.json"), "json")
    scanner.export_report(report, Path("security_report.html"), "html")
    
    print(f"\n📄 Reports exported to security_report.json and security_report.html")
    
    return 0 if report.risk_score < 30 else 1  # Exit code based on risk


if __name__ == "__main__":
    exit(main())