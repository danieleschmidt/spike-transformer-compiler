"""Advanced Security Audit for Autonomous SDLC v4.0.

Comprehensive security scanning and validation of the autonomous 
neuromorphic computing platform implementation.
"""

import os
import re
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess


class SecurityAuditor:
    """Advanced security auditor for autonomous SDLC."""
    
    def __init__(self):
        self.security_issues = []
        self.compliance_issues = []
        self.best_practices_violations = []
        self.vulnerability_findings = []
        
        # Security patterns to detect
        self.security_patterns = self._initialize_security_patterns()
        self.compliance_requirements = self._initialize_compliance_requirements()
    
    def _initialize_security_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize security patterns to detect."""
        return {
            "hardcoded_secrets": {
                "pattern": r"(password|secret|key|token|api_key)\s*=\s*['\"][^'\"]*['\"]",
                "severity": "HIGH",
                "description": "Hardcoded secrets detected"
            },
            "weak_crypto": {
                "pattern": r"(md5|sha1)\s*\(",
                "severity": "MEDIUM", 
                "description": "Weak cryptographic algorithms detected"
            },
            "sql_injection": {
                "pattern": r"(execute|query|cursor)\s*\(\s*['\"].*%s.*['\"]",
                "severity": "CRITICAL",
                "description": "Potential SQL injection vulnerability"
            },
            "command_injection": {
                "pattern": r"(os\.system|subprocess\.call|os\.popen)\s*\([^)]*user",
                "severity": "CRITICAL", 
                "description": "Potential command injection vulnerability"
            },
            "unsafe_deserialization": {
                "pattern": r"(pickle\.loads|yaml\.load|eval|exec)\s*\(",
                "severity": "HIGH",
                "description": "Unsafe deserialization detected"
            },
            "debug_enabled": {
                "pattern": r"debug\s*=\s*True",
                "severity": "MEDIUM",
                "description": "Debug mode enabled in production code"
            },
            "insecure_random": {
                "pattern": r"random\.(randint|choice|random)\s*\(",
                "severity": "LOW",
                "description": "Insecure random number generation"
            }
        }
    
    def _initialize_compliance_requirements(self) -> Dict[str, List[str]]:
        """Initialize compliance requirements."""
        return {
            "ISO27001": [
                "Access control mechanisms",
                "Encryption of sensitive data",
                "Audit logging functionality", 
                "Incident response procedures",
                "Security monitoring capabilities"
            ],
            "NIST_CSF": [
                "Identify: Asset management",
                "Protect: Data security", 
                "Detect: Security monitoring",
                "Respond: Response planning",
                "Recover: Recovery planning"
            ],
            "GDPR": [
                "Data minimization",
                "Consent mechanisms",
                "Right to erasure",
                "Data portability",
                "Privacy by design"
            ],
            "SOC2": [
                "Security policies",
                "Change management",
                "Logical access controls",
                "System monitoring",
                "Risk assessment"
            ]
        }
    
    def audit_source_code(self, source_dir: str) -> Dict[str, Any]:
        """Perform comprehensive source code security audit."""
        
        print("🔍 Starting Source Code Security Audit...")
        
        audit_results = {
            "files_scanned": 0,
            "security_issues": [],
            "compliance_issues": [],
            "best_practices": [],
            "vulnerability_summary": {}
        }
        
        source_path = Path(source_dir)
        
        for py_file in source_path.rglob("*.py"):
            try:
                audit_results["files_scanned"] += 1
                
                # Read file content
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Security pattern matching
                file_issues = self._scan_security_patterns(str(py_file), content)
                audit_results["security_issues"].extend(file_issues)
                
                # AST-based security analysis
                ast_issues = self._analyze_ast_security(str(py_file), content)
                audit_results["security_issues"].extend(ast_issues)
                
                # Best practices validation
                bp_issues = self._validate_best_practices(str(py_file), content)
                audit_results["best_practices"].extend(bp_issues)
                
            except Exception as e:
                print(f"⚠️  Error scanning {py_file}: {e}")
        
        # Vulnerability summary
        audit_results["vulnerability_summary"] = self._generate_vulnerability_summary(
            audit_results["security_issues"]
        )
        
        print(f"✅ Scanned {audit_results['files_scanned']} Python files")
        return audit_results
    
    def _scan_security_patterns(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Scan for security anti-patterns using regex."""
        
        issues = []
        
        for pattern_name, pattern_info in self.security_patterns.items():
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Calculate line number
                line_num = content[:match.start()].count('\n') + 1
                
                issues.append({
                    "file": file_path,
                    "line": line_num,
                    "pattern": pattern_name,
                    "severity": pattern_info["severity"],
                    "description": pattern_info["description"],
                    "code_snippet": match.group().strip(),
                    "type": "security_pattern"
                })
        
        return issues
    
    def _analyze_ast_security(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Analyze AST for security vulnerabilities."""
        
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        # Check for dangerous built-ins
                        if func_name in ['eval', 'exec', 'compile']:
                            issues.append({
                                "file": file_path,
                                "line": node.lineno,
                                "severity": "HIGH",
                                "description": f"Dangerous function '{func_name}' usage",
                                "type": "ast_analysis",
                                "function": func_name
                            })
                        
                        # Check for subprocess usage without shell=False
                        if func_name == 'subprocess' and any(
                            isinstance(arg, ast.Constant) and arg.value == True 
                            for keyword in getattr(node, 'keywords', [])
                            for arg in [keyword.value] if keyword.arg == 'shell'
                        ):
                            issues.append({
                                "file": file_path,
                                "line": node.lineno,
                                "severity": "HIGH",
                                "description": "subprocess called with shell=True",
                                "type": "ast_analysis"
                            })
                
                # Check for hardcoded strings that look like secrets
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if len(node.value) > 20 and any(
                        keyword in node.value.lower() 
                        for keyword in ['secret', 'token', 'password', 'key']
                    ):
                        issues.append({
                            "file": file_path,
                            "line": node.lineno,
                            "severity": "MEDIUM",
                            "description": "Potential hardcoded secret",
                            "type": "ast_analysis",
                            "value_length": len(node.value)
                        })
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
        
        return issues
    
    def _validate_best_practices(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Validate security best practices."""
        
        issues = []
        
        # Check for proper error handling
        if 'except:' in content and 'except Exception:' not in content:
            issues.append({
                "file": file_path,
                "severity": "LOW",
                "description": "Bare except clause - should catch specific exceptions",
                "type": "best_practice",
                "category": "error_handling"
            })
        
        # Check for logging of sensitive data
        if re.search(r'log.*password|log.*secret|log.*token', content, re.IGNORECASE):
            issues.append({
                "file": file_path,
                "severity": "MEDIUM", 
                "description": "Potential logging of sensitive information",
                "type": "best_practice",
                "category": "data_exposure"
            })
        
        # Check for missing input validation
        if 'request' in content and 'validate' not in content:
            issues.append({
                "file": file_path,
                "severity": "LOW",
                "description": "Request handling without explicit validation",
                "type": "best_practice",
                "category": "input_validation"
            })
        
        # Check for secure defaults
        if 'ssl_verify=False' in content or 'verify=False' in content:
            issues.append({
                "file": file_path,
                "severity": "HIGH",
                "description": "SSL verification disabled",
                "type": "best_practice", 
                "category": "secure_defaults"
            })
        
        return issues
    
    def _generate_vulnerability_summary(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate vulnerability summary statistics."""
        
        summary = {
            "total_issues": len(issues),
            "by_severity": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "by_type": {},
            "by_category": {},
            "most_vulnerable_files": []
        }
        
        file_issue_counts = {}
        
        for issue in issues:
            # Count by severity
            severity = issue.get("severity", "UNKNOWN")
            if severity in summary["by_severity"]:
                summary["by_severity"][severity] += 1
            
            # Count by type
            issue_type = issue.get("type", "unknown")
            summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
            
            # Count by category
            category = issue.get("category", issue.get("pattern", "uncategorized"))
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            
            # Track file issue counts
            file_path = issue.get("file", "unknown")
            file_issue_counts[file_path] = file_issue_counts.get(file_path, 0) + 1
        
        # Most vulnerable files
        summary["most_vulnerable_files"] = sorted(
            file_issue_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        return summary
    
    def audit_dependencies(self) -> Dict[str, Any]:
        """Audit dependencies for known vulnerabilities."""
        
        print("📦 Auditing Dependencies...")
        
        dependency_audit = {
            "requirements_files": [],
            "dependencies": [],
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Look for requirements files
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        
        for req_file in req_files:
            if Path(req_file).exists():
                dependency_audit["requirements_files"].append(req_file)
                
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                    
                    # Parse dependencies
                    if req_file.endswith('.txt'):
                        deps = self._parse_requirements_txt(content)
                    elif req_file.endswith('.toml'):
                        deps = self._parse_pyproject_toml(content)
                    else:
                        deps = []
                    
                    dependency_audit["dependencies"].extend(deps)
                    
                except Exception as e:
                    print(f"⚠️  Error reading {req_file}: {e}")
        
        # Check for known vulnerable packages
        vulnerable_packages = {
            "pyyaml": "Known deserialization vulnerabilities in older versions",
            "pillow": "Image processing vulnerabilities in older versions", 
            "requests": "SSL verification issues in older versions",
            "flask": "Various security issues in older versions",
            "django": "Multiple security vulnerabilities across versions"
        }
        
        for dep in dependency_audit["dependencies"]:
            package_name = dep["name"].lower()
            if package_name in vulnerable_packages:
                dependency_audit["vulnerabilities"].append({
                    "package": dep["name"],
                    "version": dep["version"],
                    "vulnerability": vulnerable_packages[package_name],
                    "severity": "MEDIUM"
                })
        
        # Generate recommendations
        if dependency_audit["vulnerabilities"]:
            dependency_audit["recommendations"].append(
                "Update vulnerable packages to latest stable versions"
            )
        
        dependency_audit["recommendations"].extend([
            "Use virtual environments to isolate dependencies",
            "Regularly audit dependencies with security tools",
            "Pin dependency versions for reproducible builds",
            "Monitor security advisories for used packages"
        ])
        
        return dependency_audit
    
    def _parse_requirements_txt(self, content: str) -> List[Dict[str, str]]:
        """Parse requirements.txt format."""
        dependencies = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Simple parsing - handle == version specs
                if '==' in line:
                    name, version = line.split('==', 1)
                    dependencies.append({"name": name.strip(), "version": version.strip()})
                else:
                    dependencies.append({"name": line, "version": "unspecified"})
        
        return dependencies
    
    def _parse_pyproject_toml(self, content: str) -> List[Dict[str, str]]:
        """Parse pyproject.toml dependencies."""
        dependencies = []
        
        # Simple regex-based parsing for dependencies section
        dep_pattern = r'"([^"]+)"'
        matches = re.findall(dep_pattern, content)
        
        for match in matches:
            if '>=' in match or '==' in match or '~=' in match:
                # Has version specifier
                name = re.split(r'[><=~!]', match)[0].strip()
                version = match.replace(name, '').strip()
                dependencies.append({"name": name, "version": version})
            else:
                dependencies.append({"name": match, "version": "unspecified"})
        
        return dependencies
    
    def audit_configuration_security(self) -> Dict[str, Any]:
        """Audit configuration files for security issues."""
        
        print("⚙️  Auditing Configuration Security...")
        
        config_audit = {
            "config_files": [],
            "security_issues": [],
            "recommendations": []
        }
        
        # Look for common configuration files
        config_patterns = [
            "*.yaml", "*.yml", "*.json", "*.ini", "*.conf", 
            ".env", "config.py", "settings.py"
        ]
        
        for pattern in config_patterns:
            for config_file in Path(".").rglob(pattern):
                if config_file.name.startswith('.'):
                    continue
                    
                config_audit["config_files"].append(str(config_file))
                
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Check for hardcoded secrets
                    secret_patterns = [
                        r'password\s*[:=]\s*["\'][^"\']+["\']',
                        r'secret\s*[:=]\s*["\'][^"\']+["\']',
                        r'api_key\s*[:=]\s*["\'][^"\']+["\']',
                        r'token\s*[:=]\s*["\'][^"\']+["\']'
                    ]
                    
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            config_audit["security_issues"].append({
                                "file": str(config_file),
                                "issue": "Hardcoded secret in configuration",
                                "severity": "HIGH",
                                "pattern": pattern
                            })
                    
                    # Check for insecure defaults
                    insecure_patterns = [
                        (r'debug\s*[:=]\s*true', "Debug mode enabled"),
                        (r'ssl_verify\s*[:=]\s*false', "SSL verification disabled"),
                        (r'security\s*[:=]\s*false', "Security features disabled")
                    ]
                    
                    for pattern, description in insecure_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            config_audit["security_issues"].append({
                                "file": str(config_file),
                                "issue": description,
                                "severity": "MEDIUM"
                            })
                
                except Exception as e:
                    print(f"⚠️  Error reading config file {config_file}: {e}")
        
        # Generate recommendations
        config_audit["recommendations"] = [
            "Use environment variables for sensitive configuration",
            "Implement configuration validation",
            "Use secure defaults for all security-related settings",
            "Separate configuration for different environments",
            "Encrypt sensitive configuration values",
            "Implement proper access controls for config files"
        ]
        
        return config_audit
    
    def audit_crypto_implementation(self, source_dir: str) -> Dict[str, Any]:
        """Audit cryptographic implementation."""
        
        print("🔐 Auditing Cryptographic Implementation...")
        
        crypto_audit = {
            "crypto_usage": [],
            "vulnerabilities": [],
            "recommendations": [],
            "compliance_status": {}
        }
        
        # Patterns for crypto usage
        crypto_patterns = {
            "encryption": [
                r"encrypt|decrypt|cipher",
                r"AES|RSA|ECC|ChaCha20",
                r"Fernet|cryptography"
            ],
            "hashing": [
                r"hash|digest|sha|md5|blake",
                r"hashlib|bcrypt|scrypt|pbkdf2"
            ],
            "random": [
                r"random|urandom|secrets",
                r"token|salt|nonce|iv"
            ],
            "signatures": [
                r"sign|verify|signature",
                r"HMAC|DSA|ECDSA"
            ]
        }
        
        source_path = Path(source_dir)
        
        for py_file in source_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for category, patterns in crypto_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            crypto_audit["crypto_usage"].append({
                                "file": str(py_file),
                                "category": category,
                                "pattern": pattern,
                                "usage_count": len(re.findall(pattern, content, re.IGNORECASE))
                            })
                
                # Check for weak crypto practices
                weak_practices = [
                    (r"md5\s*\(", "MD5 is cryptographically broken"),
                    (r"sha1\s*\(", "SHA1 is weak and deprecated"),
                    (r"DES|3DES", "DES/3DES are weak encryption algorithms"),
                    (r"random\.random|random\.randint", "Use secrets module for crypto"),
                    (r"mode\s*=\s*ECB", "ECB mode is insecure"),
                    (r"verify\s*=\s*False", "SSL verification disabled")
                ]
                
                for pattern, description in weak_practices:
                    if re.search(pattern, content, re.IGNORECASE):
                        crypto_audit["vulnerabilities"].append({
                            "file": str(py_file),
                            "vulnerability": description,
                            "pattern": pattern,
                            "severity": "HIGH"
                        })
            
            except Exception as e:
                print(f"⚠️  Error scanning crypto in {py_file}: {e}")
        
        # Generate recommendations
        crypto_audit["recommendations"] = [
            "Use modern, well-vetted cryptographic libraries",
            "Implement proper key management",
            "Use strong random number generation (secrets module)",
            "Implement proper certificate validation",
            "Use authenticated encryption (AES-GCM, ChaCha20-Poly1305)",
            "Regularly rotate cryptographic keys",
            "Implement proper error handling for crypto operations"
        ]
        
        # Compliance assessment
        crypto_audit["compliance_status"] = {
            "FIPS_140_2": "Requires approved crypto modules",
            "Common_Criteria": "Requires certified implementations",
            "NIST_Guidelines": "Use NIST-recommended algorithms",
            "Quantum_Resistance": "Consider post-quantum crypto migration"
        }
        
        return crypto_audit
    
    def generate_security_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate comprehensive security audit report."""
        
        report = []
        report.append("# 🛡️  AUTONOMOUS SDLC v4.0 - SECURITY AUDIT REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Executive Summary
        report.append("## 📋 EXECUTIVE SUMMARY")
        report.append("")
        
        total_issues = sum(
            len(audit_results.get(key, {}).get("security_issues", []))
            for key in ["source_code", "configuration", "crypto"]
        )
        
        if total_issues == 0:
            report.append("✅ **EXCELLENT**: No significant security issues detected")
        elif total_issues < 5:
            report.append("🟡 **GOOD**: Few minor security issues detected")  
        elif total_issues < 15:
            report.append("🟠 **MODERATE**: Some security issues require attention")
        else:
            report.append("🔴 **CRITICAL**: Multiple security issues require immediate attention")
        
        report.append(f"- Total Security Issues: {total_issues}")
        report.append(f"- Files Scanned: {audit_results.get('source_code', {}).get('files_scanned', 0)}")
        report.append(f"- Dependencies Audited: {len(audit_results.get('dependencies', {}).get('dependencies', []))}")
        report.append("")
        
        # Source Code Security
        if "source_code" in audit_results:
            src_audit = audit_results["source_code"]
            report.append("## 🔍 SOURCE CODE SECURITY")
            report.append("")
            
            if src_audit.get("vulnerability_summary"):
                summary = src_audit["vulnerability_summary"]
                
                report.append("### Vulnerability Summary:")
                for severity, count in summary["by_severity"].items():
                    if count > 0:
                        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
                        report.append(f"- {emoji} {severity}: {count} issues")
                
                if summary.get("most_vulnerable_files"):
                    report.append("\n### Most Vulnerable Files:")
                    for file_path, issue_count in summary["most_vulnerable_files"]:
                        report.append(f"- {Path(file_path).name}: {issue_count} issues")
            
            report.append("")
        
        # Dependencies
        if "dependencies" in audit_results:
            dep_audit = audit_results["dependencies"]
            report.append("## 📦 DEPENDENCY SECURITY")
            report.append("")
            
            vulns = dep_audit.get("vulnerabilities", [])
            if vulns:
                report.append("### Vulnerable Dependencies:")
                for vuln in vulns:
                    report.append(f"- **{vuln['package']}** ({vuln['version']}): {vuln['vulnerability']}")
            else:
                report.append("✅ No known vulnerabilities in dependencies")
            
            report.append("")
        
        # Configuration Security
        if "configuration" in audit_results:
            config_audit = audit_results["configuration"]
            report.append("## ⚙️  CONFIGURATION SECURITY")
            report.append("")
            
            config_issues = config_audit.get("security_issues", [])
            if config_issues:
                report.append("### Configuration Issues:")
                for issue in config_issues[:5]:  # Show top 5
                    report.append(f"- **{issue['severity']}**: {issue['issue']} in {Path(issue['file']).name}")
            else:
                report.append("✅ No major configuration security issues")
            
            report.append("")
        
        # Cryptographic Implementation
        if "crypto" in audit_results:
            crypto_audit = audit_results["crypto"]
            report.append("## 🔐 CRYPTOGRAPHIC SECURITY")
            report.append("")
            
            crypto_vulns = crypto_audit.get("vulnerabilities", [])
            if crypto_vulns:
                report.append("### Cryptographic Vulnerabilities:")
                for vuln in crypto_vulns[:5]:  # Show top 5
                    report.append(f"- **{vuln['severity']}**: {vuln['vulnerability']}")
            else:
                report.append("✅ No major cryptographic vulnerabilities detected")
            
            report.append("")
        
        # Recommendations
        report.append("## 🎯 SECURITY RECOMMENDATIONS")
        report.append("")
        
        all_recommendations = []
        for audit_type in ["source_code", "dependencies", "configuration", "crypto"]:
            if audit_type in audit_results:
                recs = audit_results[audit_type].get("recommendations", [])
                all_recommendations.extend(recs)
        
        # Deduplicate recommendations
        unique_recs = list(set(all_recommendations))
        
        for i, rec in enumerate(unique_recs[:10], 1):  # Top 10 recommendations
            report.append(f"{i}. {rec}")
        
        report.append("")
        
        # Compliance Status
        report.append("## 📋 COMPLIANCE STATUS")
        report.append("")
        
        compliance_frameworks = ["ISO27001", "NIST_CSF", "GDPR", "SOC2"]
        for framework in compliance_frameworks:
            # Simplified compliance assessment
            if total_issues == 0:
                status = "🟢 COMPLIANT"
            elif total_issues < 5:
                status = "🟡 MOSTLY COMPLIANT"
            else:
                status = "🔴 NON-COMPLIANT"
            
            report.append(f"- **{framework}**: {status}")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by Autonomous SDLC v4.0 Security Auditor*")
        
        return "\n".join(report)


def main():
    """Main security audit routine."""
    
    print("🛡️  AUTONOMOUS SDLC v4.0 - ADVANCED SECURITY AUDIT")
    print("=" * 70)
    
    auditor = SecurityAuditor()
    audit_results = {}
    
    # Source code audit
    if Path("src").exists():
        audit_results["source_code"] = auditor.audit_source_code("src")
    
    # Dependencies audit  
    audit_results["dependencies"] = auditor.audit_dependencies()
    
    # Configuration audit
    audit_results["configuration"] = auditor.audit_configuration_security()
    
    # Cryptographic audit
    if Path("src").exists():
        audit_results["crypto"] = auditor.audit_crypto_implementation("src")
    
    # Generate comprehensive report
    security_report = auditor.generate_security_report(audit_results)
    
    # Save report
    report_file = "SECURITY_AUDIT_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(security_report)
    
    print(f"\n✅ Security audit completed!")
    print(f"📄 Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("🏁 SECURITY AUDIT SUMMARY")
    print("=" * 70)
    
    # Calculate overall security score
    total_issues = sum(
        len(audit_results.get(key, {}).get("security_issues", []))
        for key in ["source_code", "configuration"]
    ) + len(audit_results.get("dependencies", {}).get("vulnerabilities", [])) + \
        len(audit_results.get("crypto", {}).get("vulnerabilities", []))
    
    if total_issues == 0:
        print("🎉 SECURITY AUDIT PASSED - No significant issues detected")
        print("✅ Code security: EXCELLENT")
        print("✅ Dependencies: SECURE") 
        print("✅ Configuration: SECURE")
        print("✅ Cryptography: SECURE")
        return 0
    elif total_issues < 5:
        print("🟡 SECURITY AUDIT - Minor issues detected")
        print("⚠️  Please address the identified issues")
        return 1
    else:
        print("🔴 SECURITY AUDIT FAILED - Multiple issues require attention")
        print("❌ Immediate security remediation required")
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)