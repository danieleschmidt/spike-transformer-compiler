#!/usr/bin/env python3
"""
Autonomous Signal Harvesting System for Value Discovery
Terragon Labs - Autonomous SDLC Enhancement
"""

import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import ast

@dataclass
class ValueOpportunity:
    """Represents a discovered value opportunity."""
    id: str
    title: str
    description: str
    category: str
    source: str
    file_path: Optional[str]
    line_number: Optional[int]
    estimated_effort: float  # hours
    business_impact: int  # 1-10 scale
    technical_debt_score: int  # 0-100 scale
    urgency: int  # 1-10 scale
    confidence: float  # 0.0-1.0
    keywords: List[str]
    discovered_at: str
    last_updated: str

class SignalHarvester:
    """Comprehensive signal harvesting for autonomous value discovery."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config = self._load_config(config_path)
        self.repo_root = Path.cwd()
        self.opportunities: List[ValueOpportunity] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            # Simple YAML-like parsing for our specific config
            with open(config_path, 'r') as f:
                content = f.read()
            # For now, use default config (full YAML parsing would require PyYAML)
            return self._default_config()
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found."""
        return {
            'discovery': {
                'sources': {
                    'git_history': {'enabled': True, 'weight': 0.25},
                    'static_analysis': {'enabled': True, 'weight': 0.30},
                    'issue_trackers': {'enabled': True, 'weight': 0.20},
                }
            },
            'classification': {
                'categories': {
                    'security': {'base_priority': 90},
                    'implementation': {'base_priority': 70},
                    'performance': {'base_priority': 60},
                    'technical_debt': {'base_priority': 50},
                }
            }
        }
    
    def harvest_all_signals(self) -> List[ValueOpportunity]:
        """Execute comprehensive signal harvesting."""
        print("🔍 Starting comprehensive signal harvesting...")
        
        # Harvest from all enabled sources
        if self.config['discovery']['sources']['git_history']['enabled']:
            self._harvest_git_signals()
            
        if self.config['discovery']['sources']['static_analysis']['enabled']:
            self._harvest_static_analysis_signals()
            
        if self.config['discovery']['sources']['issue_trackers']['enabled']:
            self._harvest_github_signals()
        
        # Custom domain-specific harvesting
        self._harvest_implementation_gaps()
        self._harvest_missing_features()
        
        print(f"📊 Discovered {len(self.opportunities)} value opportunities")
        return self.opportunities
    
    def _harvest_git_signals(self) -> None:
        """Harvest signals from Git history and code comments."""
        print("  📝 Harvesting Git history signals...")
        
        # Scan for TODO, FIXME, HACK, etc. in code
        code_patterns = [
            (r'TODO:?\s*(.+)', 'implementation', 'TODO found in code'),
            (r'FIXME:?\s*(.+)', 'technical_debt', 'FIXME found in code'),
            (r'HACK:?\s*(.+)', 'technical_debt', 'Code hack identified'),
            (r'XXX:?\s*(.+)', 'technical_debt', 'Code issue marked'),
            (r'NotImplementedError\s*\(\s*["\'](.+?)["\']\s*\)', 'implementation', 'Not implemented feature'),
            (r'raise NotImplementedError', 'implementation', 'Missing implementation'),
        ]
        
        # Scan Python files
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern, category, description in code_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            detail = match.group(1) if match.groups() else line.strip()
                            self._add_opportunity(
                                title=f"{description}: {detail[:60]}...",
                                description=f"Found in {py_file}:{line_num}\n{line.strip()}",
                                category=category,
                                source="git_history",
                                file_path=str(py_file),
                                line_number=line_num,
                                keywords=[category, "code_comment"]
                            )
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    
    def _harvest_static_analysis_signals(self) -> None:
        """Run static analysis tools and collect signals."""
        print("  🔬 Running static analysis...")
        
        # Check for missing dependencies
        self._check_missing_dependencies()
        
        # Analyze code complexity
        self._analyze_code_complexity()
        
        # Check for security issues
        self._check_security_issues()
    
    def _check_missing_dependencies(self) -> None:
        """Check for missing or outdated dependencies."""
        try:
            # Check if requirements files exist
            req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
            missing_reqs = []
            
            for req_file in req_files:
                if not (self.repo_root / req_file).exists():
                    missing_reqs.append(req_file)
            
            if missing_reqs:
                self._add_opportunity(
                    title="Missing dependency management files",
                    description=f"Missing files: {', '.join(missing_reqs)}",
                    category="automation",
                    source="static_analysis",
                    keywords=["dependencies", "missing"]
                )
            
            # Check for outdated packages if pip-outdated is available
            try:
                result = subprocess.run(
                    ["pip", "list", "--outdated", "--format=json"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    outdated = json.loads(result.stdout)
                    if outdated:
                        self._add_opportunity(
                            title=f"Outdated dependencies ({len(outdated)} packages)",
                            description=f"Outdated packages: {[p['name'] for p in outdated[:5]]}",
                            category="security",
                            source="static_analysis",
                            keywords=["dependencies", "outdated", "security"]
                        )
            except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
                pass
                
        except Exception as e:
            print(f"  ⚠️  Dependency check failed: {e}")
    
    def _analyze_code_complexity(self) -> None:
        """Analyze code complexity and identify hotspots."""
        # Simple complexity analysis
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count lines of code
                lines = [line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
                loc = len(lines)
                
                # If file is very large, suggest refactoring
                if loc > 500:
                    self._add_opportunity(
                        title=f"Large file needs refactoring: {py_file.name}",
                        description=f"File has {loc} lines of code, consider breaking into smaller modules",
                        category="technical_debt",
                        source="static_analysis",
                        file_path=str(py_file),
                        keywords=["complexity", "refactor", "large_file"]
                    )
                
                # Count function complexity (simple metric)
                try:
                    tree = ast.parse(content)
                    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    
                    for func in functions:
                        # Simple complexity: count if/for/while statements
                        complexity = len([n for n in ast.walk(func) 
                                        if isinstance(n, (ast.If, ast.For, ast.While))])
                        
                        if complexity > 10:
                            self._add_opportunity(
                                title=f"High complexity function: {func.name}",
                                description=f"Function has complexity score of {complexity}",
                                category="technical_debt",
                                source="static_analysis",
                                file_path=str(py_file),
                                line_number=func.lineno,
                                keywords=["complexity", "refactor", "function"]
                            )
                            
                except SyntaxError:
                    # Skip files with syntax errors
                    continue
                    
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    
    def _check_security_issues(self) -> None:
        """Check for basic security issues."""
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
            (r'exec\s*\(', 'Dynamic code execution found'),
            (r'eval\s*\(', 'Dynamic evaluation found'),
        ]
        
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern, description in security_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            self._add_opportunity(
                                title=f"Security issue: {description}",
                                description=f"Found in {py_file}:{line_num}",
                                category="security",
                                source="static_analysis",
                                file_path=str(py_file),
                                line_number=line_num,
                                keywords=["security", "vulnerability"]
                            )
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    
    def _harvest_github_signals(self) -> None:
        """Harvest signals from GitHub issues and PRs."""
        print("  🐙 Harvesting GitHub signals...")
        
        # Check for missing GitHub workflows
        workflows_dir = self.repo_root / ".github" / "workflows"
        if not workflows_dir.exists():
            self._add_opportunity(
                title="Missing GitHub Actions CI/CD workflows",
                description="No CI/CD automation found. This is critical for MATURING maturity level.",
                category="automation",
                source="github",
                keywords=["ci", "cd", "automation", "critical"]
            )
        else:
            # Check for essential workflows
            essential_workflows = ["test", "lint", "security", "build", "deploy"]
            existing_workflows = [f.stem for f in workflows_dir.glob("*.yml")] + [f.stem for f in workflows_dir.glob("*.yaml")]
            
            for workflow in essential_workflows:
                if not any(workflow in existing for existing in existing_workflows):
                    self._add_opportunity(
                        title=f"Missing {workflow} workflow",
                        description=f"No {workflow} automation found in GitHub Actions",
                        category="automation",
                        source="github",
                        keywords=["ci", "automation", workflow]
                    )
    
    def _harvest_implementation_gaps(self) -> None:
        """Identify implementation gaps specific to this repository."""
        print("  🔧 Identifying implementation gaps...")
        
        # Check core compiler files for implementation status
        core_files = [
            "src/spike_transformer_compiler/compiler.py",
            "src/spike_transformer_compiler/backend.py", 
            "src/spike_transformer_compiler/optimization.py"
        ]
        
        for file_path in core_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Count NotImplementedError instances
                    not_implemented_count = content.count("NotImplementedError")
                    if not_implemented_count > 0:
                        self._add_opportunity(
                            title=f"Core implementation missing in {Path(file_path).name}",
                            description=f"Found {not_implemented_count} NotImplementedError instances",
                            category="implementation", 
                            source="implementation_analysis",
                            file_path=file_path,
                            keywords=["core", "implementation", "critical"]
                        )
                        
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
            else:
                self._add_opportunity(
                    title=f"Missing core file: {Path(file_path).name}",
                    description=f"Expected core file not found: {file_path}",
                    category="implementation",
                    source="implementation_analysis", 
                    keywords=["missing", "core", "critical"]
                )
    
    def _harvest_missing_features(self) -> None:
        """Identify missing features for a MATURING repository."""
        print("  🚀 Identifying missing advanced features...")
        
        # Check for advanced SDLC features
        advanced_features = [
            (".github/dependabot.yml", "Automated dependency updates", "automation"),
            (".pre-commit-config.yaml", "Pre-commit hooks configuration", "automation"),
            ("docker-compose.test.yml", "Testing environment configuration", "testing"),
            ("benchmarks/", "Performance benchmarking", "performance"),
            (".codecov.yml", "Code coverage configuration", "testing"),
            ("docs/api/", "API documentation", "documentation"),
        ]
        
        for path, description, category in advanced_features:
            if not (self.repo_root / path).exists():
                self._add_opportunity(
                    title=f"Missing {description.lower()}",
                    description=f"MATURING repositories should have {description.lower()}",
                    category=category,
                    source="maturity_analysis",
                    keywords=["advanced", "maturing", category]
                )
    
    def _add_opportunity(
        self, 
        title: str, 
        description: str, 
        category: str,
        source: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        keywords: Optional[List[str]] = None
    ) -> None:
        """Add a discovered opportunity to the collection."""
        opportunity_id = f"{category}_{len(self.opportunities)+1:03d}"
        
        # Estimate effort based on category and complexity
        effort_map = {
            "security": 2.0,
            "implementation": 4.0, 
            "automation": 3.0,
            "performance": 3.5,
            "technical_debt": 2.5,
            "documentation": 1.5
        }
        
        # Calculate business impact based on category
        impact_map = {
            "security": 9,
            "implementation": 8,
            "automation": 7,
            "performance": 6,
            "technical_debt": 5,
            "documentation": 4
        }
        
        # Apply domain-specific boosts
        domain_keywords = ["loihi", "neuromorphic", "compiler", "spike"]
        domain_boost = 1.0
        if keywords and any(kw in " ".join(keywords).lower() for kw in domain_keywords):
            domain_boost = 1.3
        
        opportunity = ValueOpportunity(
            id=opportunity_id,
            title=title,
            description=description,
            category=category,
            source=source,
            file_path=file_path,
            line_number=line_number,
            estimated_effort=effort_map.get(category, 2.0),
            business_impact=min(10, int(impact_map.get(category, 5) * domain_boost)),
            technical_debt_score=self._calculate_debt_score(category, keywords or []),
            urgency=self._calculate_urgency(category, keywords or []),
            confidence=0.8,  # Default confidence
            keywords=keywords or [],
            discovered_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        self.opportunities.append(opportunity)
    
    def _calculate_debt_score(self, category: str, keywords: List[str]) -> int:
        """Calculate technical debt score (0-100)."""
        base_scores = {
            "security": 15,  # Security issues create minimal debt
            "implementation": 85,  # Missing implementation is high debt
            "automation": 60,  # Missing automation creates moderate debt
            "performance": 45,  # Performance issues create some debt
            "technical_debt": 80,  # Direct debt category
            "documentation": 25  # Doc issues create low debt
        }
        
        score = base_scores.get(category, 50)
        
        # Boost for critical keywords
        critical_keywords = ["critical", "core", "missing", "not_implemented"]
        if any(kw in " ".join(keywords).lower() for kw in critical_keywords):
            score = min(100, int(score * 1.2))
            
        return score
    
    def _calculate_urgency(self, category: str, keywords: List[str]) -> int:
        """Calculate urgency score (1-10)."""
        base_urgency = {
            "security": 9,
            "implementation": 7,
            "automation": 6,
            "performance": 5,
            "technical_debt": 4,
            "documentation": 3
        }
        
        urgency = base_urgency.get(category, 5)
        
        # Boost for urgent keywords
        urgent_keywords = ["critical", "breaking", "vulnerability", "missing"]
        if any(kw in " ".join(keywords).lower() for kw in urgent_keywords):
            urgency = min(10, urgency + 2)
            
        return urgency
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".egg-info"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)
    
    def export_opportunities(self, output_file: str) -> None:
        """Export discovered opportunities to JSON file."""
        with open(output_file, 'w') as f:
            json.dump([asdict(opp) for opp in self.opportunities], f, indent=2)
        
        print(f"📁 Exported {len(self.opportunities)} opportunities to {output_file}")

def main():
    """Main execution function."""
    harvester = SignalHarvester()
    opportunities = harvester.harvest_all_signals()
    
    # Export results
    harvester.export_opportunities(".terragon/discovered-opportunities.json")
    
    # Print summary
    by_category = {}
    for opp in opportunities:
        by_category.setdefault(opp.category, 0)
        by_category[opp.category] += 1
    
    print(f"\n📊 Discovery Summary:")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count} opportunities")

if __name__ == "__main__":
    main()