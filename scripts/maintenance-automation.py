#!/usr/bin/env python3
"""
Maintenance Automation Script for Spike-Transformer-Compiler

This script automates common maintenance tasks including:
- Code quality checks
- Dependency management
- Repository cleanup
- Performance monitoring
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaintenanceAutomation:
    """Automated maintenance tasks for the repository."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.results = {}
    
    def run_command(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        logger.info(f"Running: {cmd}")
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, 
            cwd=self.repo_path, check=check
        )
        if result.returncode != 0 and check:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"Error: {result.stderr}")
        return result
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Run code quality checks."""
        logger.info("Running code quality checks...")
        quality_results = {}
        
        # Run flake8
        try:
            result = self.run_command("flake8 src/ tests/ --count --statistics", check=False)
            quality_results['flake8'] = {
                'exit_code': result.returncode,
                'issues': result.stdout.count('\n') if result.stdout else 0
            }
        except Exception as e:
            quality_results['flake8'] = {'error': str(e)}
        
        # Run black check
        try:
            result = self.run_command("black --check src/ tests/", check=False)
            quality_results['black'] = {
                'exit_code': result.returncode,
                'needs_formatting': result.returncode != 0
            }
        except Exception as e:
            quality_results['black'] = {'error': str(e)}
        
        # Run mypy
        try:
            result = self.run_command("mypy src/", check=False)
            quality_results['mypy'] = {
                'exit_code': result.returncode,
                'type_errors': result.stdout.count('error:') if result.stdout else 0
            }
        except Exception as e:
            quality_results['mypy'] = {'error': str(e)}
        
        self.results['code_quality'] = quality_results
        return quality_results
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check for outdated and vulnerable dependencies."""
        logger.info("Checking dependencies...")
        dep_results = {}
        
        # Check for outdated packages
        try:
            result = self.run_command("pip list --outdated --format=json", check=False)
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                dep_results['outdated'] = {
                    'count': len(outdated),
                    'packages': [pkg['name'] for pkg in outdated]
                }
            else:
                dep_results['outdated'] = {'count': 0, 'packages': []}
        except Exception as e:
            dep_results['outdated'] = {'error': str(e)}
        
        # Run safety check
        try:
            result = self.run_command("safety check --json", check=False)
            if result.returncode == 0:
                safety_data = json.loads(result.stdout) if result.stdout else []
                dep_results['vulnerabilities'] = {
                    'count': len(safety_data),
                    'issues': safety_data
                }
            else:
                dep_results['vulnerabilities'] = {'count': 0, 'issues': []}
        except Exception as e:
            dep_results['vulnerabilities'] = {'error': str(e)}
        
        self.results['dependencies'] = dep_results
        return dep_results
    
    def cleanup_repository(self) -> Dict[str, Any]:
        """Clean up repository artifacts and temporary files."""
        logger.info("Cleaning up repository...")
        cleanup_results = {'removed_files': [], 'freed_space_mb': 0}
        
        # Patterns for files to clean up
        cleanup_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/.pytest_cache',
            '**/.coverage',
            '**/htmlcov',
            '**/.mypy_cache',
            '**/build',
            '**/dist',
            '**/*.egg-info'
        ]
        
        initial_size = self._get_directory_size()
        
        for pattern in cleanup_patterns:
            for path in self.repo_path.glob(pattern):
                if path.exists():
                    try:
                        if path.is_file():
                            path.unlink()
                            cleanup_results['removed_files'].append(str(path))
                        elif path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                            cleanup_results['removed_files'].append(str(path))
                    except Exception as e:
                        logger.warning(f"Failed to remove {path}: {e}")
        
        final_size = self._get_directory_size()
        cleanup_results['freed_space_mb'] = (initial_size - final_size) / (1024 * 1024)
        
        self.results['cleanup'] = cleanup_results
        return cleanup_results
    
    def _get_directory_size(self) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.repo_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def check_git_health(self) -> Dict[str, Any]:
        """Check Git repository health."""
        logger.info("Checking Git repository health...")
        git_results = {}
        
        # Check for uncommitted changes
        try:
            result = self.run_command("git status --porcelain", check=False)
            git_results['uncommitted_changes'] = len(result.stdout.splitlines()) if result.stdout else 0
        except Exception as e:
            git_results['uncommitted_changes'] = {'error': str(e)}
        
        # Check for unpushed commits
        try:
            result = self.run_command("git log @{u}..HEAD --oneline", check=False)
            git_results['unpushed_commits'] = len(result.stdout.splitlines()) if result.stdout else 0
        except Exception as e:
            git_results['unpushed_commits'] = {'error': str(e)}
        
        # Check repository size
        try:
            result = self.run_command("git count-objects -vH", check=False)
            git_results['repo_info'] = result.stdout if result.returncode == 0 else "Unknown"
        except Exception as e:
            git_results['repo_info'] = {'error': str(e)}
        
        self.results['git_health'] = git_results
        return git_results
    
    def run_performance_checks(self) -> Dict[str, Any]:
        """Run performance benchmarks and checks."""
        logger.info("Running performance checks...")
        perf_results = {}
        
        # Run pytest benchmarks if available
        try:
            result = self.run_command("python -m pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json", check=False)
            if result.returncode == 0 and Path("benchmark.json").exists():
                with open("benchmark.json", 'r') as f:
                    benchmark_data = json.load(f)
                perf_results['benchmarks'] = {
                    'status': 'success',
                    'tests_run': len(benchmark_data.get('benchmarks', [])),
                    'data_file': 'benchmark.json'
                }
            else:
                perf_results['benchmarks'] = {'status': 'no_benchmarks_or_failed'}
        except Exception as e:
            perf_results['benchmarks'] = {'error': str(e)}
        
        # Check import time
        try:
            result = self.run_command("python -c 'import time; start=time.time(); import spike_transformer_compiler; print(f\"Import time: {time.time()-start:.3f}s\")'")
            if result.returncode == 0:
                import_time = float(result.stdout.split(': ')[1].replace('s', ''))
                perf_results['import_time'] = {'seconds': import_time, 'status': 'fast' if import_time < 1.0 else 'slow'}
        except Exception as e:
            perf_results['import_time'] = {'error': str(e)}
        
        self.results['performance'] = perf_results
        return perf_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive maintenance report."""
        report = []
        report.append(f"# Maintenance Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Code Quality Summary
        if 'code_quality' in self.results:
            cq = self.results['code_quality']
            report.append("## Code Quality")
            if 'flake8' in cq:
                status = "✅ PASS" if cq['flake8'].get('exit_code') == 0 else "❌ FAIL"
                issues = cq['flake8'].get('issues', 0)
                report.append(f"- Flake8: {status} ({issues} issues)")
            if 'black' in cq:
                status = "✅ PASS" if not cq['black'].get('needs_formatting') else "❌ NEEDS FORMATTING"
                report.append(f"- Black: {status}")
            if 'mypy' in cq:
                status = "✅ PASS" if cq['mypy'].get('exit_code') == 0 else "❌ FAIL"
                errors = cq['mypy'].get('type_errors', 0)
                report.append(f"- MyPy: {status} ({errors} type errors)")
            report.append("")
        
        # Dependencies Summary
        if 'dependencies' in self.results:
            dep = self.results['dependencies']
            report.append("## Dependencies")
            if 'outdated' in dep:
                count = dep['outdated'].get('count', 0)
                status = "✅ UP TO DATE" if count == 0 else f"⚠️  {count} OUTDATED"
                report.append(f"- Outdated packages: {status}")
                if count > 0 and 'packages' in dep['outdated']:
                    for pkg in dep['outdated']['packages'][:5]:  # Show first 5
                        report.append(f"  - {pkg}")
                    if count > 5:
                        report.append(f"  - ... and {count - 5} more")
            
            if 'vulnerabilities' in dep:
                vuln_count = dep['vulnerabilities'].get('count', 0)
                status = "✅ SECURE" if vuln_count == 0 else f"🚨 {vuln_count} VULNERABILITIES"
                report.append(f"- Security vulnerabilities: {status}")
            report.append("")
        
        # Cleanup Summary
        if 'cleanup' in self.results:
            cleanup = self.results['cleanup']
            removed_count = len(cleanup.get('removed_files', []))
            freed_mb = cleanup.get('freed_space_mb', 0)
            report.append("## Repository Cleanup")
            report.append(f"- Files removed: {removed_count}")
            report.append(f"- Space freed: {freed_mb:.2f} MB")
            report.append("")
        
        # Git Health Summary
        if 'git_health' in self.results:
            git = self.results['git_health']
            report.append("## Git Repository Health")
            uncommitted = git.get('uncommitted_changes', 0)
            unpushed = git.get('unpushed_commits', 0)
            report.append(f"- Uncommitted changes: {uncommitted}")
            report.append(f"- Unpushed commits: {unpushed}")
            report.append("")
        
        # Performance Summary
        if 'performance' in self.results:
            perf = self.results['performance']
            report.append("## Performance")
            if 'benchmarks' in perf:
                bench_status = perf['benchmarks'].get('status', 'unknown')
                if bench_status == 'success':
                    tests_run = perf['benchmarks'].get('tests_run', 0)
                    report.append(f"- Benchmarks: ✅ {tests_run} tests completed")
                else:
                    report.append(f"- Benchmarks: ⚠️  {bench_status}")
            
            if 'import_time' in perf and 'seconds' in perf['import_time']:
                import_time = perf['import_time']['seconds']
                status = perf['import_time']['status']
                emoji = "✅" if status == 'fast' else "⚠️ "
                report.append(f"- Import time: {emoji} {import_time:.3f}s ({status})")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        recommendations = self._generate_recommendations()
        if recommendations:
            for rec in recommendations:
                report.append(f"- {rec}")
        else:
            report.append("- No specific recommendations at this time")
        
        return "\n".join(report)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate maintenance recommendations based on results."""
        recommendations = []
        
        # Code quality recommendations
        if 'code_quality' in self.results:
            cq = self.results['code_quality']
            if cq.get('flake8', {}).get('issues', 0) > 0:
                recommendations.append("Fix flake8 linting issues")
            if cq.get('black', {}).get('needs_formatting'):
                recommendations.append("Run black formatter to fix code style")
            if cq.get('mypy', {}).get('type_errors', 0) > 0:
                recommendations.append("Fix mypy type checking errors")
        
        # Dependency recommendations
        if 'dependencies' in self.results:
            dep = self.results['dependencies']
            if dep.get('outdated', {}).get('count', 0) > 5:
                recommendations.append("Update outdated dependencies")
            if dep.get('vulnerabilities', {}).get('count', 0) > 0:
                recommendations.append("🚨 URGENT: Fix security vulnerabilities")
        
        # Performance recommendations
        if 'performance' in self.results:
            perf = self.results['performance']
            if perf.get('import_time', {}).get('status') == 'slow':
                recommendations.append("Investigate slow module import times")
        
        return recommendations
    
    def save_results(self, output_file: Path = None) -> None:
        """Save maintenance results to a JSON file."""
        if output_file is None:
            output_file = self.repo_path / "maintenance-results.json"
        
        results_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_with_timestamp, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main function for the maintenance automation script."""
    parser = argparse.ArgumentParser(description='Automated maintenance for Spike-Transformer-Compiler')
    parser.add_argument('--tasks', nargs='+', 
                       choices=['quality', 'dependencies', 'cleanup', 'git', 'performance', 'all'],
                       default=['all'], help='Tasks to run')
    parser.add_argument('--output', type=Path, help='Output file for results')
    parser.add_argument('--repo-path', type=Path, default=Path.cwd(), help='Repository path')
    parser.add_argument('--report', action='store_true', help='Generate and display report')
    
    args = parser.parse_args()
    
    # Initialize maintenance automation
    maintenance = MaintenanceAutomation(args.repo_path)
    
    # Determine which tasks to run
    if 'all' in args.tasks:
        tasks = ['quality', 'dependencies', 'cleanup', 'git', 'performance']
    else:
        tasks = args.tasks
    
    # Run selected tasks
    logger.info(f"Running maintenance tasks: {', '.join(tasks)}")
    
    if 'quality' in tasks:
        maintenance.check_code_quality()
    
    if 'dependencies' in tasks:
        maintenance.check_dependencies()
    
    if 'cleanup' in tasks:
        maintenance.cleanup_repository()
    
    if 'git' in tasks:
        maintenance.check_git_health()
    
    if 'performance' in tasks:
        maintenance.run_performance_checks()
    
    # Save results
    maintenance.save_results(args.output)
    
    # Generate and display report
    if args.report:
        report = maintenance.generate_report()
        print("\n" + "="*50)
        print(report)
        print("="*50)
    
    logger.info("Maintenance automation completed successfully!")


if __name__ == "__main__":
    main()
