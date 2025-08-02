#!/usr/bin/env python3
"""
Metrics Collection Script for Spike-Transformer-Compiler

This script collects various metrics about the repository and updates
the project-metrics.json file with current values.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def run_command(cmd: str) -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def count_files_by_type() -> Dict[str, int]:
    """Count different types of files in the repository."""
    src_files = len(list(Path("src").rglob("*.py"))) if Path("src").exists() else 0
    test_files = len(list(Path("tests").rglob("*.py"))) if Path("tests").exists() else 0
    doc_files = len(list(Path("docs").rglob("*.md"))) if Path("docs").exists() else 0
    
    # Count README and other markdown files in root
    root_docs = len(list(Path(".").glob("*.md")))
    doc_files += root_docs
    
    total_files = src_files + test_files + doc_files
    
    return {
        "total_files": total_files,
        "source_files": src_files,
        "test_files": test_files,
        "doc_files": doc_files
    }


def count_lines_of_code() -> Dict[str, int]:
    """Count lines of code in different categories."""
    source_lines = 0
    test_lines = 0
    comment_lines = 0
    
    # Count source lines
    if Path("src").exists():
        for py_file in Path("src").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    source_lines += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                    comment_lines += len([l for l in lines if l.strip().startswith('#')])
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    
    # Count test lines
    if Path("tests").exists():
        for py_file in Path("tests").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    test_lines += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    
    return {
        "total": source_lines + test_lines + comment_lines,
        "source": source_lines,
        "tests": test_lines,
        "comments": comment_lines
    }


def count_dependencies() -> Dict[str, int]:
    """Count production and development dependencies."""
    production = 0
    development = 0
    
    # Count from requirements.txt
    if Path("requirements.txt").exists():
        with open("requirements.txt", 'r') as f:
            production = len([l for l in f.readlines() if l.strip() and not l.startswith('#')])
    
    # Count from requirements-dev.txt
    if Path("requirements-dev.txt").exists():
        with open("requirements-dev.txt", 'r') as f:
            development = len([l for l in f.readlines() if l.strip() and not l.startswith('#')])
    
    # Check for outdated packages
    outdated_output = run_command("pip list --outdated --format=json")
    outdated = 0
    if outdated_output:
        try:
            outdated_packages = json.loads(outdated_output)
            outdated = len(outdated_packages)
        except json.JSONDecodeError:
            pass
    
    return {
        "production": production,
        "development": development,
        "outdated": outdated
    }


def get_git_metrics() -> Dict[str, Any]:
    """Get Git-based metrics."""
    # Count contributors
    contributors_output = run_command("git log --format='%ae' | sort -u | wc -l")
    total_contributors = int(contributors_output) if contributors_output.isdigit() else 0
    
    # Active contributors in last month
    month_contributors = run_command(
        "git log --since='1 month ago' --format='%ae' | sort -u | wc -l"
    )
    active_month = int(month_contributors) if month_contributors.isdigit() else 0
    
    # Active contributors in last quarter
    quarter_contributors = run_command(
        "git log --since='3 months ago' --format='%ae' | sort -u | wc -l"
    )
    active_quarter = int(quarter_contributors) if quarter_contributors.isdigit() else 0
    
    return {
        "total": total_contributors,
        "active_month": active_month,
        "active_quarter": active_quarter
    }


def get_test_metrics() -> Dict[str, Any]:
    """Get testing-related metrics."""
    # Count total tests
    test_count_output = run_command("python -m pytest --collect-only -q 2>/dev/null | grep -c '::' || echo 0")
    total_tests = int(test_count_output) if test_count_output.isdigit() else 0
    
    # Try to get coverage if available
    coverage_output = run_command("python -m pytest --cov=spike_transformer_compiler --cov-report=term-missing 2>/dev/null | grep TOTAL || echo ''")
    coverage_percent = 0
    if "TOTAL" in coverage_output:
        try:
            # Extract percentage from coverage output
            parts = coverage_output.split()
            for part in parts:
                if "%" in part:
                    coverage_percent = int(part.replace("%", ""))
                    break
        except (ValueError, IndexError):
            pass
    
    return {
        "total_tests": total_tests,
        "coverage_percent": coverage_percent
    }


def get_build_metrics() -> Dict[str, Any]:
    """Get build and CI metrics (placeholder for actual CI integration)."""
    # These would typically come from CI/CD system APIs
    return {
        "avg_duration_minutes": 0,
        "success_rate_percent": 100,  # Assume success until we have real data
        "runs_per_day": 0
    }


def update_metrics_file(metrics: Dict[str, Any]) -> None:
    """Update the project-metrics.json file with new values."""
    metrics_file = Path(".github/project-metrics.json")
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            current_metrics = json.load(f)
    else:
        current_metrics = {}
    
    # Update timestamp
    current_metrics.setdefault("project", {})["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    # Update repository metrics
    current_metrics.setdefault("repository_metrics", {}).update({
        "files": metrics["files"],
        "lines_of_code": metrics["lines_of_code"],
        "dependencies": metrics["dependencies"],
        "contributors": metrics["contributors"]
    })
    
    # Update health metrics
    health_metrics = current_metrics.setdefault("health_metrics", {})
    code_quality = health_metrics.setdefault("code_quality", {})
    test_coverage = code_quality.setdefault("test_coverage", {})
    test_coverage["current"] = metrics["test_metrics"]["coverage_percent"]
    
    # Update CI/CD metrics
    current_metrics.setdefault("ci_cd_metrics", {}).update({
        "test_execution": {
            "total_tests": metrics["test_metrics"]["total_tests"],
            "avg_execution_time_seconds": 0,  # Placeholder
            "flaky_test_rate": 0  # Placeholder
        },
        "build_pipeline": metrics["build_metrics"]
    })
    
    # Write updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(current_metrics, f, indent=2)
    
    print(f"Metrics updated: {metrics_file}")


def main():
    """Main metrics collection function."""
    print("Collecting repository metrics...")
    
    # Ensure we're in the repository root
    if not Path(".git").exists():
        print("Error: Not in a Git repository root")
        sys.exit(1)
    
    # Collect all metrics
    metrics = {
        "files": count_files_by_type(),
        "lines_of_code": count_lines_of_code(),
        "dependencies": count_dependencies(),
        "contributors": get_git_metrics(),
        "test_metrics": get_test_metrics(),
        "build_metrics": get_build_metrics()
    }
    
    # Print summary
    print("\nMetrics Summary:")
    print(f"  Source files: {metrics['files']['source_files']}")
    print(f"  Test files: {metrics['files']['test_files']}")
    print(f"  Lines of code: {metrics['lines_of_code']['source']}")
    print(f"  Test coverage: {metrics['test_metrics']['coverage_percent']}%")
    print(f"  Total tests: {metrics['test_metrics']['total_tests']}")
    print(f"  Contributors: {metrics['contributors']['total']}")
    print(f"  Dependencies: {metrics['dependencies']['production']} prod, {metrics['dependencies']['development']} dev")
    
    # Update metrics file
    update_metrics_file(metrics)
    
    print("\nMetrics collection completed successfully!")


if __name__ == "__main__":
    main()
