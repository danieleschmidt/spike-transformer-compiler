#!/usr/bin/env python3
"""
Intelligent Work Selection and Prioritization System
Selects the next best value item for autonomous execution
Terragon Labs - Autonomous SDLC Enhancement
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class WorkItem:
    """Selected work item ready for execution."""
    id: str
    title: str
    description: str
    category: str
    priority: str  # "critical", "high", "medium", "low"
    estimated_effort: float
    execution_strategy: str
    prerequisites: List[str]
    success_criteria: List[str]
    rollback_plan: str
    files_to_modify: List[str]
    testing_requirements: List[str]
    composite_score: float
    selected_at: str
    execution_notes: str

class IntelligentWorkSelector:
    """Intelligent system for selecting the next best value work item."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.selection_history: List[str] = []
        self.blocked_items: List[str] = []
        self.current_work_in_progress: List[str] = []
        
    def select_next_best_work(
        self, 
        scored_opportunities_file: str = ".terragon/scored-opportunities.json",
        max_concurrent: int = 1
    ) -> Optional[WorkItem]:
        """Select the next best value work item for execution."""
        print("🎯 Selecting next best value work item...")
        
        # Load scored opportunities
        with open(scored_opportunities_file, 'r') as f:
            opportunities = json.load(f)
        
        # Apply selection filters
        eligible_opportunities = self._filter_eligible_opportunities(opportunities)
        
        if not eligible_opportunities:
            print("⚠️  No eligible opportunities found")
            return None
        
        # Select best opportunity using selection algorithm
        selected_opp = self._apply_selection_algorithm(eligible_opportunities)
        
        if not selected_opp:
            print("⚠️  No suitable work item found")
            return None
        
        # Convert to work item with execution details
        work_item = self._create_work_item(selected_opp)
        
        print(f"✅ Selected: {work_item.title}")
        print(f"📊 Score: {work_item.composite_score:.1f} | Priority: {work_item.priority}")
        
        return work_item
    
    def _filter_eligible_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter opportunities to find eligible work items."""
        eligible = []
        
        for opp in opportunities:
            # Skip if already worked on
            if opp['id'] in self.selection_history:
                continue
            
            # Skip if blocked
            if opp['id'] in self.blocked_items:
                continue
            
            # Skip if dependencies not met
            if not self._dependencies_met(opp):
                continue
            
            # Skip if risk exceeds threshold
            if self._assess_risk(opp) > 0.7:  # Risk threshold
                continue
            
            # Skip if conflicts with work in progress
            if self._has_conflicts(opp):
                continue
            
            # Must meet minimum score threshold
            if opp['composite_score'] < 15:
                continue
            
            eligible.append(opp)
        
        return eligible
    
    def _dependencies_met(self, opp: Dict[str, Any]) -> bool:
        """Check if all dependencies for this opportunity are met."""
        
        # Critical dependencies for different categories
        dependency_checks = {
            'implementation': self._check_implementation_dependencies,
            'automation': self._check_automation_dependencies,
            'security': self._check_security_dependencies,
            'performance': self._check_performance_dependencies,
            'testing': self._check_testing_dependencies,
            'documentation': self._check_documentation_dependencies
        }
        
        category = opp.get('category', '')
        check_func = dependency_checks.get(category, lambda x: True)
        
        return check_func(opp)
    
    def _check_implementation_dependencies(self, opp: Dict[str, Any]) -> bool:
        """Check implementation-specific dependencies."""
        # For core compiler implementation, ensure basic structure exists
        if 'compiler.py' in opp.get('file_path', ''):
            # Check if basic classes are defined
            compiler_file = self.repo_root / "src" / "spike_transformer_compiler" / "compiler.py"
            if compiler_file.exists():
                return True
        return True
    
    def _check_automation_dependencies(self, opp: Dict[str, Any]) -> bool:
        """Check automation-specific dependencies."""
        # For CI/CD workflows, ensure repository structure is ready
        if 'workflow' in opp.get('title', '').lower():
            # Check for basic project files
            pyproject_exists = (self.repo_root / "pyproject.toml").exists()
            requirements_exist = (self.repo_root / "requirements.txt").exists()
            return pyproject_exists or requirements_exist
        return True
    
    def _check_security_dependencies(self, opp: Dict[str, Any]) -> bool:
        """Check security-specific dependencies."""
        # Security fixes usually don't have blockers
        return True
    
    def _check_performance_dependencies(self, opp: Dict[str, Any]) -> bool:
        """Check performance-specific dependencies."""
        # Performance work usually needs baseline implementation
        return True
    
    def _check_testing_dependencies(self, opp: Dict[str, Any]) -> bool:
        """Check testing-specific dependencies."""
        # Testing requires code to test
        return True
    
    def _check_documentation_dependencies(self, opp: Dict[str, Any]) -> bool:
        """Check documentation-specific dependencies."""
        # Documentation can usually be done independently
        return True
    
    def _assess_risk(self, opp: Dict[str, Any]) -> float:
        """Assess risk level of implementing this opportunity (0.0-1.0)."""
        base_risk = 0.3  # Default medium-low risk
        
        # Category-based risk adjustments
        risk_adjustments = {
            'security': -0.1,    # Security fixes are lower risk
            'implementation': 0.2, # Implementation changes are higher risk
            'automation': 0.1,   # Automation has moderate risk
            'performance': 0.15, # Performance changes need validation
            'technical_debt': 0.05, # Refactoring is moderate risk
            'testing': -0.05,    # Adding tests is low risk
            'documentation': -0.15 # Documentation is very low risk
        }
        
        risk = base_risk + risk_adjustments.get(opp.get('category', ''), 0)
        
        # Increase risk for complex changes
        if 'complex' in ' '.join(opp.get('keywords', [])).lower():
            risk += 0.2
        
        # Increase risk for core files
        core_files = ['compiler.py', 'backend.py', 'optimization.py']
        if opp.get('file_path') and any(cf in opp['file_path'] for cf in core_files):
            risk += 0.15
        
        # Increase risk for large changes
        if opp.get('estimated_effort', 0) > 6:  # Hours
            risk += 0.1
        
        return min(1.0, max(0.0, risk))
    
    def _has_conflicts(self, opp: Dict[str, Any]) -> bool:
        """Check if opportunity conflicts with current work in progress."""
        # For this autonomous implementation, we limit to 1 concurrent task
        if self.current_work_in_progress:
            return True
        
        # Check file-level conflicts
        opp_files = set()
        if opp.get('file_path'):
            opp_files.add(opp['file_path'])
        
        # If we had work in progress, we'd check for file overlaps here
        return False
    
    def _apply_selection_algorithm(self, opportunities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Apply intelligent selection algorithm to choose best opportunity."""
        
        if not opportunities:
            return None
        
        # Sort by composite score (already done, but ensure it)
        opportunities.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Apply strategic selection criteria
        for opp in opportunities:
            # Check execution readiness
            if self._is_execution_ready(opp):
                return opp
        
        # Fallback to highest scoring item if none are perfectly ready
        return opportunities[0] if opportunities else None
    
    def _is_execution_ready(self, opp: Dict[str, Any]) -> bool:
        """Check if opportunity is ready for immediate execution."""
        
        # Must have clear scope
        if not opp.get('title') or len(opp['title']) < 10:
            return False
        
        # Must have actionable description
        if not opp.get('description') or len(opp['description']) < 20:
            return False
        
        # Implementation items need clear location
        if opp['category'] == 'implementation':
            if not opp.get('file_path'):
                return False
        
        # Security items should be immediately actionable
        if opp['category'] == 'security':
            return True
        
        # Automation items need clear requirements
        if opp['category'] == 'automation':
            if 'missing' in opp.get('title', '').lower():
                return True
        
        return True
    
    def _create_work_item(self, opp: Dict[str, Any]) -> WorkItem:
        """Convert opportunity to executable work item."""
        
        # Determine priority level
        priority = self._determine_priority(opp)
        
        # Create execution strategy
        execution_strategy = self._create_execution_strategy(opp)
        
        # Identify prerequisites
        prerequisites = self._identify_prerequisites(opp)
        
        # Define success criteria
        success_criteria = self._define_success_criteria(opp)
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(opp)
        
        # Identify files to modify
        files_to_modify = self._identify_files_to_modify(opp)
        
        # Define testing requirements
        testing_requirements = self._define_testing_requirements(opp)
        
        # Generate execution notes
        execution_notes = self._generate_execution_notes(opp)
        
        return WorkItem(
            id=opp['id'],
            title=opp['title'],
            description=opp['description'],
            category=opp['category'],
            priority=priority,
            estimated_effort=opp.get('estimated_effort', 2.0),
            execution_strategy=execution_strategy,
            prerequisites=prerequisites,
            success_criteria=success_criteria,
            rollback_plan=rollback_plan,
            files_to_modify=files_to_modify,
            testing_requirements=testing_requirements,
            composite_score=opp['composite_score'],
            selected_at=datetime.now().isoformat(),
            execution_notes=execution_notes
        )
    
    def _determine_priority(self, opp: Dict[str, Any]) -> str:
        """Determine priority level based on score and category."""
        score = opp['composite_score']
        category = opp['category']
        
        # Security is always high priority
        if category == 'security':
            return "critical"
        
        # Score-based priority
        if score >= 200:
            return "critical"
        elif score >= 100:
            return "high"
        elif score >= 50:
            return "medium"
        else:
            return "low"
    
    def _create_execution_strategy(self, opp: Dict[str, Any]) -> str:
        """Create specific execution strategy for the opportunity."""
        
        category = opp['category']
        strategies = {
            'security': "1. Analyze security issue 2. Apply fix 3. Verify security scan passes 4. Run full test suite",
            'implementation': "1. Analyze requirements 2. Implement functionality 3. Add tests 4. Update documentation",
            'automation': "1. Research best practices 2. Create configuration 3. Test automation 4. Document process",
            'performance': "1. Establish baseline 2. Implement optimization 3. Measure improvement 4. Validate no regressions",
            'technical_debt': "1. Analyze current code 2. Plan refactoring 3. Implement changes 4. Verify functionality unchanged",
            'testing': "1. Analyze testing gaps 2. Write tests 3. Verify coverage improvement 4. Ensure tests pass",
            'documentation': "1. Analyze documentation needs 2. Write content 3. Review for accuracy 4. Integrate with existing docs"
        }
        
        return strategies.get(category, "1. Analyze requirements 2. Implement solution 3. Test changes 4. Document work")
    
    def _identify_prerequisites(self, opp: Dict[str, Any]) -> List[str]:
        """Identify prerequisites for this work item."""
        prerequisites = []
        
        category = opp['category']
        
        if category == 'implementation':
            prerequisites.append("Development environment setup")
            prerequisites.append("Understanding of existing codebase")
        
        if category == 'automation':
            prerequisites.append("Access to repository settings")
            prerequisites.append("Understanding of CI/CD best practices")
        
        if category == 'security':
            prerequisites.append("Security scanning tools available")
            prerequisites.append("Understanding of vulnerability details")
        
        if category == 'performance':
            prerequisites.append("Performance monitoring tools") 
            prerequisites.append("Baseline performance metrics")
        
        return prerequisites
    
    def _define_success_criteria(self, opp: Dict[str, Any]) -> List[str]:
        """Define success criteria for this work item."""
        criteria = []
        
        category = opp['category']
        
        # Common criteria
        criteria.append("All tests pass")
        criteria.append("No regressions introduced")
        
        # Category-specific criteria
        if category == 'security':
            criteria.append("Security scan passes")
            criteria.append("No new vulnerabilities introduced")
        
        if category == 'implementation':
            criteria.append("Functionality works as expected")
            criteria.append("Code coverage maintained or improved")
        
        if category == 'automation':
            criteria.append("Automation runs successfully")
            criteria.append("Process documented")
        
        if category == 'performance':
            criteria.append("Performance improvement measurable")
            criteria.append("No degradation in other metrics")
        
        if category == 'technical_debt':
            criteria.append("Code complexity reduced")
            criteria.append("Maintainability improved")
        
        return criteria
    
    def _create_rollback_plan(self, opp: Dict[str, Any]) -> str:
        """Create rollback plan in case of issues."""
        
        if opp.get('file_path'):
            return f"1. Revert changes to {opp['file_path']} 2. Run tests to verify rollback 3. Check for any dependent changes"
        
        category = opp['category']
        
        rollback_plans = {
            'security': "1. Revert security fix 2. Re-run security scan 3. Document temporary mitigation",
            'implementation': "1. Revert code changes 2. Run full test suite 3. Verify system stability",
            'automation': "1. Disable automation 2. Revert configuration changes 3. Resume manual process",
            'performance': "1. Revert optimization 2. Verify performance baseline restored 3. Check for side effects",
            'technical_debt': "1. Revert refactoring 2. Run regression tests 3. Ensure functionality unchanged",
            'testing': "1. Remove failing tests 2. Verify existing tests still pass 3. Document test gaps",
            'documentation': "1. Revert documentation changes 2. Verify links still work 3. Check formatting"
        }
        
        return rollback_plans.get(category, "1. Revert all changes 2. Run verification tests 3. Document issues encountered")
    
    def _identify_files_to_modify(self, opp: Dict[str, Any]) -> List[str]:
        """Identify files that will likely be modified."""
        files = []
        
        # If specific file mentioned, add it
        if opp.get('file_path'):
            files.append(opp['file_path'])
        
        category = opp['category']
        
        # Category-specific file patterns
        if category == 'automation' and 'workflow' in opp.get('title', '').lower():
            files.append('.github/workflows/ci.yml')
        
        if category == 'security' and 'dependencies' in opp.get('title', '').lower():
            files.extend(['requirements.txt', 'requirements-dev.txt'])
        
        if category == 'testing':
            files.append('tests/')
        
        if category == 'documentation':
            files.append('README.md')
        
        return files
    
    def _define_testing_requirements(self, opp: Dict[str, Any]) -> List[str]:
        """Define testing requirements for this work item."""
        requirements = []
        
        category = opp['category']
        
        # Common requirements
        requirements.append("Run existing test suite")
        
        # Category-specific requirements
        if category == 'implementation':
            requirements.append("Add unit tests for new functionality")
            requirements.append("Add integration tests if needed")
        
        if category == 'security':
            requirements.append("Run security scan")
            requirements.append("Test security fix effectiveness")
        
        if category == 'automation':
            requirements.append("Test automation pipeline")
            requirements.append("Verify automation triggers correctly")
        
        if category == 'performance':
            requirements.append("Run performance benchmarks")
            requirements.append("Compare before/after metrics")
        
        return requirements
    
    def _generate_execution_notes(self, opp: Dict[str, Any]) -> str:
        """Generate specific execution notes and tips."""
        
        category = opp['category']
        
        notes = {
            'security': "Handle security issues with care. Verify fix addresses root cause, not just symptoms.",
            'implementation': "Focus on clean, maintainable code. Follow existing patterns and conventions.",
            'automation': "Test automation thoroughly before enabling. Document any manual steps remaining.",
            'performance': "Measure twice, optimize once. Ensure optimizations don't compromise correctness.",
            'technical_debt': "Refactor incrementally. Maintain functionality while improving structure.",
            'testing': "Write clear, focused tests. Ensure tests are maintainable and valuable.",
            'documentation': "Keep documentation concise and up-to-date. Focus on user needs."
        }
        
        base_note = notes.get(category, "Follow best practices and maintain code quality.")
        
        # Add specific notes based on opportunity details
        if 'compiler' in opp.get('title', '').lower():
            base_note += " This is core compiler functionality - ensure changes are well-tested."
        
        if opp.get('composite_score', 0) > 150:
            base_note += " High-value item - prioritize quality over speed."
        
        return base_note
    
    def export_work_item(self, work_item: WorkItem, output_file: str) -> None:
        """Export selected work item to file."""
        work_dict = {
            'id': work_item.id,
            'title': work_item.title,
            'description': work_item.description,
            'category': work_item.category,
            'priority': work_item.priority,
            'estimated_effort': work_item.estimated_effort,
            'execution_strategy': work_item.execution_strategy,
            'prerequisites': work_item.prerequisites,
            'success_criteria': work_item.success_criteria,
            'rollback_plan': work_item.rollback_plan,
            'files_to_modify': work_item.files_to_modify,
            'testing_requirements': work_item.testing_requirements,
            'composite_score': work_item.composite_score,
            'selected_at': work_item.selected_at,
            'execution_notes': work_item.execution_notes
        }
        
        with open(output_file, 'w') as f:
            json.dump(work_dict, f, indent=2)
        
        print(f"📁 Exported work item to {output_file}")

def main():
    """Main execution function."""
    selector = IntelligentWorkSelector()
    
    # Select next best work item
    work_item = selector.select_next_best_work()
    
    if work_item:
        # Export for execution
        selector.export_work_item(work_item, ".terragon/selected-work-item.json")
        
        # Print execution plan
        print(f"\n📋 Execution Plan:")
        print(f"Strategy: {work_item.execution_strategy}")
        print(f"Prerequisites: {', '.join(work_item.prerequisites)}")
        print(f"Success Criteria: {', '.join(work_item.success_criteria)}")
        print(f"Files to Modify: {', '.join(work_item.files_to_modify)}")
        print(f"Notes: {work_item.execution_notes}")
    else:
        print("❌ No suitable work item found for execution")

if __name__ == "__main__":
    main()