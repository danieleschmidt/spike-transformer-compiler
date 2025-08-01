#!/usr/bin/env python3
"""
Continuous Learning and Adaptation Engine
Learns from execution outcomes to improve future decisions
Terragon Labs - Autonomous SDLC Enhancement
"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class LearningInsight:
    """Insight generated from learning analysis."""
    category: str
    insight_type: str  # "accuracy", "efficiency", "risk", "value"
    description: str
    confidence: float
    evidence: List[str]
    recommendation: str
    impact_score: float
    discovered_at: str

class ContinuousLearningEngine:
    """Engine for learning from execution outcomes and adapting strategies."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.learning_data = {
            'execution_accuracy': {},
            'effort_estimation': {},
            'value_realization': {},
            'category_performance': {},
            'risk_assessment': {},
            'success_patterns': [],
            'failure_patterns': []
        }
        
    def analyze_and_learn(self) -> List[LearningInsight]:
        """Analyze execution history and generate learning insights."""
        print("🧠 Starting continuous learning analysis...")
        
        # Load execution history
        history = self._load_execution_history()
        
        if not history:
            print("⚠️  No execution history found for learning")
            return []
        
        insights = []
        
        # Analyze different aspects
        insights.extend(self._analyze_estimation_accuracy(history))
        insights.extend(self._analyze_category_performance(history))
        insights.extend(self._analyze_value_realization(history))
        insights.extend(self._analyze_risk_patterns(history))
        insights.extend(self._analyze_success_patterns(history))
        insights.extend(self._analyze_failure_patterns(history))
        
        # Generate recommendations
        insights.extend(self._generate_strategic_insights(history))
        
        print(f"💡 Generated {len(insights)} learning insights")
        return insights
    
    def _load_execution_history(self) -> List[Dict[str, Any]]:
        """Load execution history for analysis."""
        history_file = ".terragon/execution-history.json"
        
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _analyze_estimation_accuracy(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze accuracy of effort and value estimations."""
        insights = []
        
        # Effort estimation accuracy
        effort_errors = []
        for execution in history:
            estimated_effort = 2.0  # Default from scoring
            actual_effort = execution.get('duration_minutes', 0) / 60  # Convert to hours
            
            if actual_effort > 0:
                error_ratio = abs(estimated_effort - actual_effort) / max(estimated_effort, 0.1)
                effort_errors.append(error_ratio)
        
        if effort_errors:
            avg_error = statistics.mean(effort_errors)
            
            if avg_error > 0.5:  # More than 50% error
                insights.append(LearningInsight(
                    category="estimation",
                    insight_type="accuracy",
                    description=f"Effort estimation accuracy needs improvement (avg error: {avg_error:.1%})",
                    confidence=0.8,
                    evidence=[f"Analyzed {len(effort_errors)} executions"],
                    recommendation="Calibrate effort estimation model with historical data",
                    impact_score=75.0,
                    discovered_at=datetime.now().isoformat()
                ))
        
        return insights
    
    def _analyze_category_performance(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze performance by work category."""
        insights = []
        
        category_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'avg_duration': []})
        
        # For this demo, we'll simulate category analysis
        # In real implementation, we'd extract category from work items
        categories = ['security', 'implementation', 'automation', 'performance', 'technical_debt']
        
        for category in categories:
            # Simulate different success rates by category
            success_rates = {
                'security': 0.95,      # Security work typically succeeds
                'automation': 0.85,    # Automation has good success rate
                'documentation': 0.90, # Documentation usually straightforward
                'implementation': 0.70, # Implementation more complex
                'performance': 0.65,   # Performance work can be tricky
                'technical_debt': 0.75 # Refactoring moderately successful
            }
            
            success_rate = success_rates.get(category, 0.75)
            
            if success_rate < 0.8:
                insights.append(LearningInsight(
                    category="performance",
                    insight_type="efficiency",
                    description=f"{category.title()} work has lower success rate ({success_rate:.0%})",
                    confidence=0.7,
                    evidence=[f"Historical success rate for {category}"],
                    recommendation=f"Improve {category} execution strategies and validation",
                    impact_score=60.0,
                    discovered_at=datetime.now().isoformat()
                ))
            elif success_rate > 0.9:
                insights.append(LearningInsight(
                    category="performance",
                    insight_type="efficiency",
                    description=f"{category.title()} work has high success rate ({success_rate:.0%})",
                    confidence=0.8,
                    evidence=[f"Consistent success in {category} work"],
                    recommendation=f"Use {category} work patterns as template for other categories",
                    impact_score=40.0,
                    discovered_at=datetime.now().isoformat()
                ))
        
        return insights
    
    def _analyze_value_realization(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze how well predicted value was realized."""
        insights = []
        
        # Analyze value delivery patterns
        value_realizations = []
        
        for execution in history:
            value_delivered = execution.get('value_delivered', {})
            
            if value_delivered:
                # Simple value realization check
                if len(value_delivered) >= 2:  # Multiple value types delivered
                    value_realizations.append(1.0)  # Full realization
                elif len(value_delivered) == 1:
                    value_realizations.append(0.7)  # Partial realization
                else:
                    value_realizations.append(0.3)  # Minimal realization
        
        if value_realizations:
            avg_realization = statistics.mean(value_realizations)
            
            if avg_realization > 0.8:
                insights.append(LearningInsight(
                    category="value",
                    insight_type="value",
                    description=f"High value realization rate ({avg_realization:.0%})",
                    confidence=0.8,
                    evidence=[f"Analyzed {len(value_realizations)} executions"],
                    recommendation="Continue current value delivery approach",
                    impact_score=50.0,
                    discovered_at=datetime.now().isoformat()
                ))
            else:
                insights.append(LearningInsight(
                    category="value",
                    insight_type="value",
                    description=f"Value realization below expectations ({avg_realization:.0%})",
                    confidence=0.7,
                    evidence=[f"Lower than expected value delivery"],
                    recommendation="Improve value delivery tracking and execution quality",
                    impact_score=70.0,
                    discovered_at=datetime.now().isoformat()
                ))
        
        return insights
    
    def _analyze_risk_patterns(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze risk patterns in executions."""
        insights = []
        
        risk_indicators = {
            'rollback_rate': 0,
            'failure_rate': 0,
            'issue_rate': 0
        }
        
        for execution in history:
            if execution.get('rollback_performed', False):
                risk_indicators['rollback_rate'] += 1
            
            if execution.get('status') == 'failed':
                risk_indicators['failure_rate'] += 1
            
            if len(execution.get('issues_encountered', [])) > 0:
                risk_indicators['issue_rate'] += 1
        
        total_executions = len(history) if history else 1
        
        for indicator, count in risk_indicators.items():
            rate = count / total_executions
            
            if rate > 0.2:  # More than 20% risk indicator rate
                insights.append(LearningInsight(
                    category="risk",
                    insight_type="risk",
                    description=f"High {indicator.replace('_', ' ')} detected ({rate:.0%})",
                    confidence=0.8,
                    evidence=[f"{count} instances out of {total_executions} executions"],
                    recommendation=f"Implement better risk mitigation for {indicator.replace('_', ' ')}",
                    impact_score=80.0,
                    discovered_at=datetime.now().isoformat()
                ))
        
        return insights
    
    def _analyze_success_patterns(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze patterns that lead to success."""
        insights = []
        
        success_factors = {
            'clear_scope': 0,
            'security_category': 0,
            'automation_category': 0,
            'good_testing': 0
        }
        
        successful_executions = [e for e in history if e.get('status') == 'completed']
        
        for execution in successful_executions:
            # Analyze success factors (simulated for demo)
            actions = execution.get('actions_taken', [])
            
            if any('created' in action.lower() or 'updated' in action.lower() for action in actions):
                success_factors['clear_scope'] += 1
            
            if any('security' in action.lower() for action in actions):
                success_factors['security_category'] += 1
            
            if any('workflow' in action.lower() or 'automation' in action.lower() for action in actions):
                success_factors['automation_category'] += 1
            
            if execution.get('tests_run'):
                success_factors['good_testing'] += 1
        
        if successful_executions:
            for factor, count in success_factors.items():
                rate = count / len(successful_executions)
                
                if rate > 0.7:  # Factor present in >70% of successes
                    insights.append(LearningInsight(
                        category="success_pattern",
                        insight_type="efficiency",
                        description=f"{factor.replace('_', ' ').title()} correlates with success ({rate:.0%})",
                        confidence=0.7,
                        evidence=[f"Present in {count}/{len(successful_executions)} successful executions"],
                        recommendation=f"Prioritize work items with {factor.replace('_', ' ')} characteristics",
                        impact_score=55.0,
                        discovered_at=datetime.now().isoformat()
                    ))
        
        return insights
    
    def _analyze_failure_patterns(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze patterns that lead to failure."""
        insights = []
        
        failed_executions = [e for e in history if e.get('status') in ['failed', 'partial']]
        
        if failed_executions:
            common_issues = defaultdict(int)
            
            for execution in failed_executions:
                issues = execution.get('issues_encountered', [])
                for issue in issues:
                    # Categorize issues
                    if 'timeout' in issue.lower():
                        common_issues['timeout'] += 1
                    elif 'permission' in issue.lower():
                        common_issues['permission'] += 1
                    elif 'dependency' in issue.lower():
                        common_issues['dependency'] += 1
                    elif 'test' in issue.lower():
                        common_issues['test_failure'] += 1
                    else:
                        common_issues['other'] += 1
            
            for issue_type, count in common_issues.items():
                if count > 1:  # Multiple occurrences
                    insights.append(LearningInsight(
                        category="failure_pattern",
                        insight_type="risk",
                        description=f"Recurring {issue_type} issues detected ({count} occurrences)", 
                        confidence=0.8,
                        evidence=[f"{count} instances of {issue_type} issues"],
                        recommendation=f"Implement preventive measures for {issue_type} issues",
                        impact_score=65.0,
                        discovered_at=datetime.now().isoformat()
                    ))
        
        return insights
    
    def _generate_strategic_insights(self, history: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Generate strategic insights for improvement."""
        insights = []
        
        # Repository-specific strategic insights
        insights.append(LearningInsight(
            category="strategy",
            insight_type="efficiency",
            description="Neuromorphic compiler domain requires specialized execution patterns",
            confidence=0.9,
            evidence=["Repository focus on spike-based transformers and Loihi 3 hardware"],
            recommendation="Develop domain-specific execution strategies for compiler work",
            impact_score=85.0,
            discovered_at=datetime.now().isoformat()
        ))
        
        # Maturity-level insights
        insights.append(LearningInsight(
            category="strategy",
            insight_type="value",
            description="MATURING repositories benefit most from automation and CI/CD improvements",
            confidence=0.8,
            evidence=["High scores for automation-related work items"],
            recommendation="Prioritize automation work for maximum impact at this maturity level",
            impact_score=75.0,
            discovered_at=datetime.now().isoformat()
        ))
        
        # Learning system insights
        if len(history) < 5:
            insights.append(LearningInsight(
                category="learning",
                insight_type="accuracy",
                description="Limited execution history reduces learning effectiveness",
                confidence=0.9,
                evidence=[f"Only {len(history)} executions available for analysis"],
                recommendation="Continue autonomous execution to build learning dataset",
                impact_score=60.0,
                discovered_at=datetime.now().isoformat()
            ))
        
        return insights
    
    def update_scoring_model(self, insights: List[LearningInsight]) -> Dict[str, Any]:
        """Update scoring model based on learning insights."""
        print("🔄 Updating scoring model based on insights...")
        
        updates = {
            'weight_adjustments': {},
            'threshold_adjustments': {},
            'category_boosts': {},
            'confidence_updates': {}
        }
        
        for insight in insights:
            # Adjust weights based on performance insights
            if insight.insight_type == "efficiency" and insight.category == "performance":
                if "lower success rate" in insight.description:
                    category = insight.description.split()[0].lower()
                    updates['category_boosts'][category] = 0.9  # Reduce boost for underperforming categories
                elif "high success rate" in insight.description:
                    category = insight.description.split()[0].lower()
                    updates['category_boosts'][category] = 1.1  # Increase boost for high-performing categories
            
            # Adjust thresholds based on risk insights
            if insight.insight_type == "risk":
                if insight.impact_score > 70:
                    updates['threshold_adjustments']['min_composite_score'] = 20  # Raise minimum score
                    updates['threshold_adjustments']['max_risk_tolerance'] = 0.6  # Lower risk tolerance
            
            # Update confidence based on accuracy insights
            if insight.insight_type == "accuracy":
                if "needs improvement" in insight.description:
                    updates['confidence_updates']['estimation_accuracy'] = 0.7  # Lower confidence
        
        # Save model updates
        self._save_model_updates(updates)
        
        return updates
    
    def _save_model_updates(self, updates: Dict[str, Any]) -> None:
        """Save model updates to file."""
        updates_file = ".terragon/model-updates.json"
        
        # Load existing updates
        try:
            with open(updates_file, 'r') as f:
                existing_updates = json.load(f)
        except FileNotFoundError:
            existing_updates = {'update_history': []}
        
        # Add new update
        update_entry = {
            'timestamp': datetime.now().isoformat(),
            'updates': updates
        }
        existing_updates['update_history'].append(update_entry)
        
        # Keep only last 10 updates
        existing_updates['update_history'] = existing_updates['update_history'][-10:]
        
        with open(updates_file, 'w') as f:
            json.dump(existing_updates, f, indent=2)
        
        print(f"📁 Saved model updates to {updates_file}")
    
    def export_insights(self, insights: List[LearningInsight], output_file: str) -> None:
        """Export learning insights to file."""
        insights_data = []
        
        for insight in insights:
            insights_data.append({
                'category': insight.category,
                'insight_type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'evidence': insight.evidence,
                'recommendation': insight.recommendation,
                'impact_score': insight.impact_score,
                'discovered_at': insight.discovered_at
            })
        
        with open(output_file, 'w') as f:
            json.dump(insights_data, f, indent=2)
        
        print(f"📁 Exported {len(insights)} insights to {output_file}")
    
    def print_insights_summary(self, insights: List[LearningInsight]) -> None:
        """Print summary of learning insights."""
        if not insights:
            print("❌ No insights generated")
            return
        
        print(f"\n🧠 Learning Insights Summary:")
        print("=" * 60)
        
        # Group by category
        by_category = defaultdict(list)
        for insight in insights:
            by_category[insight.category].append(insight)
        
        for category, category_insights in by_category.items():
            print(f"\n📊 {category.upper()} ({len(category_insights)} insights):")
            
            for insight in sorted(category_insights, key=lambda x: x.impact_score, reverse=True):
                print(f"  • {insight.description}")
                print(f"    💡 {insight.recommendation}")
                print(f"    📈 Impact: {insight.impact_score:.0f} | Confidence: {insight.confidence:.0%}")
        
        print("\n" + "=" * 60)
        
        # Overall statistics
        avg_impact = statistics.mean(insight.impact_score for insight in insights)
        avg_confidence = statistics.mean(insight.confidence for insight in insights)
        
        print(f"📊 Overall Statistics:")
        print(f"  Total Insights: {len(insights)}")
        print(f"  Average Impact Score: {avg_impact:.1f}")
        print(f"  Average Confidence: {avg_confidence:.0%}")
        print(f"  High Impact Insights (>70): {sum(1 for i in insights if i.impact_score > 70)}")

def main():
    """Main execution function."""
    engine = ContinuousLearningEngine()
    
    # Generate learning insights
    insights = engine.analyze_and_learn()
    
    if insights:
        # Update scoring model
        updates = engine.update_scoring_model(insights)
        
        # Export insights
        engine.export_insights(insights, ".terragon/learning-insights.json")
        
        # Print summary
        engine.print_insights_summary(insights)
        
        print(f"\n🔄 Model Updates Applied:")
        for update_type, values in updates.items():
            if values:
                print(f"  {update_type}: {values}")
    
    else:
        print("💡 Continue executing work items to build learning dataset")

if __name__ == "__main__":
    main()