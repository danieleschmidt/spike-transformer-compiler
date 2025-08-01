#!/usr/bin/env python3
"""
Adaptive Scoring Engine for Value-Driven Work Prioritization
Combines WSJF, ICE, and Technical Debt scoring with domain-specific optimizations
Terragon Labs - Autonomous SDLC Enhancement
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ScoredOpportunity:
    """Value opportunity with comprehensive scoring."""
    # Base opportunity data
    id: str
    title: str
    description: str
    category: str
    source: str
    file_path: Optional[str]
    line_number: Optional[int]
    keywords: List[str]
    
    # Scoring components
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    
    # WSJF breakdown
    user_business_value: int
    time_criticality: int
    risk_reduction: int
    opportunity_enablement: int
    job_size: float
    
    # ICE breakdown
    impact: int
    confidence: float
    ease: int
    
    # Additional metrics
    domain_boost: float
    priority_rank: int
    estimated_value_delivery: float  # $ or time saved
    
    # Metadata
    scored_at: str
    confidence_level: str

class AdaptiveScoringEngine:
    """Advanced scoring engine with multiple methodologies."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config = self._load_default_config()
        self.repo_maturity = "maturing"  # Based on our assessment
        self.domain = "neuromorphic-computing"
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load scoring configuration."""
        return {
            'scoring': {
                'weights': {
                    'maturing': {
                        'wsjf': 0.6,
                        'ice': 0.1,
                        'technical_debt': 0.2,
                        'security': 0.1
                    }
                },
                'thresholds': {
                    'min_composite_score': 15,
                    'security_boost_multiplier': 2.5,
                    'compliance_boost_multiplier': 2.0,
                    'performance_boost_multiplier': 1.5
                },
                'domain_boosts': {
                    'hardware_optimization': 1.8,
                    'energy_efficiency': 1.6,
                    'compilation_performance': 1.4,
                    'spike_encoding': 1.3
                }
            }
        }
    
    def score_opportunities(self, opportunities_file: str) -> List[ScoredOpportunity]:
        """Score all discovered opportunities."""
        print("🎯 Starting comprehensive opportunity scoring...")
        
        # Load discovered opportunities
        with open(opportunities_file, 'r') as f:
            raw_opportunities = json.load(f)
        
        scored_opportunities = []
        
        for opp in raw_opportunities:
            scored_opp = self._score_single_opportunity(opp)
            scored_opportunities.append(scored_opp)
        
        # Sort by composite score (descending)
        scored_opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign priority ranks
        for i, opp in enumerate(scored_opportunities, 1):
            opp.priority_rank = i
        
        print(f"📊 Scored {len(scored_opportunities)} opportunities")
        return scored_opportunities
    
    def _score_single_opportunity(self, opp: Dict[str, Any]) -> ScoredOpportunity:
        """Score a single opportunity using all methodologies."""
        
        # Calculate WSJF components
        wsjf_components = self._calculate_wsjf_components(opp)
        wsjf_score = self._calculate_wsjf(wsjf_components)
        
        # Calculate ICE components  
        ice_components = self._calculate_ice_components(opp)
        ice_score = self._calculate_ice(ice_components)
        
        # Technical debt score (already calculated during discovery)
        tech_debt_score = opp.get('technical_debt_score', 50)
        
        # Apply domain-specific boosts
        domain_boost = self._calculate_domain_boost(opp)
        
        # Calculate composite score using adaptive weights
        composite_score = self._calculate_composite_score(
            wsjf_score, ice_score, tech_debt_score, opp, domain_boost
        )
        
        # Estimate value delivery
        value_delivery = self._estimate_value_delivery(opp, composite_score)
        
        return ScoredOpportunity(
            # Base data
            id=opp['id'],
            title=opp['title'], 
            description=opp['description'],
            category=opp['category'],
            source=opp['source'],
            file_path=opp.get('file_path'),
            line_number=opp.get('line_number'),
            keywords=opp.get('keywords', []),
            
            # Scores
            wsjf_score=wsjf_score,
            ice_score=ice_score,
            technical_debt_score=tech_debt_score,
            composite_score=composite_score,
            
            # WSJF breakdown
            user_business_value=wsjf_components['user_business_value'],
            time_criticality=wsjf_components['time_criticality'],
            risk_reduction=wsjf_components['risk_reduction'],
            opportunity_enablement=wsjf_components['opportunity_enablement'],
            job_size=wsjf_components['job_size'],
            
            # ICE breakdown
            impact=ice_components['impact'],
            confidence=ice_components['confidence'],
            ease=ice_components['ease'],
            
            # Additional metrics
            domain_boost=domain_boost,
            priority_rank=0,  # Will be set after sorting
            estimated_value_delivery=value_delivery,
            
            # Metadata
            scored_at=datetime.now().isoformat(),
            confidence_level=self._determine_confidence_level(opp)
        )
    
    def _calculate_wsjf_components(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate WSJF (Weighted Shortest Job First) components."""
        
        # User/Business Value (1-20 scale)
        business_value_map = {
            'security': 18,      # Critical for production systems
            'implementation': 15, # Core functionality
            'automation': 12,    # Development efficiency
            'performance': 10,   # User experience
            'technical_debt': 8, # Long-term maintainability
            'testing': 7,        # Quality assurance
            'documentation': 5   # User support
        }
        base_value = business_value_map.get(opp['category'], 10)
        
        # Boost for neuromorphic domain relevance
        neuromorphic_keywords = ['loihi', 'spike', 'neuromorphic', 'compiler', 'hardware']
        if any(kw in ' '.join(opp.get('keywords', [])).lower() for kw in neuromorphic_keywords):
            base_value = min(20, int(base_value * 1.3))
        
        # Time Criticality (1-20 scale) 
        criticality_map = {
            'security': 20,      # Immediate fix needed
            'implementation': 12, # Blocks other work
            'automation': 10,    # Slows development
            'performance': 8,    # Affects user experience
            'technical_debt': 6, # Accumulates over time
            'testing': 8,        # Quality impacts
            'documentation': 4   # Lower urgency
        }
        time_criticality = criticality_map.get(opp['category'], 8)
        
        # Boost for core files
        core_files = ['compiler.py', 'backend.py', 'optimization.py']
        if opp.get('file_path') and any(cf in opp['file_path'] for cf in core_files):
            time_criticality = min(20, int(time_criticality * 1.4))
        
        # Risk Reduction/Opportunity Enablement (1-20 scale)
        risk_reduction_map = {
            'security': 18,      # High risk mitigation
            'implementation': 14, # Enables other features
            'automation': 12,    # Reduces human error
            'performance': 6,    # Moderate risk
            'technical_debt': 10, # Prevents future problems
            'testing': 8,        # Quality assurance
            'documentation': 4   # Low risk
        }
        risk_reduction = risk_reduction_map.get(opp['category'], 8)
        
        # Opportunity Enablement - what this unlocks
        enablement_boost = 0
        enabling_keywords = ['core', 'foundation', 'framework', 'infrastructure']
        if any(kw in ' '.join(opp.get('keywords', [])).lower() for kw in enabling_keywords):
            enablement_boost = 5
        
        opportunity_enablement = min(20, risk_reduction + enablement_boost)
        
        # Job Size (effort in ideal days)
        job_size = opp.get('estimated_effort', 2.0)
        
        # Adjust job size based on complexity indicators
        complexity_keywords = ['refactor', 'large', 'complex', 'multiple']
        if any(kw in ' '.join(opp.get('keywords', [])).lower() for kw in complexity_keywords):
            job_size *= 1.5
        
        return {
            'user_business_value': base_value,
            'time_criticality': time_criticality,
            'risk_reduction': risk_reduction,
            'opportunity_enablement': opportunity_enablement,
            'job_size': job_size
        }
    
    def _calculate_wsjf(self, components: Dict[str, Any]) -> float:
        """Calculate WSJF score: Cost of Delay / Job Size."""
        
        cost_of_delay = (
            components['user_business_value'] +
            components['time_criticality'] +
            components['risk_reduction'] +
            components['opportunity_enablement']
        )
        
        job_size = max(0.5, components['job_size'])  # Avoid division by zero
        
        wsjf = cost_of_delay / job_size
        return round(wsjf, 2)
    
    def _calculate_ice_components(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ICE (Impact, Confidence, Ease) components."""
        
        # Impact (1-10 scale) - similar to business value but normalized
        impact_map = {
            'security': 10,      # Maximum impact
            'implementation': 9, # High impact
            'automation': 7,     # Good impact  
            'performance': 6,    # Medium impact
            'technical_debt': 5, # Gradual impact
            'testing': 6,        # Quality impact
            'documentation': 4   # Lower impact
        }
        impact = impact_map.get(opp['category'], 5)
        
        # Confidence (0.1-1.0 scale) - how sure we are about the solution
        confidence_map = {
            'security': 0.9,     # Well-defined solutions
            'implementation': 0.7, # Some unknowns
            'automation': 0.8,   # Established practices
            'performance': 0.6,  # Needs measurement
            'technical_debt': 0.7, # Clear refactoring
            'testing': 0.8,      # Standard approaches
            'documentation': 0.9 # Straightforward
        }
        base_confidence = confidence_map.get(opp['category'], 0.7)
        
        # Reduce confidence for complex tasks
        if 'complex' in ' '.join(opp.get('keywords', [])).lower():
            base_confidence *= 0.8
        
        # Boost confidence for well-defined tasks
        if any(kw in opp.get('title', '').lower() for kw in ['missing', 'add', 'create']):
            base_confidence = min(1.0, base_confidence * 1.1)
        
        # Ease (1-10 scale) - how easy to implement
        ease_map = {
            'security': 6,       # May need expertise
            'implementation': 4, # Complex coding
            'automation': 7,     # Standard tools
            'performance': 5,    # Requires analysis
            'technical_debt': 6, # Refactoring effort
            'testing': 8,        # Well-understood
            'documentation': 9   # Usually straightforward
        }
        ease = ease_map.get(opp['category'], 6)
        
        return {
            'impact': impact,
            'confidence': base_confidence,
            'ease': ease
        }
    
    def _calculate_ice(self, components: Dict[str, Any]) -> float:
        """Calculate ICE score: Impact × Confidence × Ease."""
        ice = components['impact'] * components['confidence'] * components['ease']
        return round(ice, 2)
    
    def _calculate_domain_boost(self, opp: Dict[str, Any]) -> float:
        """Calculate domain-specific boost factor."""
        boost = 1.0
        
        # Neuromorphic domain keywords
        domain_keywords = {
            'loihi': 1.8,
            'neuromorphic': 1.6,
            'spike': 1.4,
            'compiler': 1.3,
            'hardware': 1.2,
            'energy': 1.3,
            'optimization': 1.2
        }
        
        text = ' '.join([
            opp.get('title', ''),
            opp.get('description', ''),
            ' '.join(opp.get('keywords', []))
        ]).lower()
        
        for keyword, multiplier in domain_keywords.items():
            if keyword in text:
                boost = max(boost, multiplier)
        
        return boost
    
    def _calculate_composite_score(
        self, 
        wsjf_score: float, 
        ice_score: float, 
        tech_debt_score: float, 
        opp: Dict[str, Any],
        domain_boost: float
    ) -> float:
        """Calculate final composite score using adaptive weights."""
        
        # Get weights for current maturity level
        weights = self.config['scoring']['weights'][self.repo_maturity]
        
        # Normalize scores to 0-100 scale
        normalized_wsjf = min(100, wsjf_score * 5)  # WSJF typically 0-20
        normalized_ice = min(100, ice_score * 10)   # ICE typically 0-10  
        normalized_debt = tech_debt_score  # Already 0-100
        
        # Calculate weighted composite
        composite = (
            weights['wsjf'] * normalized_wsjf +
            weights['ice'] * normalized_ice +
            weights['technical_debt'] * normalized_debt
        )
        
        # Apply security boost
        if opp['category'] == 'security':
            composite *= self.config['scoring']['thresholds']['security_boost_multiplier']
        
        # Apply domain boost
        composite *= domain_boost
        
        # Apply category-specific boosts
        if 'ci' in ' '.join(opp.get('keywords', [])).lower():
            composite *= 1.5  # CI/CD is critical for maturing repos
        
        if 'core' in ' '.join(opp.get('keywords', [])).lower():
            composite *= 1.3  # Core functionality boost
        
        return round(composite, 2)
    
    def _estimate_value_delivery(self, opp: Dict[str, Any], composite_score: float) -> float:
        """Estimate value delivery in dollars or time saved."""
        
        # Base value estimates (in hours saved or $ value)
        base_values = {
            'security': 5000,    # Security incidents are expensive
            'implementation': 2000, # Core functionality value
            'automation': 1500,  # Time savings compound
            'performance': 1000, # User experience value
            'technical_debt': 800, # Maintenance time saved
            'testing': 600,      # Quality assurance value
            'documentation': 300 # Support time saved
        }
        
        base_value = base_values.get(opp['category'], 500)
        
        # Scale by composite score (normalized to 0-1)
        score_multiplier = min(3.0, composite_score / 100)
        
        value = base_value * score_multiplier
        return round(value, 2)
    
    def _determine_confidence_level(self, opp: Dict[str, Any]) -> str:
        """Determine confidence level in the opportunity assessment."""
        
        # Factors that increase confidence
        confidence_factors = {
            'well_defined': any(kw in opp.get('title', '').lower() 
                              for kw in ['missing', 'add', 'create', 'implement']),
            'measurable': opp['category'] in ['security', 'performance', 'testing'],
            'established_solution': opp['category'] in ['automation', 'documentation'],
            'clear_scope': opp.get('file_path') is not None
        }
        
        confidence_score = sum(confidence_factors.values())
        
        if confidence_score >= 3:
            return "High"
        elif confidence_score >= 2:
            return "Medium"
        else:
            return "Low"
    
    def export_scored_opportunities(
        self, 
        opportunities: List[ScoredOpportunity], 
        output_file: str
    ) -> None:
        """Export scored opportunities to JSON file."""
        with open(output_file, 'w') as f:
            json.dump([asdict(opp) for opp in opportunities], f, indent=2)
        
        print(f"📁 Exported {len(opportunities)} scored opportunities to {output_file}")
    
    def print_top_opportunities(self, opportunities: List[ScoredOpportunity], top_n: int = 10) -> None:
        """Print summary of top opportunities."""
        print(f"\n🏆 Top {top_n} Value Opportunities:")
        print("=" * 80)
        
        for i, opp in enumerate(opportunities[:top_n], 1):
            print(f"\n{i}. {opp.title}")
            print(f"   Category: {opp.category.title()} | Score: {opp.composite_score:.1f}")
            print(f"   WSJF: {opp.wsjf_score} | ICE: {opp.ice_score} | Debt: {opp.technical_debt_score}")
            print(f"   Value: ${opp.estimated_value_delivery:,.0f} | Confidence: {opp.confidence_level}")
            
            if opp.file_path:
                location = f"{opp.file_path}"
                if opp.line_number:
                    location += f":{opp.line_number}"
                print(f"   Location: {location}")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function."""
    engine = AdaptiveScoringEngine()
    
    # Score all discovered opportunities
    scored_opportunities = engine.score_opportunities(".terragon/discovered-opportunities.json")
    
    # Export scored results
    engine.export_scored_opportunities(scored_opportunities, ".terragon/scored-opportunities.json")
    
    # Print top opportunities
    engine.print_top_opportunities(scored_opportunities, 10)
    
    # Print scoring summary
    print(f"\n📊 Scoring Summary:")
    print(f"Total opportunities: {len(scored_opportunities)}")
    print(f"Average composite score: {sum(opp.composite_score for opp in scored_opportunities) / len(scored_opportunities):.1f}")
    print(f"Highest score: {scored_opportunities[0].composite_score:.1f}")
    print(f"Total estimated value: ${sum(opp.estimated_value_delivery for opp in scored_opportunities):,.0f}")

if __name__ == "__main__":
    main()