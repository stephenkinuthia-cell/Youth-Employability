"""
Ranking module for skill recommendations.

This module ranks skill recommendations by their predicted impact on
employability and selects the top recommendations.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rank_recommendations(simulation_results: List[Dict[str, Any]],
                        metric: str = 'improvement') -> List[Dict[str, Any]]:
    """
    Rank recommendations by improvement impact.

    Args:
        simulation_results: List of simulation results from simulate module
        metric: Metric to rank by ('improvement' or 'improvement_pct')

    Returns:
        Sorted list of recommendations (best first)
    """
    logger.info(f"Ranking {len(simulation_results)} recommendations by {metric}...")

    # Sort by selected metric
    ranked = sorted(simulation_results, key=lambda x: x[metric], reverse=True)

    for i, rec in enumerate(ranked):
        rec['rank'] = i + 1

    logger.info(f"Ranking complete. Top recommendation: {ranked[0][metric]:.4f}")
    return ranked

def get_top_recommendations(ranked_recommendations: List[Dict[str, Any]],
                           top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Extract top N recommendations.

    Args:
        ranked_recommendations: Ranked list from rank_recommendations
        top_n: Number of recommendations to return

    Returns:
        Top N recommendations
    """
    top_recs = ranked_recommendations[:top_n]
    logger.info(f"Selected top {len(top_recs)} recommendations")
    return top_recs

def format_recommendations(top_recommendations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format recommendations into a readable DataFrame.

    Args:
        top_recommendations: Top recommendations from get_top_recommendations

    Returns:
        Formatted DataFrame with recommendations
    """
    formatted_data = []

    for rec in top_recommendations:
        mod = rec['modification']
        rec_type = rec['recommendation_type']

        if rec_type == 'skill_acquisition':
            recommendation = f"Acquire: {mod['skill_name']}"
            category = "Skill"
        else:
            feature_label = mod['feature_label'].replace('_', ' ').title()
            recommendation = f"Improve: {feature_label}"
            category = "Feature"

        row = {
            'rank': rec['rank'],
            'recommendation': recommendation,
            'category': category,
            'improvement': rec['improvement'],
            'improvement_pct': rec['improvement_pct'],
            'current_probability': rec['prob_original'],
            'predicted_probability': rec['prob_modified']
        }

        formatted_data.append(row)

    return pd.DataFrame(formatted_data)

def print_recommendations(top_recommendations: List[Dict[str, Any]]):
    """
    Print recommendations in a readable format.

    Args:
        top_recommendations: Top recommendations
    """
    print("\n" + "="*70)
    print("TOP SKILL RECOMMENDATIONS FOR IMPROVED EMPLOYABILITY")
    print("="*70)

    for i, rec in enumerate(top_recommendations, 1):
        mod = rec['modification']
        rec_type = rec['recommendation_type']

        print(f"\n#{i} Recommendation:")

        if rec_type == 'skill_acquisition':
            print(f"  Action: Acquire Skill")
            print(f"  Skill: {mod['skill_name']} ({mod['skill_level'].upper()})")
        else:
            feature_label = mod['feature_label'].replace('_', ' ').title()
            print(f"  Action: Improve Feature")
            print(f"  Feature: {feature_label}")

        print(f"  Expected Impact:")
        print(".4f")
        print(".2f")
        print(".4f")
        print(".4f")

    print("\n" + "="*70)

def create_summary_report(top_recommendations: List[Dict[str, Any]],
                         user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a summary report of recommendations.

    Args:
        top_recommendations: Top recommendations
        user_id: Optional user identifier

    Returns:
        Dictionary with summary report
    """
    total_potential_improvement = sum(rec['improvement'] for rec in top_recommendations)
    avg_improvement = total_potential_improvement / len(top_recommendations) if top_recommendations else 0

    report = {
        'user_id': user_id or 'unknown',
        'recommendation_count': len(top_recommendations),
        'total_potential_improvement': float(total_potential_improvement),
        'average_improvement_per_action': float(avg_improvement),
        'recommendations': []
    }

    for rec in top_recommendations:
        mod = rec['modification']
        if rec['recommendation_type'] == 'skill_acquisition':
            skill_name = mod['skill_name']
        else:
            skill_name = mod['feature_label']

        report['recommendations'].append({
            'rank': rec['rank'],
            'recommendation': skill_name,
            'improvement': float(rec['improvement']),
            'improvement_pct': float(rec['improvement_pct'])
        })

    return report

if __name__ == "__main__":
    # Example usage - would be called from recommend module
    print("Ranking module loaded")
