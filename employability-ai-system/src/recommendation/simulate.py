"""
Simulation module for counterfactual skill improvement analysis.

This module performs counterfactual simulations by creating modified user
profiles with improved skills and predicting employment probability changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Realistic improvement increments
IMPROVEMENT_INCREMENTS = {
    'skill_acquisition': 1.0,      # Add missing 1-hot encoded skill
    'demand_boost': 10.0,           # Increase skill demand score by 10 points
    'reputation_boost': 5.0,        # Increase reputation by 5 points
    'remote_work_boost': 5.0        # Increase remote availability by 5%
}

def create_modified_profile(user_profile: pd.Series, modification: Dict[str, any]) -> pd.Series:
    """
    Create a modified user profile with simulated skill improvement.

    Args:
        user_profile: Original user profile
        modification: Dictionary describing the modification
                     {'type': 'skill_add', 'column': 'Skill_X_SkillName'}
                     or {'type': 'feature_boost', 'column': 'Feature', 'value': increment}

    Returns:
        Modified user profile
    """
    modified = user_profile.copy()
    mod_type = modification.get('type')

    if mod_type == 'skill_add':
        # Add missing skill (one-hot encoded)
        col = modification['column']
        if col in modified.index and modified[col] == 0:
            modified[col] = 1
            logger.debug(f"Added skill: {col}")

    elif mod_type == 'feature_boost':
        # Boost numerical feature (e.g., demand score, reputation)
        col = modification['column']
        if col in modified.index:
            current_value = modified[col]
            boost = modification['value']
            # Ensure bounded values (don't exceed max)
            if col.endswith('(%)'):
                modified[col] = min(current_value + boost, 100.0)
            else:
                modified[col] = min(current_value + boost, 100.0)  # Scale 0-100
            logger.debug(f"Boosted {col} by {boost}")

    return modified

def simulate_skill_improvement(model, user_profile: pd.Series,
                              modification: Dict[str, any]) -> Dict[str, float]:
    """
    Simulate employability outcome for a modified profile.

    Args:
        model: Trained model
        user_profile: Original user profile
        modification: Modification to apply

    Returns:
        Dictionary with original and modified probabilities
    """
    # Original prediction
    X_original = user_profile.values.reshape(1, -1)
    prob_original = model.predict_proba(X_original)[0, 1]

    # Modified prediction
    modified_profile = create_modified_profile(user_profile, modification)
    X_modified = modified_profile.values.reshape(1, -1)
    prob_modified = model.predict_proba(X_modified)[0, 1]

    # Calculate improvement
    improvement = prob_modified - prob_original
    improvement_pct = (improvement / prob_original * 100) if prob_original > 0 else 0

    result = {
        'modification': modification,
        'prob_original': float(prob_original),
        'prob_modified': float(prob_modified),
        'improvement': float(improvement),
        'improvement_pct': float(improvement_pct)
    }

    return result

def simulate_all_skill_additions(model, user_profile: pd.Series,
                                missing_skills: Dict[str, list]) -> list:
    """
    Simulate adding each missing skill.

    Args:
        model: Trained model
        user_profile: User profile
        missing_skills: Dictionary of missing skills from skill_gap

    Returns:
        List of simulation results
    """
    results = []

    for skill_level, skills in missing_skills.items():
        for skill_col, skill_name in skills:
            modification = {
                'type': 'skill_add',
                'column': skill_col,
                'skill_name': skill_name,
                'skill_level': skill_level
            }

            result = simulate_skill_improvement(model, user_profile, modification)
            result['recommendation_type'] = 'skill_acquisition'
            results.append(result)

            logger.debug(f"Simulated adding {skill_name}: +{result['improvement']:.4f}")

    return results

def simulate_feature_improvements(model, user_profile: pd.Series,
                                 weak_features: Dict[str, Tuple]) -> list:
    """
    Simulate improving weak numerical features.

    Args:
        model: Trained model
        user_profile: User profile
        weak_features: Dictionary of weak features from skill_gap

    Returns:
        List of simulation results
    """
    results = []

    for feature_label, (feature_col, current_value) in weak_features.items():
        # Determine boost amount
        if 'demand' in feature_label:
            boost = IMPROVEMENT_INCREMENTS['demand_boost']
        elif 'reputation' in feature_label:
            boost = IMPROVEMENT_INCREMENTS['reputation_boost']
        elif 'remote' in feature_label:
            boost = IMPROVEMENT_INCREMENTS['remote_work_boost']
        else:
            boost = 5.0  # Default boost

        modification = {
            'type': 'feature_boost',
            'column': feature_col,
            'feature_label': feature_label,
            'value': boost
        }

        result = simulate_skill_improvement(model, user_profile, modification)
        result['recommendation_type'] = 'feature_improvement'
        results.append(result)

        logger.debug(f"Simulated boosting {feature_label}: +{result['improvement']:.4f}")

    return results

def extract_recommendation_details(simulation_result: Dict[str, any]) -> Dict[str, any]:
    """
    Extract readable recommendation details from simulation result.

    Args:
        simulation_result: Result from simulate_skill_improvement

    Returns:
        Dictionary with readable recommendation details
    """
    mod = simulation_result['modification']
    rec_type = simulation_result['recommendation_type']

    if rec_type == 'skill_acquisition':
        return {
            'recommendation_type': 'Acquire Skill',
            'detail': mod['skill_name'],
            'level': mod['skill_level'],
            'improvement': simulation_result['improvement'],
            'improvement_pct': simulation_result['improvement_pct']
        }
    else:  # feature_improvement
        label = mod['feature_label'].replace('_', ' ').title()
        return {
            'recommendation_type': 'Improve Feature',
            'detail': label,
            'improvement': simulation_result['improvement'],
            'improvement_pct': simulation_result['improvement_pct']
        }

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from skill_gap import analyze_skill_gap

    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
    features_df = pd.read_csv(features_path)
    user_profile = features_df.drop(columns=['Employment_Rate_12_Months (%)']).iloc[0]

    # Load model
    from ..models.registry import load_best_model
    model = load_best_model()

    # Analyze gaps
    analysis = analyze_skill_gap(user_profile, features_df)

    # Simulate skill additions
    skill_results = simulate_all_skill_additions(model, user_profile, analysis['missing_skills'])
    print(f"Simulated {len(skill_results)} skill additions")

    # Show top improvements
    skill_results_sorted = sorted(skill_results, key=lambda x: x['improvement'], reverse=True)
    for i, result in enumerate(skill_results_sorted[:3]):
        rec = extract_recommendation_details(result)
        print(f"{i+1}. {rec['detail']}: +{rec['improvement']:.4f} ({rec['improvement_pct']:.2f}%)")
