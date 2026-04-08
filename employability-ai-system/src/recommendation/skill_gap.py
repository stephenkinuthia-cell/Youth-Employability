"""
Skill gap analysis module for identifying weak or missing skills.

This module analyzes user profiles to identify skills that are either
missing or underutilized, which could be targeted for improvement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define skill-related columns
SKILL_COLUMNS = {
    'categorical_skills': [
        'Skill_1_Public Policy', 'Skill_1_SolidWorks', 'Skill_1_NLP', 
        'Skill_1_Strategic Planning', 'Skill_1_Qualitative Research', 
        'Skill_1_Machine Learning', 'Skill_1_Communication', 'Skill_1_Diagnostics',
        'Skill_1_Scientific Writing', 'Skill_1_Data Visualization', 
        'Skill_1_Experimental Design', 'Skill_1_Laboratory Skills', 'Skill_1_Excel',
        'Skill_1_Python', 'Skill_1_Data Analysis', 'Skill_1_Lean Six Sigma',
        'Skill_1_Market Research', 'Skill_1_Clinical Research', 'Skill_1_Project Management',
        'Skill_2_Data Visualization', 'Skill_2_Cloud Computing', 'Skill_2_Qualitative Research',
        'Skill_2_SolidWorks', 'Skill_2_Data Analysis', 'Skill_2_Communication', 
        'Skill_2_Machine Learning', 'Skill_2_Financial Modeling', 'Skill_2_Big Data',
        'Skill_2_Experimental Design', 'Skill_2_Statistical Analysis', 
        'Skill_2_Public Policy', 'Skill_2_Laboratory Skills', 'Skill_2_Lean Six Sigma',
        'Skill_2_Scientific Writing', 'Skill_2_Project Management', 'Skill_3_Experimental Design',
        'Skill_3_Strategic Planning', 'Skill_3_Patient Care', 'Skill_3_R', 
        'Skill_3_Project Management', 'Skill_3_MATLAB', 'Skill_3_Laboratory Skills',
        'Skill_3_Cybersecurity', 'Skill_3_Communication', 'Skill_3_Data Visualization',
        'Skill_3_Qualitative Research', 'Skill_3_Data Analysis', 'Skill_3_Diagnostics',
        'Skill_3_Scientific Writing'
    ],
    'numerical_features': [
        'Skill_Demand_Score (1–100)',
        'Employer_Reputation_Score (1–100)',
        'Skill_Demand_Score (1–100)',
        'Remote_Work_Availability (%)'
    ]
}

def extract_skill_names(columns: List[str]) -> Dict[int, Dict[str, List[str]]]:
    """
    Extract unique skills from one-hot encoded columns.

    Args:
        columns: List of feature column names

    Returns:
        Dictionary mapping skill position to skill names
    """
    skills_map = {'skill_1': [], 'skill_2': [], 'skill_3': []}

    for col in columns:
        for skill_level in ['Skill_1_', 'Skill_2_', 'Skill_3_']:
            if col.startswith(skill_level):
                skill_name = col.replace(skill_level, '')
                level = skill_level.lower().rstrip('_')
                if skill_name not in skills_map[level]:
                    skills_map[level].append(skill_name)

    return skills_map

def identify_missing_skills(user_profile: pd.Series, skill_columns: List[str]) -> Dict[str, List[str]]:
    """
    Identify missing skills in user profile.

    Args:
        user_profile: Single row user profile
        skill_columns: List of skill feature columns

    Returns:
        Dictionary with missing skills by level
    """
    missing_skills = {'skill_1': [], 'skill_2': [], 'skill_3': []}

    for col in skill_columns:
        if col in user_profile.index and user_profile[col] == 0:
            for level in ['Skill_1_', 'Skill_2_', 'Skill_3_']:
                if col.startswith(level):
                    skill_name = col.replace(level, '')
                    skill_level = level.lower().rstrip('_')
                    if skill_name not in missing_skills[skill_level]:
                        missing_skills[skill_level].append((col, skill_name))

    return missing_skills

def identify_weak_features(user_profile: pd.Series) -> Dict[str, Tuple[str, float]]:
    """
    Identify weak numerical features that could be improved.

    Args:
        user_profile: Single row user profile

    Returns:
        Dictionary with weak features and their values
    """
    weak_features = {}

    feature_mapping = {
        'Skill_Demand_Score (1–100)': 'demand',
        'Employer_Reputation_Score (1–100)': 'reputation',
        'Remote_Work_Availability (%)': 'remote_work'
    }

    for feature, label in feature_mapping.items():
        if feature in user_profile.index:
            value = user_profile[feature]
            # Consider anything below median as "weak"
            if value < 50:  # Using normalized scale
                weak_features[label] = (feature, value)

    return weak_features

def analyze_skill_gap(user_profile: pd.Series, all_features: pd.DataFrame) -> Dict[str, any]:
    """
    Complete skill gap analysis for a user.

    Args:
        user_profile: Single row user profile
        all_features: All features dataframe to understand skill distribution

    Returns:
        Dictionary with skill gap analysis
    """
    logger.info("Analyzing skill gaps...")

    # Get skill columns
    skill_columns = [col for col in user_profile.index if 'Skill' in col and col.startswith('Skill_')]

    # Identify missing skills
    missing_skills = identify_missing_skills(user_profile, skill_columns)

    # Identify weak features
    weak_features = identify_weak_features(user_profile)

    # Extract skill names from dataset
    skills_map = extract_skill_names(all_features.columns.tolist())

    analysis = {
        'missing_skills': missing_skills,
        'weak_features': weak_features,
        'skill_columns': skill_columns,
        'skills_map': skills_map,
        'total_missing_skills': sum(len(v) for v in missing_skills.values()),
        'total_weak_features': len(weak_features)
    }

    logger.info(f"Found {analysis['total_missing_skills']} missing skills and {analysis['total_weak_features']} weak features")
    return analysis

def print_skill_gap_report(analysis: Dict[str, any]):
    """
    Print skill gap analysis report.

    Args:
        analysis: Results from analyze_skill_gap
    """
    print("\n" + "="*60)
    print("SKILL GAP ANALYSIS REPORT")
    print("="*60)

    print(f"\nMissing Skills:")
    for level, skills in analysis['missing_skills'].items():
        print(f"  {level.upper()}: {len(skills)} missing")
        for col, skill_name in skills[:5]:  # Show top 5
            print(f"    - {skill_name}")
        if len(skills) > 5:
            print(f"    ... and {len(skills) - 5} more")

    print(f"\nWeak Features:")
    for feature, (col_name, value) in analysis['weak_features'].items():
        print(f"  {feature}: {value:.2f}")

if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
    features_df = pd.read_csv(features_path)

    # Use first user as example
    user_profile = features_df.drop(columns=['Employment_Rate_12_Months (%)']).iloc[0]

    analysis = analyze_skill_gap(user_profile, features_df)
    print_skill_gap_report(analysis)
