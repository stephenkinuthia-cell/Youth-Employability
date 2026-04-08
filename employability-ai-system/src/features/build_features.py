"""
Build derived features from cleaned dataset.

This module creates meaningful features through domain-driven transformations:
- Interaction features (product of related variables)
- Aggregations (combining related variables)
- Time-based features (derived from dates)
- Domain-specific features

Note: NO encoding or scaling is performed here.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_cleaned_data_path() -> Path:
    """
    Get the path to the cleaned dataset.
    
    Returns:
        Path: Path to cleaned_data.csv
    """
    src_path = Path(__file__).parent.parent.parent
    return src_path / 'data' / 'processed' / 'cleaned_data.csv'


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    
    Raises:
        FileNotFoundError: If cleaned data not found.
    """
    data_path = get_cleaned_data_path()
    
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded cleaned data from {data_path}")
    logger.info(f"Shape: {df.shape}")
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features from related variables.
    
    Interactions:
    - Demand × Reputation: Combined hiring demand and employer prestige
    - Demand × Remote: Demand in remote-friendly roles
    - Reputation × Remote: Prestigious companies with remote availability
    
    Args:
        df: Input DataFrame (must have Skill_Demand_Score, 
            Employer_Reputation_Score, Remote_Work_Availability).
    
    Returns:
        DataFrame with interaction features added.
    """
    df = df.copy()
    
    # Create interactions
    df['Demand_x_Reputation'] = (
        df['Skill_Demand_Score (1–100)'] * df['Employer_Reputation_Score (1–100)'] / 100
    )
    
    df['Demand_x_Remote'] = (
        df['Skill_Demand_Score (1–100)'] * df['Remote_Work_Availability (%)'] / 100
    )
    
    df['Reputation_x_Remote'] = (
        df['Employer_Reputation_Score (1–100)'] * df['Remote_Work_Availability (%)'] / 100
    )
    
    logger.info("Created 3 interaction features")
    
    return df


def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregation features from related skill variables.
    
    Aggregations:
    - Average skill demand (mean of all skills)
    
    Args:
        df: Input DataFrame (must have Skill_1, Skill_2, Skill_3).
    
    Returns:
        DataFrame with aggregation features added.
    """
    df = df.copy()
    
    # Note: Skills are categorical, but we can count unique skills
    # Create a feature counting unique skills required
    df['Unique_Skills_Count'] = df[['Skill_1', 'Skill_2', 'Skill_3']].nunique(axis=1)
    
    logger.info("Created 1 aggregation feature (Unique_Skills_Count)")
    
    return df


def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features derived from time/year information.
    
    Features:
    - Years since graduation
    - Career stage indicator
    
    Args:
        df: Input DataFrame (must have Graduation_Year and Year).
    
    Returns:
        DataFrame with time-based features added.
    """
    df = df.copy()
    
    # Years since graduation (proxy for career stage)
    df['Years_Since_Graduation'] = df['Year'] - df['Graduation_Year']
    
    # Career early/mid/late stage
    df['Career_Stage'] = pd.cut(
        df['Years_Since_Graduation'],
        bins=[-1, 2, 5, 100],
        labels=['Early', 'Mid', 'Late']
    )
    
    logger.info("Created 2 time-based features (Years_Since_Graduation, Career_Stage)")
    
    return df


def create_salary_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features derived from salary information.
    
    Features:
    - Salary per year of experience
    
    Args:
        df: Input DataFrame (must have Average_Starting_Salary_USD, 
            Years_Since_Graduation).
    
    Returns:
        DataFrame with salary-based features added.
    """
    df = df.copy()
    
    # Salary per year of experience (avoid division by zero)
    df['Salary_Per_Year_Experience'] = np.where(
        df['Years_Since_Graduation'] > 0,
        df['Average_Starting_Salary_USD'] / df['Years_Since_Graduation'],
        df['Average_Starting_Salary_USD']
    )
    
    logger.info("Created 1 salary-based feature (Salary_Per_Year_Experience)")
    
    return df


def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create geography-based features.
    
    Features:
    - Is_US: Binary indicator for USA
    - Region encoded categorically
    
    Args:
        df: Input DataFrame (must have Country, Region).
    
    Returns:
        DataFrame with location features added.
    """
    df = df.copy()
    
    # Binary indicator for US
    df['Is_US'] = (df['Country'] == 'USA').astype(int)
    
    logger.info("Created 1 location feature (Is_US)")
    
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature building steps.
    
    Args:
        df: Raw cleaned DataFrame.
    
    Returns:
        DataFrame with all built features.
    """
    logger.info("Starting feature building pipeline")
    
    df = create_interaction_features(df)
    df = create_aggregation_features(df)
    df = create_time_based_features(df)
    df = create_salary_based_features(df)
    df = create_location_features(df)
    
    logger.info(f"Feature building complete. New shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    df = load_cleaned_data()
    df_with_features = build_all_features(df)
    
    print("\nNew features created:")
    new_cols = set(df_with_features.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    print(f"\nFinal shape: {df_with_features.shape}")
    print("\nFirst few rows of new features:")
    print(df_with_features[list(new_cols)].head())
