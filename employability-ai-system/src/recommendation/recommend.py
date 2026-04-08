"""
Recommendation orchestration module.

This module brings together skill gap analysis, counterfactual simulations,
and ranking to provide personalized skill recommendations for users.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_recommendations_for_user(user_profile: pd.Series, model, all_features: pd.DataFrame,
                                top_n: int = 3, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get personalized skill recommendations for a user.

    Complete workflow:
    1. Analyze skill gaps
    2. Simulate improvements
    3. Rank and select top recommendations

    Args:
        user_profile: User's feature profile (1D pandas Series)
        model: Trained model for predictions
        all_features: All features dataframe for context
        top_n: Number of recommendations to return
        user_id: Optional user identifier

    Returns:
        Dictionary with personalized recommendations
    """
    logger.info(f"Generating recommendations for user {user_id or 'unknown'}...")

    # Step 1: Analyze skill gaps
    from .skill_gap import analyze_skill_gap
    gap_analysis = analyze_skill_gap(user_profile, all_features)
    logger.info(f"Gap analysis: {gap_analysis['total_missing_skills']} missing skills, "
                f"{gap_analysis['total_weak_features']} weak features")

    # Step 2: Run simulations
    from .simulate import simulate_all_skill_additions, simulate_feature_improvements
    
    skill_simulations = simulate_all_skill_additions(model, user_profile, gap_analysis['missing_skills'])
    feature_simulations = simulate_feature_improvements(model, user_profile, gap_analysis['weak_features'])

    all_simulations = skill_simulations + feature_simulations
    logger.info(f"Completed {len(all_simulations)} simulations")

    # Step 3: Rank recommendations
    from .rank import rank_recommendations, get_top_recommendations, format_recommendations, create_summary_report
    
    ranked = rank_recommendations(all_simulations, metric='improvement')
    top_recs = get_top_recommendations(ranked, top_n=top_n)

    # Step 4: Format results
    recommendations_df = format_recommendations(top_recs)
    summary = create_summary_report(top_recs, user_id=user_id)

    logger.info(f"Generated {len(top_recs)} recommendations")

    results = {
        'user_id': user_id,
        'recommendations_df': recommendations_df,
        'recommendations': top_recs,
        'summary': summary,
        'gap_analysis': gap_analysis
    }

    return results

def get_batch_recommendations(user_profiles: pd.DataFrame, model,
                             top_n: int = 3, user_id_column: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate recommendations for multiple users.

    Args:
        user_profiles: DataFrame with user profiles (including target column)
        model: Trained model
        top_n: Number of recommendations per user
        user_id_column: Optional column name for user IDs

    Returns:
        List of recommendation results per user
    """
    logger.info(f"Generating recommendations for {len(user_profiles)} users...")

    target_col = 'Employment_Rate_12_Months (%)'
    features_df = user_profiles.drop(columns=[target_col])

    all_results = []

    for idx, user_profile in features_df.iterrows():
        user_id = user_profiles[user_id_column].iloc[idx] if user_id_column else f"user_{idx}"

        try:
            recs = get_recommendations_for_user(user_profile, model, user_profiles, 
                                               top_n=top_n, user_id=user_id)
            all_results.append(recs)
        except Exception as e:
            logger.error(f"Error generating recommendations for {user_id}: {str(e)}")

    logger.info(f"Generated recommendations for {len(all_results)} users successfully")
    return all_results

def save_recommendations(recommendations: Dict[str, Any], output_dir: Path):
    """
    Save recommendations to CSV files.

    Args:
        recommendations: Results from get_recommendations_for_user
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_id = recommendations['user_id'] or 'unknown'

    # Save recommendations
    rec_path = output_dir / f"recommendations_{user_id}.csv"
    recommendations['recommendations_df'].to_csv(rec_path, index=False)
    logger.info(f"Recommendations saved to {rec_path}")

    # Save summary
    summary = recommendations['summary']
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"summary_{user_id}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

def save_batch_recommendations(all_recommendations: List[Dict[str, Any]], output_dir: Path):
    """
    Save batch recommendations to consolidated CSV.

    Args:
        all_recommendations: List of results from get_batch_recommendations
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all recommendations
    all_recs_data = []
    for user_rec in all_recommendations:
        recs_df = user_rec['recommendations_df']
        recs_df['user_id'] = user_rec['user_id']
        all_recs_data.append(recs_df)

    if all_recs_data:
        combined_df = pd.concat(all_recs_data, ignore_index=True)
        output_path = output_dir / "all_recommendations.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined recommendations saved to {output_path}")

    # Combine all summaries
    all_summaries = [rec['summary'] for rec in all_recommendations]
    summaries_df = pd.DataFrame(all_summaries)
    summary_path = output_dir / "recommendations_summary.csv"
    summaries_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from ..models.registry import load_best_model

    logger.info("Loading data and model...")
    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
    features_df = pd.read_csv(features_path)

    model = load_best_model()

    # Get recommendations for first user
    user_profile = features_df.drop(columns=['Employment_Rate_12_Months (%)']).iloc[0]
    results = get_recommendations_for_user(user_profile, model, features_df, user_id="user_0")

    print("\nRecommendations DataFrame:")
    print(results['recommendations_df'])

    print("\nSummary:")
    print(results['summary'])
