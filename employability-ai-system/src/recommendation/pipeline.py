"""
Recommendation pipeline for generating and saving skill recommendations.

This script runs the complete recommendation workflow for all users and
saves results to CSV files.
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_recommendation_pipeline(sample_size: int = 50):
    """Run the complete recommendation pipeline.
    
    Args:
        sample_size: Number of users to generate recommendations for (for demo)
    """
    logger.info("Starting recommendation pipeline...")

    # Step 1: Load data and model
    logger.info("Step 1: Loading data and model...")
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from models.registry import load_best_model

    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
    features_df = pd.read_csv(features_path)

    logger.info(f"Loaded {len(features_df)} user profiles")

    model = load_best_model()
    logger.info("Loaded trained model")

    # Step 2: Generate recommendations (sample for efficiency)
    logger.info("Step 2: Generating recommendations...")
    from .recommend import get_recommendations_for_user

    sample_size = min(sample_size, len(features_df))
    recommendations_list = []

    for idx in range(sample_size):
        if idx % 10 == 0:
            logger.info(f"Processing user {idx + 1}/{sample_size}...")

        user_profile = features_df.drop(columns=['Employment_Rate_12_Months (%)']).iloc[idx]

        try:
            user_recs = get_recommendations_for_user(
                user_profile, model, features_df,
                top_n=3, user_id=f"user_{idx}"
            )
            recommendations_list.append(user_recs)
        except Exception as e:
            logger.warning(f"Error for user_{idx}: {str(e)}")

    logger.info(f"Generated recommendations for {len(recommendations_list)} users")

    # Step 3: Compile results
    logger.info("Step 3: Compiling results...")
    all_recommendations_data = []
    summaries_data = []

    for user_rec in recommendations_list:
        # Add recommendations
        rec_df = user_rec['recommendations_df'].copy()
        rec_df = rec_df.assign(user_id=user_rec['user_id'])
        all_recommendations_data.append(rec_df)

        # Add summary
        summaries_data.append(user_rec['summary'])

    # Combine into single DataFrames
    if all_recommendations_data:
        recommendations_combined = pd.concat(all_recommendations_data, ignore_index=True)
    else:
        recommendations_combined = pd.DataFrame()

    summaries_df = pd.DataFrame(summaries_data)

    # Step 4: Save results
    logger.info("Step 4: Saving results...")
    output_dir = Path(__file__).parent.parent.parent / "reports" / "recommendations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save recommendations
    rec_path = output_dir / "skill_recommendations.csv"
    recommendations_combined.to_csv(rec_path, index=False)
    logger.info(f"Recommendations saved to {rec_path}")

    # Save summaries
    summary_path = output_dir / "recommendations_summary.csv"
    summaries_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    # Step 5: Print summary statistics
    logger.info("Step 5: Printing summary statistics...")
    print("\n" + "="*70)
    print("RECOMMENDATION PIPELINE COMPLETE")
    print("="*70)
    print(f"\nUsers Processed: {len(recommendations_list)}")
    print(f"Total Recommendations: {len(recommendations_combined)}")

    if len(summaries_df) > 0:
        avg_improvement = summaries_df['total_potential_improvement'].mean()
        max_improvement = summaries_df['total_potential_improvement'].max()
        print(f"\nImprovement Statistics:")
        print(".4f")
        print(".4f")

    print(f"\nResults saved to:")
    print(f"  - {rec_path}")
    print(f"  - {summary_path}")
    print("="*70)

    return recommendations_combined, summaries_df

if __name__ == "__main__":
    # Run pipeline for 50 sample users
    recommendations_df, summaries_df = run_recommendation_pipeline(sample_size=50)