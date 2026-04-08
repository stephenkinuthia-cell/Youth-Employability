"""
Quick test of recommendation system for a single user.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommendation.recommend import get_recommendations_for_user
from models.registry import load_best_model

# Load data
features_path = Path(__file__).parent / "data" / "processed" / "final_features.csv"
features_df = pd.read_csv(features_path)

# Load model
model = load_best_model()

# Get recommendations for first user
user_profile = features_df.drop(columns=['Employment_Rate_12_Months (%)']).iloc[0]

print("Generating recommendations for user_0...")
results = get_recommendations_for_user(user_profile, model, features_df, top_n=3, user_id="user_0")

print("\n" + "="*100)
print("RECOMMENDATIONS - DETAILED VIEW")
print("="*100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(results['recommendations_df'].to_string())

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"User: {results['summary']['user_id']}")
print(f"Total Recommendations: {results['summary']['recommendation_count']}")

print("\nRecommendation Details:")
for rec in results['summary']['recommendations']:
    print(f"  {rec['rank']}. {rec['recommendation']}: +{rec['improvement']:.4f} ({rec['improvement_pct']:.2f}%)")