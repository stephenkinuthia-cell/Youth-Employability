#!/usr/bin/env python3
"""
Test script for the Employability AI API endpoints.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from api.routes import convert_profile_to_dataframe
from api.schemas import UserProfile

def test_api_endpoints():
    """Test the API endpoints without running the server."""

    print("Testing Employability AI API endpoints...")

    # Create a sample user profile
    sample_profile = UserProfile(
        employment_rate_6_months=85.6,
        average_starting_salary_usd=50000,
        reputation_x_remote=0.5,
        employer_reputation_score=75,
        demand_x_reputation=0.8,
        skill_demand_score=80,
        remote_work_availability=60,
        graduation_year=2023,
        degree_level_master=0,
        degree_level_phd=0,
        is_us=1,
        job_role_data_analyst=0,
        job_role_machine_learning_engineer=0,
        job_role_software_engineer=1,
        job_role_business_analyst=0,
        job_role_project_manager=0,
        region_north_america=1,
        region_europe=0,
        region_asia_pacific=0,
        region_latin_america=0,
        region_middle_east_africa=0,
        field_computer_science=1,
        field_engineering=0,
        field_business=0,
        field_data_science=0,
        field_social_sciences=0,
        skill_python=1,
        skill_machine_learning=0,
        skill_data_analysis=0,
        skill_sql=0,
        skill_excel=0,
        skill_communication=1,
        skill_project_management=0,
        skill_laboratory_skills=0
    )

    print("✓ Sample user profile created")

    # Test profile conversion
    try:
        df = convert_profile_to_dataframe(sample_profile)
        print(f"✓ Profile conversion successful: {df.shape[1]} features")
    except Exception as e:
        print(f"✗ Profile conversion failed: {e}")
        return

    # Test model loading
    try:
        from models.registry import load_best_model
        model = load_best_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return

    # Test prediction
    try:
        probabilities = model.predict_proba(df)[:, 1]
        probability = float(probabilities[0])
        print(".4f")
        print("✓ Prediction successful")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test recommendation system
    try:
        from recommendation.recommend import get_recommendations_for_user
        from models.registry import load_best_model
        import pandas as pd

        features_path = Path(__file__).parent / "data" / "processed" / "final_features.csv"
        features_df = pd.read_csv(features_path)

        results = get_recommendations_for_user(df.iloc[0], model, features_df, top_n=3)
        print(f"✓ Recommendations generated: {len(results['summary']['recommendations'])} recommendations")

        # Show top recommendation
        top_rec = results['summary']['recommendations'][0]
        print(f"✓ Top recommendation: {top_rec['recommendation']} (+{top_rec['improvement']:.4f})")

    except Exception as e:
        print(f"✗ Recommendations failed: {e}")
        return

    print("\n🎉 All API components tested successfully!")
    print("\nAPI Endpoints ready:")
    print("- POST /api/v1/predict")
    print("- POST /api/v1/recommend")
    print("- GET /api/v1/health")

if __name__ == "__main__":
    test_api_endpoints()