"""
API routes for the Employability AI service.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import logging

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import load_best_model
from recommendation.recommend import get_recommendations_for_user
from .schemas import (
    PredictionRequest, PredictionResponse,
    RecommendationRequest, RecommendationResponse,
    SkillRecommendation, ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global variables for model and data (loaded once)
_model = None
_features_df = None


def get_model():
    """Dependency to get the trained model."""
    global _model
    if _model is None:
        try:
            _model = load_best_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    return _model


def get_features_data():
    """Dependency to get the features dataframe."""
    global _features_df
    if _features_df is None:
        try:
            features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
            _features_df = pd.read_csv(features_path)
            logger.info("Features data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load features data: {e}")
            raise HTTPException(status_code=500, detail="Features data loading failed")
    return _features_df


def convert_profile_to_dataframe(user_profile) -> pd.DataFrame:
    """Convert UserProfile to pandas DataFrame format expected by the model."""
    # Create a dictionary with all the features from the user profile
    profile_dict = {}

    # Add all attributes from the user profile
    for field, value in user_profile.__dict__.items():
        if field != 'additional_features':
            # Convert field names to match the training data format
            if field.startswith('skill_'):
                # Handle skill fields - map to the expected format
                skill_mappings = {
                    'skill_python': 'Skill_1_Python',
                    'skill_machine_learning': 'Skill_1_Machine Learning',
                    'skill_data_analysis': 'Skill_2_Data Analysis',
                    'skill_sql': 'Skill_1_SQL',
                    'skill_excel': 'Skill_1_Excel',
                    'skill_communication': 'Skill_2_Communication',
                    'skill_project_management': 'Skill_3_Project Management',
                    'skill_laboratory_skills': 'Skill_3_Laboratory Skills'
                }
                if field in skill_mappings:
                    profile_dict[skill_mappings[field]] = value
            elif field.startswith('job_role_'):
                # Handle job role fields
                job_mappings = {
                    'job_role_data_analyst': 'Job_Role_Data Analyst',
                    'job_role_machine_learning_engineer': 'Job_Role_Machine Learning Engineer',
                    'job_role_software_engineer': 'Job_Role_Software Engineer',
                    'job_role_business_analyst': 'Job_Role_Business Analyst',
                    'job_role_project_manager': 'Job_Role_Project Manager'
                }
                if field in job_mappings:
                    profile_dict[job_mappings[field]] = value
            elif field.startswith('region_'):
                # Handle region fields
                region_mappings = {
                    'region_north_america': 'Region_North America',
                    'region_europe': 'Region_Europe',
                    'region_asia_pacific': 'Region_Asia-Pacific',
                    'region_latin_america': 'Region_Latin America',
                    'region_middle_east_africa': 'Region_Middle East & Africa'
                }
                if field in region_mappings:
                    profile_dict[region_mappings[field]] = value
            elif field.startswith('field_'):
                # Handle field of study fields
                field_mappings = {
                    'field_computer_science': 'Field_of_Study_Computer Science',
                    'field_engineering': 'Field_of_Study_Engineering',
                    'field_business': 'Field_of_Study_Business',
                    'field_data_science': 'Field_of_Study_Data Science & AI',
                    'field_social_sciences': 'Field_of_Study_Social Sciences'
                }
                if field in field_mappings:
                    profile_dict[field_mappings[field]] = value
            else:
                # Handle direct mappings
                direct_mappings = {
                    'employment_rate_6_months': 'Employment_Rate_6_Months (%)',
                    'average_starting_salary_usd': 'Average_Starting_Salary_USD',
                    'reputation_x_remote': 'Reputation_x_Remote',
                    'employer_reputation_score': 'Employer_Reputation_Score (1–100)',
                    'demand_x_reputation': 'Demand_x_Reputation',
                    'skill_demand_score': 'Skill_Demand_Score (1–100)',
                    'remote_work_availability': 'Remote_Work_Availability (%)',
                    'graduation_year': 'Graduation_Year',
                    'degree_level_master': 'Degree_Level_Master',
                    'degree_level_phd': 'Degree_Level_PhD',
                    'is_us': 'Is_US'
                }
                if field in direct_mappings:
                    profile_dict[direct_mappings[field]] = value

    # Add additional features if provided
    if hasattr(user_profile, 'additional_features') and user_profile.additional_features:
        profile_dict.update(user_profile.additional_features)

    # Convert to DataFrame
    df = pd.DataFrame([profile_dict])

    # Load the expected feature names from the training data
    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
    expected_features = pd.read_csv(features_path).drop(columns=['Employment_Rate_12_Months (%)']).columns.tolist()

    # Ensure all expected columns are present (fill missing with 0)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Return only the expected features in the correct order
    return df[expected_features]


@router.post("/predict", response_model=PredictionResponse)
async def predict_employability(
    request: PredictionRequest,
    model = Depends(get_model)
) -> PredictionResponse:
    """
    Predict employability probability for a user profile.

    Returns the predicted employability probability and related metrics.
    """
    try:
        # Convert user profile to DataFrame
        user_df = convert_profile_to_dataframe(request.user_profile)

        # Make prediction
        probabilities = model.predict_proba(user_df)[:, 1]  # Probability of positive class
        probability = float(probabilities[0])

        # Calculate percentage
        percentage = probability * 100

        # For now, use probability as confidence (can be enhanced with proper confidence intervals)
        confidence = probability

        response = PredictionResponse(
            user_id=getattr(request, 'user_id', None),
            employability_probability=probability,
            employability_percentage=percentage,
            confidence_score=confidence
        )

        logger.info(f"Prediction completed for user: probability={probability:.4f}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/recommend", response_model=RecommendationResponse)
async def get_skill_recommendations(
    request: RecommendationRequest,
    model = Depends(get_model),
    features_df = Depends(get_features_data)
) -> RecommendationResponse:
    """
    Get personalized skill recommendations for a user profile.

    Returns top N skill recommendations based on counterfactual simulations.
    """
    try:
        # Convert user profile to DataFrame
        user_df = convert_profile_to_dataframe(request.user_profile)

        # Get recommendations
        results = get_recommendations_for_user(
            user_df.iloc[0],  # Convert to Series
            model,
            features_df,
            top_n=request.top_n,
            user_id=request.user_id
        )

        # Convert recommendations to proper format
        recommendations = []
        for rec in results['summary']['recommendations']:
            skill_rec = SkillRecommendation(
                rank=rec['rank'],
                recommendation=rec['recommendation'],
                category=rec['category'],
                improvement=rec['improvement'],
                improvement_pct=rec['improvement_pct'],
                current_probability=rec['current_probability'],
                predicted_probability=rec['predicted_probability']
            )
            recommendations.append(skill_rec)

        response = RecommendationResponse(
            user_id=request.user_id,
            total_recommendations=len(recommendations),
            recommendations=recommendations,
            summary={
                "total_potential_improvement": sum(r.improvement for r in recommendations),
                "average_improvement": sum(r.improvement for r in recommendations) / len(recommendations) if recommendations else 0,
                "recommendation_count": len(recommendations)
            }
        )

        logger.info(f"Recommendations generated for user {request.user_id}: {len(recommendations)} recommendations")
        return response

    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "employability-ai-api"}
