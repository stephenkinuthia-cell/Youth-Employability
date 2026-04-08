"""
Pydantic schemas for the Employability AI API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Input schema for user profile data."""

    # Numerical features
    employment_rate_6_months: float = Field(..., ge=0, le=100, description="Employment rate after 6 months (%)")
    average_starting_salary_usd: float = Field(..., gt=0, description="Average starting salary in USD")
    reputation_x_remote: float = Field(..., description="Reputation score multiplied by remote work availability")
    employer_reputation_score: float = Field(..., ge=0, le=100, description="Employer reputation score (1-100)")
    demand_x_reputation: float = Field(..., description="Demand score multiplied by reputation")
    skill_demand_score: float = Field(..., ge=0, le=100, description="Skill demand score (1-100)")
    remote_work_availability: float = Field(..., ge=0, le=100, description="Remote work availability (%)")
    graduation_year: int = Field(..., ge=1900, le=2030, description="Year of graduation")

    # Categorical/binary features (skills, countries, etc.)
    degree_level_master: int = Field(0, ge=0, le=1, description="Master's degree level (0/1)")
    degree_level_phd: int = Field(0, ge=0, le=1, description="PhD degree level (0/1)")
    is_us: int = Field(0, ge=0, le=1, description="Is US-based (0/1)")

    # Job roles (one-hot encoded)
    job_role_data_analyst: int = Field(0, ge=0, le=1, description="Data Analyst role (0/1)")
    job_role_machine_learning_engineer: int = Field(0, ge=0, le=1, description="ML Engineer role (0/1)")
    job_role_software_engineer: int = Field(0, ge=0, le=1, description="Software Engineer role (0/1)")
    job_role_business_analyst: int = Field(0, ge=0, le=1, description="Business Analyst role (0/1)")
    job_role_project_manager: int = Field(0, ge=0, le=1, description="Project Manager role (0/1)")

    # Regions (one-hot encoded)
    region_north_america: int = Field(0, ge=0, le=1, description="North America region (0/1)")
    region_europe: int = Field(0, ge=0, le=1, description="Europe region (0/1)")
    region_asia_pacific: int = Field(0, ge=0, le=1, description="Asia Pacific region (0/1)")
    region_latin_america: int = Field(0, ge=0, le=1, description="Latin America region (0/1)")
    region_middle_east_africa: int = Field(0, ge=0, le=1, description="Middle East & Africa region (0/1)")

    # Fields of study (one-hot encoded)
    field_computer_science: int = Field(0, ge=0, le=1, description="Computer Science field (0/1)")
    field_engineering: int = Field(0, ge=0, le=1, description="Engineering field (0/1)")
    field_business: int = Field(0, ge=0, le=1, description="Business field (0/1)")
    field_data_science: int = Field(0, ge=0, le=1, description="Data Science field (0/1)")
    field_social_sciences: int = Field(0, ge=0, le=1, description="Social Sciences field (0/1)")

    # Key skills (one-hot encoded)
    skill_python: int = Field(0, ge=0, le=1, description="Python skill (0/1)")
    skill_machine_learning: int = Field(0, ge=0, le=1, description="Machine Learning skill (0/1)")
    skill_data_analysis: int = Field(0, ge=0, le=1, description="Data Analysis skill (0/1)")
    skill_sql: int = Field(0, ge=0, le=1, description="SQL skill (0/1)")
    skill_excel: int = Field(0, ge=0, le=1, description="Excel skill (0/1)")
    skill_communication: int = Field(0, ge=0, le=1, description="Communication skill (0/1)")
    skill_project_management: int = Field(0, ge=0, le=1, description="Project Management skill (0/1)")
    skill_laboratory_skills: int = Field(0, ge=0, le=1, description="Laboratory Skills (0/1)")

    # Additional features can be added as needed
    additional_features: Optional[Dict[str, Any]] = Field(None, description="Additional feature values")


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    user_profile: UserProfile = Field(..., description="User profile data")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    user_id: Optional[str] = Field(None, description="User identifier")
    employability_probability: float = Field(..., ge=0, le=1, description="Predicted employability probability (0-1)")
    employability_percentage: float = Field(..., ge=0, le=100, description="Predicted employability percentage (0-100%)")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Model confidence score (0-1)")


class RecommendationRequest(BaseModel):
    """Request schema for recommendation endpoint."""
    user_profile: UserProfile = Field(..., description="User profile data")
    top_n: Optional[int] = Field(3, ge=1, le=10, description="Number of top recommendations to return")
    user_id: Optional[str] = Field(None, description="User identifier")


class SkillRecommendation(BaseModel):
    """Schema for individual skill recommendation."""
    rank: int = Field(..., ge=1, description="Recommendation rank")
    recommendation: str = Field(..., description="Recommendation description")
    category: str = Field(..., description="Category (Skill/Feature)")
    improvement: float = Field(..., description="Probability improvement")
    improvement_pct: float = Field(..., description="Improvement percentage")
    current_probability: float = Field(..., ge=0, le=1, description="Current employability probability")
    predicted_probability: float = Field(..., ge=0, le=1, description="Predicted probability after improvement")


class RecommendationResponse(BaseModel):
    """Response schema for recommendation endpoint."""
    user_id: Optional[str] = Field(None, description="User identifier")
    total_recommendations: int = Field(..., ge=0, description="Total number of recommendations generated")
    recommendations: List[SkillRecommendation] = Field(..., description="List of skill recommendations")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
