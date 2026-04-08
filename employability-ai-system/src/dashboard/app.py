"""
Streamlit Dashboard for Employability AI System.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import load_best_model
from recommendation.recommend import get_recommendations_for_user
from api.routes import convert_profile_to_dataframe

# Configure page
st.set_page_config(
    page_title="Employability AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data (cached)
@st.cache_resource
def load_model_and_data():
    """Load model and features data."""
    try:
        # Add debug information
        st.write("🔄 Loading ML model...")
        model = load_best_model()
        st.write("✅ Model loaded successfully")

        st.write("🔄 Loading feature data...")
        features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
        features_df = pd.read_csv(features_path)
        st.write(f"✅ Data loaded successfully: {features_df.shape}")

        return model, features_df
    except Exception as e:
        st.error(f"❌ Failed to load model or data: {e}")
        st.exception(e)
        return None, None

def get_risk_category(probability):
    """Get risk category based on probability."""
    if probability >= 0.85:
        return "Low Risk", "risk-low", "🟢"
    elif probability >= 0.70:
        return "Medium Risk", "risk-medium", "🟡"
    else:
        return "High Risk", "risk-high", "🔴"

def create_feature_importance_plot(model, feature_names, top_n=10):
    """Create a simple feature importance plot."""
    try:
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # Get top N features
            indices = np.argsort(importance)[-top_n:]
            top_features = [feature_names[i] for i in indices]
            top_importance = importance[indices]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(top_features)), top_importance, color='#1f77b4', alpha=0.7)

            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([f.split('_')[-1][:20] + '...' if len(f.split('_')[-1]) > 20 else f.split('_')[-1] for f in top_features])
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importance')

            # Add value labels on bars
            for bar, value in zip(bars, top_importance):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       '.3f', ha='left', va='center', fontsize=9)

            plt.tight_layout()
            return fig
    except Exception as e:
        st.warning(f"Could not create feature importance plot: {e}")
        return None

def main():
    """Main dashboard function."""
    st.markdown('<div class="main-header">🎓 Employability AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Predict graduate employability and get personalized skill recommendations.")

    # Initialize session state for model and data
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.features_df = None
        st.session_state.loaded = False

    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=False, help="Show detailed loading information")

    # Load button
    if st.sidebar.button("🚀 Load Model & Data", type="primary") or not st.session_state.loaded:
        with st.spinner("Loading model and data... This may take a moment."):
            model, features_df = load_model_and_data()
            st.session_state.model = model
            st.session_state.features_df = features_df
            st.session_state.loaded = (model is not None and features_df is not None)

    # Check if loading was successful
    if not st.session_state.loaded or st.session_state.model is None or st.session_state.features_df is None:
        st.warning("⚠️ Please click 'Load Model & Data' to initialize the dashboard.")
        if st.session_state.model is None:
            st.error("Failed to load the ML model.")
        if st.session_state.features_df is None:
            st.error("Failed to load the feature data.")
        return

    if debug_mode:
        st.success("✅ All components loaded successfully!")
        st.write(f"Model type: {type(st.session_state.model)}")
        st.write(f"Data shape: {st.session_state.features_df.shape}")
    else:
        st.success("🎯 Ready to analyze your employability profile!")

    # Sidebar for inputs
    st.sidebar.header("📝 User Profile Input")

    # Create tabs for different input sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "🎓 Education", "💼 Career", "🛠️ Skills"])

    user_profile = {}

    with tab1:
        st.subheader("Demographic Information")
        col1, col2 = st.columns(2)

        with col1:
            user_profile['employment_rate_6_months'] = st.slider(
                "Employment Rate (6 months)",
                min_value=0.0, max_value=100.0, value=75.0, step=0.1,
                help="Employment rate 6 months after graduation (%)"
            )

            user_profile['graduation_year'] = st.number_input(
                "Graduation Year",
                min_value=2000, max_value=2030, value=2023,
                help="Year of graduation"
            )

        with col2:
            user_profile['remote_work_availability'] = st.slider(
                "Remote Work Availability",
                min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                help="Remote work availability (%)"
            )

            user_profile['is_us'] = 1 if st.checkbox("US-based", value=True) else 0

    with tab2:
        st.subheader("Education Background")
        col1, col2 = st.columns(2)

        with col1:
            degree_options = {
                "Bachelor's": {"master": 0, "phd": 0},
                "Master's": {"master": 1, "phd": 0},
                "PhD": {"master": 0, "phd": 1}
            }

            degree = st.selectbox("Degree Level", list(degree_options.keys()))
            user_profile['degree_level_master'] = degree_options[degree]["master"]
            user_profile['degree_level_phd'] = degree_options[degree]["phd"]

        with col2:
            field_options = {
                "Computer Science": "field_computer_science",
                "Engineering": "field_engineering",
                "Business": "field_business",
                "Data Science & AI": "field_data_science",
                "Social Sciences": "field_social_sciences"
            }

            field = st.selectbox("Field of Study", list(field_options.keys()))
            # Reset all field flags
            for field_key in field_options.values():
                user_profile[field_key] = 0
            # Set selected field
            user_profile[field_options[field]] = 1

    with tab3:
        st.subheader("Career Information")
        col1, col2 = st.columns(2)

        with col1:
            user_profile['average_starting_salary_usd'] = st.number_input(
                "Expected Starting Salary (USD)",
                min_value=0, max_value=200000, value=60000, step=1000,
                help="Expected starting salary in USD"
            )

            user_profile['employer_reputation_score'] = st.slider(
                "Employer Reputation Score",
                min_value=0, max_value=100, value=70,
                help="Employer reputation score (1-100)"
            )

        with col2:
            user_profile['skill_demand_score'] = st.slider(
                "Skill Demand Score",
                min_value=0, max_value=100, value=75,
                help="Demand score for your skills (1-100)"
            )

            region_options = {
                "North America": "region_north_america",
                "Europe": "region_europe",
                "Asia-Pacific": "region_asia_pacific",
                "Latin America": "region_latin_america",
                "Middle East & Africa": "region_middle_east_africa"
            }

            region = st.selectbox("Target Region", list(region_options.keys()))
            # Reset all region flags
            for region_key in region_options.values():
                user_profile[region_key] = 0
            # Set selected region
            user_profile[region_options[region]] = 1

    with tab4:
        st.subheader("Technical Skills")
        st.markdown("Select the skills you currently possess:")

        # Key skills
        skill_cols = st.columns(2)
        skills = {
            "Python": "skill_python",
            "Machine Learning": "skill_machine_learning",
            "Data Analysis": "skill_data_analysis",
            "SQL": "skill_sql",
            "Excel": "skill_excel",
            "Communication": "skill_communication",
            "Project Management": "skill_project_management",
            "Laboratory Skills": "skill_laboratory_skills"
        }

        for i, (skill_name, skill_key) in enumerate(skills.items()):
            with skill_cols[i % 2]:
                user_profile[skill_key] = 1 if st.checkbox(skill_name, value=False) else 0

    # Calculate derived features
    user_profile['reputation_x_remote'] = user_profile['employer_reputation_score'] * user_profile['remote_work_availability'] / 100
    user_profile['demand_x_reputation'] = user_profile['skill_demand_score'] * user_profile['employer_reputation_score'] / 100

    # Job role selection (simplified - set to most common)
    job_roles = {
        'job_role_data_analyst': 0,
        'job_role_machine_learning_engineer': 0,
        'job_role_software_engineer': 0,
        'job_role_business_analyst': 0,
        'job_role_project_manager': 0
    }

    # Auto-select job role based on skills and field
    if user_profile.get('skill_data_analysis') and user_profile.get('field_data_science'):
        job_roles['job_role_data_analyst'] = 1
    elif user_profile.get('skill_machine_learning') and user_profile.get('field_computer_science'):
        job_roles['job_role_machine_learning_engineer'] = 1
    elif user_profile.get('skill_python') and user_profile.get('field_computer_science'):
        job_roles['job_role_software_engineer'] = 1
    elif user_profile.get('skill_communication') and user_profile.get('field_business'):
        job_roles['job_role_business_analyst'] = 1
    elif user_profile.get('skill_project_management'):
        job_roles['job_role_project_manager'] = 1
    else:
        job_roles['job_role_data_analyst'] = 1  # Default

    user_profile.update(job_roles)

    # Prediction button
    if st.sidebar.button("🔮 Generate Prediction & Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing profile and generating recommendations..."):
            try:
                # Convert to DataFrame
                user_df = convert_profile_to_dataframe(user_profile)

                # Make prediction
                probabilities = st.session_state.model.predict_proba(user_df)[:, 1]
                probability = float(probabilities[0])
                percentage = probability * 100

                # Get risk category
                risk_category, risk_class, risk_icon = get_risk_category(probability)

                # Get recommendations
                results = get_recommendations_for_user(
                    user_df.iloc[0],
                    st.session_state.model,
                    st.session_state.features_df,
                    top_n=3,
                    user_id="dashboard_user"
                )

                # Display results
                st.success("Analysis complete!")

                # Main results in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Employability Score</h3>
                        <h2>{percentage:.1f}%</h2>
                        <p>Probability: {probability:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚠️ Risk Assessment</h3>
                        <h2 class="{risk_class}">{risk_icon} {risk_category}</h2>
                        <p>Based on current profile</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💡 Recommendations</h3>
                        <h2>{len(results['summary']['recommendations'])}</h2>
                        <p>Personalized suggestions</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommendations section
                st.header("🎯 Top Skill Recommendations")

                if results['summary']['recommendations']:
                    for rec in results['summary']['recommendations']:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{rec['rank']}: {rec['recommendation']}</h4>
                            <p><strong>Expected Improvement:</strong> +{rec['improvement']:.1f}% ({rec['improvement_pct']:.1f}%)</p>
                            <p><strong>Current Probability:</strong> {rec['current_probability']:.1%} → <strong>{rec['predicted_probability']:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific recommendations needed - your profile looks strong!")

                # Feature importance visualization
                st.header("📊 Feature Importance Insights")

                # Get feature names
                feature_names = user_df.columns.tolist()

                # Create and display plot
                fig = create_feature_importance_plot(st.session_state.model, feature_names, top_n=8)
                if fig:
                    st.pyplot(fig)
                    st.caption("Top 8 features influencing the prediction (higher values = more important)")
                else:
                    st.info("Feature importance visualization not available for this model type.")

                # Additional insights
                st.header("💡 Key Insights")

                insight_col1, insight_col2 = st.columns(2)

                with insight_col1:
                    st.subheader("Strengths")
                    strengths = []
                    if probability > 0.8:
                        strengths.append("Strong overall employability profile")
                    if user_profile.get('skill_python') or user_profile.get('skill_machine_learning'):
                        strengths.append("Technical skills are a major asset")
                    if user_profile.get('employer_reputation_score', 0) > 80:
                        strengths.append("Excellent employer reputation")
                    if user_profile.get('remote_work_availability', 0) > 70:
                        strengths.append("Good remote work flexibility")

                    if strengths:
                        for strength in strengths:
                            st.success(f"✅ {strength}")
                    else:
                        st.info("Keep building your profile!")

                with insight_col2:
                    st.subheader("Areas for Improvement")
                    improvements = []
                    if probability < 0.7:
                        improvements.append("Consider skill development programs")
                    if user_profile.get('employer_reputation_score', 0) < 60:
                        improvements.append("Target higher-reputation employers")
                    if user_profile.get('remote_work_availability', 0) < 50:
                        improvements.append("Explore remote work opportunities")
                    if not any(user_profile.get(f'skill_{skill}', 0) for skill in ['python', 'machine_learning', 'data_analysis']):
                        improvements.append("Develop technical skills")

                    if improvements:
                        for improvement in improvements:
                            st.warning(f"💡 {improvement}")
                    else:
                        st.success("Your profile is well-balanced!")

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown("*Employability AI System - Powered by Machine Learning*")
    st.markdown("*Built with Streamlit & Scikit-learn*")

if __name__ == "__main__":
    st.sidebar.warning(
        "This app should be launched with `streamlit run src/dashboard/app.py` or `python run_dashboard.py`."
    )
    print("Please launch the dashboard using Streamlit: streamlit run src/dashboard/app.py")
    sys.exit(1)
