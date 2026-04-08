# Employability AI System

A comprehensive machine learning pipeline for predicting graduate employability and providing personalized skill recommendations.

## Project Overview

This system implements a complete ML pipeline that:

1. **Data Processing**: Cleans and validates raw employability data
2. **Feature Engineering**: Creates domain-driven features from raw data
3. **Model Training**: Trains and evaluates multiple ML models
4. **Fairness Analysis**: Evaluates model performance across demographic groups
5. **Personalized Recommendations**: Provides counterfactual-based skill recommendations

## Security & Git Configuration

### Files Excluded from Version Control

The following sensitive and large files are excluded from Git to protect data privacy and reduce repository size:

**Data Files:**

- `data/raw/` - Raw dataset files
- `data/processed/` - Processed features and cleaned data
- `data/interim/` - Intermediate processing files
- `data/external/` - External data sources

**Model Files:**

- `models/trained/` - Trained ML models (.pkl, .joblib files)
- `models/*.pkl` - All pickle model files
- `models/*.joblib` - All joblib model files

**Sensitive Configuration:**

- `config/secrets.yaml` - API keys and secrets
- `config/api_keys.yaml` - Authentication credentials
- `.env*` - Environment variables

**Generated Outputs:**

- `reports/figures/` - Generated plots and visualizations
- `reports/results/` - Model evaluation results
- `reports/recommendations/` - Generated recommendation files
- `test_output.txt` - Test output logs

**Cache & Temporary Files:**

- `__pycache__/` - Python bytecode cache
- `.streamlit/` - Streamlit cache
- `*.log` - Log files
- `*.tmp` - Temporary files

### Repository Structure

Only the following are tracked in Git:

- Source code (`src/`)
- Configuration templates (`config/` except secrets)
- Documentation (`README.md`)
- Requirements (`requirements.txt`)
- Test files (`tests/`)
- Empty directory placeholders (`.gitkeep` files)

### Setup After Cloning

After cloning this repository, you'll need to:

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data and models:**
   - Place your dataset in `data/raw/`
   - Run the data processing pipeline to generate features
   - Train models or download pre-trained models
   - Update configuration files as needed

3. **Configure secrets:**
   - Copy `config/config.yaml` to set up your configuration
   - Add any required API keys to `config/secrets.yaml` (not tracked in Git)

## Current Status

✅ **All 6 sprints completed and tested**

- Data pipeline (3500 × 103 features)
- Model training (GradientBoosting: 97.14% ROC-AUC)
- Evaluation & fairness analysis (regional disparities detected)
- Counterfactual recommendation engine (functional)
- FastAPI service (3 endpoints, tested and working)
- Streamlit dashboard (interactive UI with predictions and recommendations)

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Interactive Dashboard

```bash
python run_dashboard.py
```

**Output**: Streamlit dashboard opens at http://localhost:8502

> Note: Do not run `src/dashboard/app.py` directly with `python`; launch it via `streamlit run src/dashboard/app.py` or via `run_dashboard.py` to avoid Streamlit runtime warnings.

- Input your profile information across 4 tabs
- Get real-time employability predictions
- View personalized skill recommendations
- Explore feature importance visualizations

### Run API Server

```bash
python run_api.py
```

**Output**: FastAPI server starts on http://localhost:8000

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

### Run Single User Recommendations

```bash
python test_recommendations.py
```

**Output**: Top 3 skill recommendations for user_0 with improvement scores

### Run Batch Recommendations

```bash
python run_recommendations.py
```

**Output**: Recommendations for 20 sample users saved to `reports/recommendations/`

### Run Model Evaluation

```bash
python run_evaluation.py
```

**Output**: Comprehensive evaluation results saved to `reports/results/model_evaluation.csv`

### Start API Server

```bash
python run_api.py
```

**Output**: FastAPI server starts on http://localhost:8000

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

## Project Structure

```
employability-ai-system/
├── main.py                    # Main entry point (empty)
├── run_api.py                 # API server launcher
├── run_dashboard.py           # Dashboard launcher
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # Configuration settings
├── data/
│   ├── raw/                  # Raw dataset
│   ├── processed/            # Cleaned/processed data
│   └── interim/              # Intermediate processing files
├── models/
│   ├── trained/              # Saved model files
│   └── metadata/             # Model metadata
├── src/
│   ├── api/                  # FastAPI service
│   │   ├── app.py            # FastAPI application
│   │   ├── routes.py         # API endpoints
│   │   └── schemas.py        # Pydantic models
│   ├── dashboard/            # Streamlit dashboard
│   │   └── app.py            # Interactive UI
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering
│   ├── models/               # ML training & evaluation
│   ├── evaluation/           # Model evaluation & fairness
│   └── recommendation/       # Skill recommendation engine
├── reports/
│   ├── results/              # Evaluation results
│   └── recommendations/      # Generated recommendations
└── tests/                    # Unit tests
```

## Key Components

### 1. Data Pipeline (`src/data/`)

- **load_data.py**: Load raw CSV data
- **validate_data.py**: Data quality checks
- **preprocess.py**: Cleaning, imputation, outlier removal
- **split_data.py**: Train/validation/test splits

### 2. Feature Engineering (`src/features/`)

- **build_features.py**: Domain-driven feature creation
- **transform_features.py**: Encoding, scaling, preprocessing
- **select_features.py**: Feature selection and importance

### 3. Model Training (`src/models/`)

- **train.py**: Train multiple baseline models
- **evaluate.py**: Model performance metrics
- **tune.py**: Hyperparameter optimization
- **registry.py**: Model persistence
- **predict.py**: Prediction interface

### 4. Evaluation & Fairness (`src/evaluation/`)

- **metrics.py**: Classification metrics
- **subgroup_analysis.py**: Performance by region
- **fairness.py**: Bias detection across groups

### 5. Recommendations (`src/recommendation/`)

- **skill_gap.py**: Identify missing/weak skills
- **simulate.py**: Counterfactual simulations
- **rank.py**: Rank recommendations by impact
- **recommend.py**: Orchestrate recommendation workflow
- **pipeline.py**: Batch processing

### 6. API Service (`src/api/`)

- **app.py**: FastAPI application with CORS and error handling
- **routes.py**: API endpoints for predictions and recommendations
- **schemas.py**: Pydantic models for request/response validation

### 7. Interactive Dashboard (`src/dashboard/`)

- **app.py**: Streamlit application with tabbed input forms
- **Features**: Real-time predictions, risk assessment, skill recommendations, feature importance visualization

## Usage Examples

### Single User Recommendations

```python
from src.recommendation.recommend import get_recommendations_for_user
from src.models.registry import load_best_model
import pandas as pd

# Load data and model
features_df = pd.read_csv('data/processed/final_features.csv')
model = load_best_model()

# Get recommendations for first user
user_profile = features_df.iloc[0]
results = get_recommendations_for_user(user_profile, model, features_df, top_n=3)

print(results['recommendations_df'])
print(results['summary'])
```

### Batch Processing

```python
from src.recommendation.pipeline import run_recommendation_pipeline

# Generate recommendations for 50 users
recommendations_df, summaries_df = run_recommendation_pipeline(sample_size=50)

# Save results
recommendations_df.to_csv('reports/recommendations/skill_recommendations.csv')
summaries_df.to_csv('reports/recommendations/recommendations_summary.csv')
```

## Model Performance

**Best Model**: GradientBoostingClassifier

- **ROC-AUC**: 97.14%
- **Accuracy**: 91.57%
- **Precision**: 0.92
- **Recall**: 0.90
- **F1-Score**: 0.91

**Fairness Analysis**: Detected disparities in error rates across 5 regions (max 36.7% difference).

## Recommendation Engine

**Methodology**: Counterfactual simulations

- Identifies missing skills (value=0) and weak features (<50)
- Simulates skill acquisition/feature improvement
- Predicts probability changes using trained model
- Ranks by improvement impact

**Example Output**:

```
Top 3 Recommendations for user_0:
1. Laboratory Skills: +24.81% (0.531 → 0.663)
2. Data Analysis: +18.44% (0.531 → 0.629)
3. Project Management: +16.13% (0.531 → 0.617)
```

## API Endpoints

The system includes a FastAPI service for programmatic access:

### Start the API Server

```bash
python run_api.py
```

### Available Endpoints

#### POST `/api/v1/predict`

Predict employability probability for a user profile.

**Request Body**:

```json
{
  "user_profile": {
    "employment_rate_6_months": 85.6,
    "average_starting_salary_usd": 50000,
    "employer_reputation_score": 75,
    "skill_demand_score": 80,
    "remote_work_availability": 60,
    "graduation_year": 2023,
    "region_north_america": 1,
    "field_computer_science": 1,
    "skill_python": 1,
    "skill_machine_learning": 0
  }
}
```

**Response**:

```json
{
  "employability_probability": 0.853,
  "employability_percentage": 85.3,
  "confidence_score": 0.853
}
```

#### POST `/api/v1/recommend`

Get personalized skill recommendations.

**Request Body**:

```json
{
  "user_profile": { ... },
  "top_n": 3,
  "user_id": "user_123"
}
```

**Response**:

```json
{
  "user_id": "user_123",
  "total_recommendations": 3,
  "recommendations": [
    {
      "rank": 1,
      "recommendation": "Acquire: Laboratory Skills",
      "category": "Skill",
      "improvement": 0.1319,
      "improvement_pct": 24.81,
      "current_probability": 0.5314,
      "predicted_probability": 0.6632
    }
  ],
  "summary": {
    "total_potential_improvement": 0.308,
    "average_improvement": 0.1027,
    "recommendation_count": 3
  }
}
```

#### GET `/api/v1/health`

Health check endpoint.

## Interactive Dashboard

The system includes a Streamlit-based interactive dashboard for real-time predictions and recommendations.

### Start the Dashboard

```bash
python run_dashboard.py
```

### Dashboard Features

#### Input Form (4 Tabs)

- **Demographics**: Employment rate, salary, employer reputation, remote work availability
- **Education**: Graduation year, field of study, region
- **Career**: Career goals, industry preferences
- **Skills**: Technical and soft skills (Python, ML, Project Management, etc.)

#### Results Display

- **Employability Probability**: Percentage and risk category (Low/Medium/High)
- **Risk Assessment**: Color-coded risk levels with visual indicators
- **Skill Recommendations**: Top 3 personalized recommendations with improvement metrics
- **Feature Importance**: Horizontal bar chart showing most influential factors

#### User Experience

- **Responsive Design**: Clean, professional interface with custom styling
- **Real-time Updates**: Instant predictions as you modify inputs
- **Visual Feedback**: Color-coded risk categories and progress indicators
- **Error Handling**: Graceful handling of invalid inputs and system errors

## Configuration

Edit `config/config.yaml` to modify:

- Model hyperparameters
- Feature engineering settings
- Evaluation parameters
- Recommendation thresholds

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Features

1. Add feature creation logic to `src/features/build_features.py`
2. Update preprocessing in `src/features/transform_features.py`
3. Retrain models: `python quick_train.py`

### Custom Recommendations

Modify `src/recommendation/simulate.py` to add new improvement types or thresholds.

## Dependencies

- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.2.0
- joblib>=1.1.0
- fastapi>=0.104.0
- uvicorn>=0.24.0
- streamlit>=1.28.0
- pydantic>=2.0.0
- matplotlib>=3.6.0

## License

This project is for educational and research purposes.
