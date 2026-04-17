# Youth Employability Project

## Overview

This repository contains a dual-stack employability solution combining an AI model pipeline and a Django web application. It is built to analyze graduate employability data, train classification models for employment risk, generate personalized skill recommendations, and deliver prediction results through a web interface.

### What it solves

- Builds a machine learning pipeline for graduate employability prediction and recommendation.
- Produces actionable skill-improvement recommendations and career path suggestions.
- Surfaces prediction reports via a Django app for authenticated users.

### Who it is for

- Data scientists and ML engineers evaluating employability models.
- Product teams building graduate career support platforms.
- Developers integrating prediction-driven career advice into web applications.

## Project Structure

```
Youth-Employability/
├── data/
│   └── processed/
│       ├── model_predictions.csv
│       └── processed_graduate_employability.csv
├── employability-ai-system/
│   ├── config/
│   │   └── config.yaml
│   ├── data/
│   │   ├── external/
│   │   ├── interim/
│   │   ├── processed/
│   │   │   ├── cleaned_data.csv
│   │   │   └── final_features.csv
│   │   └── raw/
│   ├── figures/
│   ├── main.py
│   ├── quick_train.py
│   ├── run_evaluation.py
│   ├── run_recommendations.py
│   ├── simple_train.py
│   ├── src/
│   │   ├── data/
│   │   │   ├── load_data.py
│   │   │   ├── preprocess.py
│   │   │   ├── split_data.py
│   │   │   └── validate_data.py
│   │   ├── evaluation/
│   │   │   ├── fairness.py
│   │   │   ├── metrics.py
│   │   │   ├── pipeline.py
│   │   │   └── subgroup_analysis.py
│   │   ├── features/
│   │   │   ├── build_features.py
│   │   │   ├── select_features.py
│   │   │   ├── transform_features.py
│   │   │   └── pipeline.py
│   │   ├── models/
│   │   │   ├── evaluate.py
│   │   │   ├── pipeline.py
│   │   │   ├── predict.py
│   │   │   ├── registry.py
│   │   │   ├── train.py
│   │   │   └── tune.py
│   │   ├── recommendation/
│   │   │   ├── pipeline.py
│   │   │   ├── rank.py
│   │   │   ├── recommend.py
│   │   │   ├── simulate.py
│   │   │   └── skill_gap.py
│   │   └── utils/
│   ├── reports/
│   │   ├── recommendation_experiments/
│   │   └── results/
│   └── notebooks/
│       ├── 01_eda.ipynb
│       ├── 02_feature_engineering.ipynb
│       ├── 03_modelling.ipynb
│       ├── 04_evaluation.ipynb
│       └── 05_recommendation_experiments.ipynb
├── employabilityapp/
│   ├── manage.py
│   ├── db.sqlite3
│   ├── employabilityapp/
│   │   ├── asgi.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── AIrecommendation/
│   ├── analytics_dashboard/
│   ├── jobs/
│   ├── prediction/
│   ├── recommendations/
│   ├── templates/
│   │   ├── analytics_dashboard/
│   │   ├── base.html
│   │   ├── jobs/
│   │   ├── prediction/
│   │   ├── recommendations/
│   │   └── users/
│   └── users/
├── generated_plots/
│   ├── advanced_model_comparison.png
│   ├── classification_model_performance.png
│   ├── correlation_heatmap.png
│   ├── regression_model_performance.png
│   └── ...
├── cleaned_employability_analysis.ipynb
├── Employability_analysis.ipynb
├── requirements.txt
└── tmp_inspect_data.py
```

### Folder and file purposes

- `data/processed/`

  - Contains cleaned and derived dataset outputs.
  - `model_predictions.csv` and `processed_graduate_employability.csv` are prepared datasets used for analysis and downstream model workflows.
- `employability-ai-system/`

  - Core machine learning and recommendation pipeline.
  - `main.py`: central entrypoint that documents available commands and verifies file checks.
  - `quick_train.py`: trains baseline classification models and saves the best performer.
  - `run_evaluation.py`: executes model evaluation and fairness analysis.
  - `run_recommendations.py`: executes batch recommendation generation.
  - `src/data/`: data ingestion, cleaning, splitting, and validation logic.
  - `src/features/`: feature engineering pipeline for dataset transformation.
  - `src/models/`: model training, evaluation, prediction, registry, and tuning logic.
  - `src/evaluation/`: overall evaluation pipeline, subgroup analysis, and fairness assessment.
  - `src/recommendation/`: skill-gap analysis, simulation, ranking, and recommendation output.
  - `reports/`: saved evaluation and recommendation CSV outputs.
  - `notebooks/`: exploratory data analysis and modelling notebooks.
- `employabilityapp/`

  - Django application delivering prediction forms, result pages, recommendation summaries, and analytics dashboards.
  - `manage.py`: Django admin and local server control.
  - `employabilityapp/settings.py`: Django settings, SQLite configuration, installed apps, templates, and URLs.
  - `employabilityapp/urls.py`: routes requests to prediction, recommendations, jobs, analytics, and user modules.
  - `prediction/`: main application logic for employability assessment forms, summary pages, detail views, report download and prediction services.
  - `recommendations/`: app for storing and showing recommendation models results.
  - `jobs/`: job posting logic used for matching assessments to active roles.
  - `analytics_dashboard/`: dashboard pages and middleware for tracking application usage.
  - `users/`: authentication, registration, permissions, profile forms, and user management.
- `generated_plots/`

  - Exported visualization artifacts from the analysis notebooks.
  - Includes model performance charts, correlation heatmaps, cluster analysis dashboards, and more.
- `cleaned_employability_analysis.ipynb` and `Employability_analysis.ipynb`

  - Jupyter notebooks for data cleaning, exploration, and model development.
- `requirements.txt`

  - Lists Python dependencies required for both the Django app and the AI/pipeline stack.
- `tmp_inspect_data.py`

  - Utility script likely used for inspecting dataset characteristics during development.

## Installation and Setup

### Prerequisites

- Python 3.12 (or compatible 3.x interpreter)
- `pip` package manager
- Local development environment with access to repository files

### Install dependencies

```bash
cd c:\Users\kinut\Youth-Employability
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Machine learning pipeline setup

1. Verify that processed dataset exists:
   - `employability-ai-system/data/processed/final_features.csv`
2. Run model training:
   - `python employability-ai-system/quick_train.py`
3. Evaluate the trained model:
   - `python employability-ai-system/run_evaluation.py`
4. Generate sample recommendations:
   - `python employability-ai-system/run_recommendations.py`

### Django app setup

1. Change directory into the web app:
   - `cd employabilityapp`
2. Apply migrations:
   - `python manage.py migrate`
3. Create a superuser if needed:
   - `python manage.py createsuperuser`
4. Start the development server:
   - `python manage.py runserver`
5. Open the app in a browser:
   - `http://127.0.0.1:8000/`

## Methodology / Approach

### Data and model approach

- The AI system uses a processed dataset built from `employability-ai-system/data/processed/final_features.csv`.
- The target variable is `Employment_Rate_12_Months (%)`.
- The model training pipeline binarizes this target at an 85% employment threshold, producing a classification label.
- Baseline classifiers trained in `src/models/train.py` include:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- `quick_train.py` evaluates these models and selects `GradientBoosting` as the best performer by ROC-AUC.

### Evaluation and fairness

- `src/evaluation/pipeline.py` computes:
  - accuracy
  - precision
  - recall
  - F1-score
  - ROC-AUC
  - confusion matrix
- It also runs subgroup analysis by region and computes fairness disparity metrics to flag potential bias across demographic segments.

### Recommendation logic

- The recommendation engine uses a three-stage workflow:
  1. `skill_gap.py` identifies missing skills and weak profile features.
  2. `simulate.py` estimates the benefit of adding skills or improving feature values.
  3. `rank.py` ranks recommendations by predicted improvement.
- The output includes skill recommendations, portfolio guidance, and career path suggestions.

### Django app prediction service

- `prediction/services.py` builds a feature frame from user input and attempts to load the trained model.
- If no saved model is available, the app falls back to a heuristic scoring mechanism.
- Model scores are calibrated with a 65/35 weighting between model probability and heuristic score.
- The service returns:
  - `employability_score`
  - `risk_category`
  - `strengths` / `weaknesses`
  - recommended skill actions
  - career path suggestions
  - HTML report content

## Usage

### AI system commands

- Check system status:
  - `python employability-ai-system/main.py --status`
- Train and save models:
  - `python employability-ai-system/quick_train.py`
- Evaluate models and fairness:
  - `python employability-ai-system/run_evaluation.py`
- Generate a single user recommendation demo:
  - `python employability-ai-system/test_recommendations.py`
- Generate batch recommendations:
  - `python employability-ai-system/run_recommendations.py`

### Django application workflow

- Access landing page at `/`.
- Log in via `/users/` pages.
- Submit employability profile through the prediction dashboard.
- View assessment details, matched jobs, AI report, and recommendations.
- Download the HTML report from the assessment detail page.

## Results / Outputs

### AI system outputs

- Saved evaluation results in `employability-ai-system/reports/results/model_evaluation.csv`
- Saved recommendation outputs in `employability-ai-system/reports/recommendations/skill_recommendations.csv`
- Saved summary output in `employability-ai-system/reports/recommendations/recommendations_summary.csv`

### Visual outputs

- Analysis notebooks export charts to `generated_plots/` including:
  - model performance comparisons
  - regression and classification diagnostics
  - correlation heatmaps
  - cluster analysis dashboards

### Django outputs

- Prediction assessment records in SQLite at `employabilityapp/db.sqlite3`
- Stored `SkillRecommendation`, `CareerPathSuggestion`, and `AIReport` records for assessments

## Key Insights

- The repository combines a standalone AI modeling pipeline with a production-style Django front end.
- The training pipeline prefers Gradient Boosting for classification, reflecting higher ROC-AUC compared to baseline models.
- Recommendation logic is grounded in skill gaps and simulated impact rather than only rule-based output.
- The application emphasizes both model performance and fairness across regional subgroups.

## Limitations

- The repository does not include all raw source datasets within the root project tree. Some expected dataset files referenced by the prediction service are missing, such as `global_graduate_employability_index.csv` and `model_ready_data.csv`.
- The Django app currently relies on SQLite and debug mode, which is not production-ready.
- The current scoring model is a hybrid of trained model output and heuristic fallback, so results depend on available saved model artifacts.
- There is no root-level documentation describing deployment or CI, so setup requires inspection of scripts and code.

## Future Improvements

- Add a root-level `README.md` and documentation for dataset prerequisites and deployment.
- Replace SQLite with a production-grade database for the Django app.
- Add explicit dataset ingestion scripts to `employability-ai-system/data/raw/` and integrate them into pipeline automation.
- Implement model hyperparameter search or cross-validation inside `src/models/tune.py`.
- Add unit tests for the Django prediction flow and AI recommendation endpoints.
- Add a dedicated API or Streamlit dashboard for real-time recommendations and model explainability.

## Technologies Used

- Python
- Django 5.2.x
- pandas
- numpy
- scikit-learn
- joblib
- FastAPI
- Uvicorn
- Pydantic
- Streamlit
- Matplotlib
