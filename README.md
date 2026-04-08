# AI-Powered Employability and Skill Recommendation Platform

This repository contains a Django web application and a machine learning pipeline for predicting graduate employability, generating personalized skill recommendations, suggesting career paths, and supporting employer job matching.

## Project overview

The platform combines two major parts:

- A Django web app for authentication, role-based access, dashboards, jobs, analytics, and support workflows
- A machine learning workspace for data processing, feature engineering, model training, evaluation, and recommendation logic

## Core capabilities

- Role-based accounts for `User`, `Employer`, and `Administrator`
- Employability prediction dashboard powered by the trained ML model
- Personalized skill recommendations and career path suggestions
- Job posting and job browsing workflows
- In-app admin dashboard for account management, traffic monitoring, and effectiveness tracking
- Support messaging between users or employers and the admin team

## Current status

- End-to-end Django platform implemented
- Trained model integrated into the prediction flow
- In-app admin dashboard available without needing Django admin
- Role-based access rules enforced for results and management views
- Support inbox and status tracking implemented

## Repository structure

```text
Youth-Employability/
|-- employabilityapp/                    # Django project
|   |-- analytics_dashboard/
|   |-- jobs/
|   |-- prediction/
|   |-- recommendations/
|   |-- templates/
|   `-- users/
|-- employability-ai-system/            # ML pipeline and trained model workspace
|   |-- data/
|   |-- models/
|   |-- src/
|   `-- quick_train.py
|-- global_graduate_employability_index.csv
|-- model_ready_data.csv
|-- requirements.txt
`-- README.md
```

## Machine learning pipeline

The ML workspace supports:

1. Data loading and preprocessing
2. Feature engineering and transformation
3. Model training and evaluation
4. Fairness and subgroup analysis
5. Recommendation generation
6. Model persistence for Django inference

### ML structure

```text
employability-ai-system/
|-- quick_train.py
|-- data/
|   |-- processed/
|   |-- raw/
|   `-- interim/
|-- models/
|   |-- trained/
|   `-- metadata/
`-- src/
    |-- api/
    |-- dashboard/
    |-- data/
    |-- evaluation/
    |-- features/
    |-- models/
    `-- recommendation/
```

### Main ML components

- `src/data/`: dataset loading, validation, and preprocessing
- `src/features/`: feature building, transformation, and selection
- `src/models/`: training, evaluation, tuning, persistence, and prediction
- `src/evaluation/`: metrics, fairness checks, and subgroup analysis
- `src/recommendation/`: skill-gap detection, simulation, ranking, and recommendation workflows

## Jupyter notebook

The repository also includes a Jupyter notebook:

- `Employability_analysis.ipynb`

This notebook is the main exploratory analysis workspace for the project. It captures the earlier research and experimentation process that informed both the machine learning pipeline and the Django platform.

### What the notebook is about

The notebook focuses on understanding graduate employability patterns from the source dataset and translating those findings into useful prediction features and product ideas.

It is designed to help answer questions such as:

- which academic and market factors are most associated with stronger employability outcomes
- how salary, demand, reputation, and remote-work availability relate to graduate success
- which fields of study appear to have stronger or weaker outcomes
- what kinds of skill patterns are common among higher-employability profiles
- how the dataset should be cleaned and prepared before model training

### What the notebook typically contains

The notebook is used for a full exploratory and analytical workflow, including:

- loading the employability dataset into pandas
- checking data quality, missing values, and column structure
- exploring categorical fields such as degree level, region, and field of study
- summarizing numeric indicators like salary, employer reputation, remote-work availability, and skill demand
- generating charts and visual summaries to reveal patterns and outliers
- examining relationships between candidate features and employability-related outcomes
- preparing features that can later be reused in the ML scripts
- testing ideas for modeling, clustering, scoring, or recommendation logic

### How it relates to the rest of the repository

The notebook is complementary to the production code:

- it is the research and experimentation environment
- the code in `employability-ai-system/` is the reusable ML pipeline built from that experimentation
- the code in `employabilityapp/` is the deployed Django application that consumes the trained model

In other words:

- use the notebook to explore, analyze, visualize, and prototype
- use the ML scripts to train and persist repeatable models
- use the Django app to serve predictions and recommendations to real users through the web interface

### Typical outputs from the notebook

Depending on which cells are run, the notebook may produce:

- summary tables
- descriptive statistics
- correlation views
- distribution plots
- outlier comparisons
- feature-oriented observations
- intermediate datasets that help guide feature engineering and model-building decisions

Some generated plots and CSV outputs in the repository come from this broader analysis workflow, including files such as:

- `correlation_heatmap.png`
- `outlier_plots.png`
- `outlier_treatment_comparison.png`
- `salary_by_field.png`
- `classification_model_results.csv`
- `regression_model_results.csv`

### When to use the notebook

The notebook is especially useful when you want to:

- understand the dataset before retraining the model
- validate whether the web-app inputs still match the original analysis assumptions
- experiment with new features or scoring approaches
- inspect trends for presentation or academic reporting
- compare data insights with what the live Django application is showing

If you want to run the notebook locally, activate your virtual environment and install Jupyter if needed:

```powershell
pip install jupyter
jupyter notebook
```

Then open `Employability_analysis.ipynb` from the browser interface.

The notebook complements the production Django app and the reusable ML scripts in `employability-ai-system/`, but it is mainly intended for analysis, experimentation, and reporting.

## Local deployment guide

These steps deploy and run the full platform on a local machine.

### 1. Prerequisites

- Python 3.11 or newer
- `pip`
- Git optional, if you are cloning the project

### 2. Open the project folder

```powershell
cd path\to\Youth-Employability
```

### 3. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```

### 5. Train or refresh the machine learning model

The Django app loads the trained model from:

`employability-ai-system/models/trained/best_model.pkl`

To generate or refresh it locally, run from the repository root:

```powershell
python employability-ai-system\quick_train.py
```

This writes updated artifacts into:

- `employability-ai-system/models/trained/`
- `employability-ai-system/data/processed/`

If the trained model already exists and you do not need to rebuild it, you can skip this step.

### 6. Apply Django migrations

Move into the Django project folder:

```powershell
cd employabilityapp
```

Run:

```powershell
python manage.py makemigrations
python manage.py migrate
```

The project uses SQLite by default for local development, so no separate database setup is required.

### 7. Start the local server

From inside `employabilityapp/`:

```powershell
python manage.py runserver
```

Open:

`http://127.0.0.1:8000/`

### 8. Register accounts and use the app

Important routes:

- Home: `http://127.0.0.1:8000/`
- Register: `http://127.0.0.1:8000/users/register/`
- Login: `http://127.0.0.1:8000/users/login/`
- Admin dashboard: `http://127.0.0.1:8000/analytics/`

Available roles:

- `User`
- `Employer`
- `Administrator`

Role behavior:

- Users can create assessments and only view their own results
- Employers can post jobs and view candidate results
- Users can browse employers and listed jobs
- Admins can manage users, employers, support messages, and usage analytics inside the app

### 9. Admin signup code

Admin registration requires a special signup code generated in:

`employabilityapp/employabilityapp/settings.py`

Current logic:

```python
ADMIN_SIGNUP_CODE = hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:12].upper()
```

With the current local `SECRET_KEY`, the code is:

```text

```

If you change `SECRET_KEY`, the generated admin code will also change.

## Useful commands

From the repository root:

```powershell
pip install -r requirements.txt
python employability-ai-system\quick_train.py
pip install jupyter
jupyter notebook
```

From `employabilityapp/`:

```powershell
python manage.py check
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

## Model performance

The current trained workflow achieved strong results during local training:

- Best model: `GradientBoostingClassifier`
- ROC-AUC: about `0.9723`

The web app uses the saved trained model and calibrates scores at inference time so manually entered profiles behave more realistically in the UI.

## Recommendation methodology

The recommendation layer is based on model-informed profile gaps and field-specific expectations.

It currently supports:

- skill-gap detection
- targeted upskilling suggestions
- career-path recommendations
- report-ready strengths and weaknesses summaries

The ML workspace also contains more advanced recommendation modules under:

- `employability-ai-system/src/recommendation/`

## Legacy ML utilities in the repository

The repository still contains supporting ML code for:

- FastAPI service scaffolding
- Streamlit dashboard scaffolding
- evaluation scripts
- recommendation batch scripts

Examples:

```powershell
python employability-ai-system\run_evaluation.py
python employability-ai-system\run_recommendations.py
```

These scripts are secondary to the Django app but remain useful for experimentation and model development.

## Technology stack

- Django 5
- SQLite for local development
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Streamlit
- Bootstrap HTML templates

## Notes for future production deployment

- Move `SECRET_KEY` and other secrets into environment variables
- Set `DEBUG = False`
- Expand `ALLOWED_HOSTS`
- Replace SQLite with PostgreSQL
- Configure static files for production
- Add a production WSGI or ASGI server such as Gunicorn, Uvicorn, or IIS-compatible hosting depending on target environment

## License

This project is for educational and research purposes.
