# AI-Powered Employability and Skill Recommendation Platform

This repository contains a Django web platform plus a trained machine learning workflow for predicting graduate employability, recommending skill improvements, and matching candidates to jobs.

## What the app does

- Lets every person create an account as a `User`, `Employer`, or `Admin`
- Predicts employability likelihood from education, skills, and experience inputs
- Generates personalized skill recommendations and suggested career paths
- Allows employers to publish jobs and view candidate results
- Gives admins an in-app control panel for user management, support messages, traffic, and platform effectiveness
- Tracks support issues so users and employers can see when an admin has reviewed them

## Main folders

```text
Youth-Employability/
|-- employabilityapp/           # Django project and templates
|-- employability-ai-system/    # Training code and saved ML models
|-- global_graduate_employability_index.csv
|-- model_ready_data.csv
|-- requirements.txt
`-- README.md
```

## Local deployment guide

These steps deploy and run the platform on a local machine.

### 1. Prerequisites

- Python 3.11 or newer
- `pip`
- Git optional, if you are cloning the project

### 2. Open the project folder

If you already have the project, open the root folder:

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

### 4. Install the dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```

### 5. Train or refresh the machine learning model

The Django app is wired to load the saved trained model from:

`employability-ai-system/models/trained/best_model.pkl`

If you want to rebuild that model locally, run:

```powershell
python employability-ai-system\quick_train.py
```

This command uses the prepared dataset and writes the trained model files into:

- `employability-ai-system/models/trained/`
- `employability-ai-system/data/processed/`

If `best_model.pkl` is already present and you do not need to retrain, you can skip this step.

### 6. Apply the Django database migrations

Move into the Django project folder:

```powershell
cd employabilityapp
```

Then run:

```powershell
python manage.py makemigrations
python manage.py migrate
```

The app uses SQLite by default for local development, so no extra database setup is required.

### 7. Start the Django development server

Still inside `employabilityapp/`, run:

```powershell
python manage.py runserver
```

The local site will be available at:

`http://127.0.0.1:8000/`

### 8. Create and use accounts

Open the browser and go to:

- Home page: `http://127.0.0.1:8000/`
- Register: `http://127.0.0.1:8000/users/register/`
- Login: `http://127.0.0.1:8000/users/login/`

During registration, a person must choose one of these roles:

- `User`
- `Employer`
- `Administrator`

Notes:

- Employers can add job information during registration.
- Users can only access their own assessment results.
- Employers and admins can view other users' results.
- Admins manage the system through the in-app dashboard and do not need to use Django admin.

### 9. Admin registration code

Admin registration requires the app's admin signup code.

In this project, it is generated in `employabilityapp/employabilityapp/settings.py` from the Django `SECRET_KEY`.

Current logic:

```python
ADMIN_SIGNUP_CODE = hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:12].upper()
```

With the current `SECRET_KEY`, the code resolves to:

```text
4AB9C3A615C8
```

If you change `SECRET_KEY`, the admin signup code will also change.

### 10. Access the main features

After logging in:

- Users can submit employability assessments and view their own results
- Employers can browse users' results and manage jobs
- Users can browse available employers and jobs
- Users and employers can contact the admin from the app
- Admins can review support messages and track usage at `/analytics/`

## Useful local commands

From `employabilityapp/`:

```powershell
python manage.py check
python manage.py migrate
python manage.py runserver
```

From the repository root:

```powershell
python employability-ai-system\quick_train.py
```

## Technology used

- Django 5
- SQLite for local development
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Bootstrap-based HTML templates

## Notes for future deployment

- Local development uses SQLite and Django's built-in development server.
- For production, you should move the `SECRET_KEY` and debug settings into environment variables.
- For production, PostgreSQL and a proper static-file setup are recommended.
