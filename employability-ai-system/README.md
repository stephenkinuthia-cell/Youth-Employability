# Employability AI System

This folder contains the machine learning workspace used by the Django employability platform.

For the full application setup, local deployment steps, account roles, and Django run instructions, use the root project `README.md`.

## What this folder contains

- model training scripts
- recommendation pipeline code
- saved trained models
- processed training data

## Main command

From the repository root, retrain the model with:

```powershell
python employability-ai-system\quick_train.py
```

This generates the model artifact used by Django:

- `employability-ai-system/models/trained/best_model.pkl`

## Dependency note

The shared project dependencies are listed in the root `requirements.txt`.
