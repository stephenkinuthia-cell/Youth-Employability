"""
Quick test to train and save the best model.
"""

from src.models.train import run_training_pipeline
from src.models.evaluate import evaluate_all_models
from src.models.registry import save_best_model, save_all_models
import pandas as pd

# Run training
models, X_train, X_test, y_train, y_test = run_training_pipeline()

# Evaluate
results = evaluate_all_models(models, X_test, y_test)

# Select best (GradientBoosting has highest ROC-AUC)
best_model_name = 'GradientBoosting'
best_model = models[best_model_name]
best_score = results[results['Model'] == best_model_name]['ROC_AUC'].iloc[0]

print(f"Best model: {best_model_name} with ROC-AUC: {best_score:.4f}")

# Save
save_best_model(best_model, best_model_name)
save_all_models(models)

print("Models saved successfully!")