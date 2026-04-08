"""
Evaluation module for machine learning models.

This module evaluates trained models using classification metrics including
accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a single model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted'),
    }

    # ROC-AUC (only for binary classification)
    if y_proba is not None and len(np.unique(y_test)) == 2:
        metrics['ROC_AUC'] = roc_auc_score(y_test, y_proba)
    else:
        metrics['ROC_AUC'] = np.nan

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['Confusion_Matrix'] = cm.tolist()  # Convert to list for DataFrame

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['Classification_Report'] = report

    logger.info(f"{model_name} - Accuracy: {metrics['Accuracy']:.4f}, ROC-AUC: {metrics['ROC_AUC']:.4f}")
    return metrics

def evaluate_all_models(models: dict, X_test, y_test):
    """
    Evaluate all models and return results as DataFrame.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target

    Returns:
        DataFrame with evaluation results
    """
    results = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    logger.info("Evaluation complete for all models")
    return df_results

def print_evaluation_summary(df_results: pd.DataFrame):
    """
    Print a summary of evaluation results.

    Args:
        df_results: DataFrame with evaluation results
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)

    # Sort by ROC-AUC
    df_sorted = df_results.sort_values('ROC_AUC', ascending=False)

    for _, row in df_sorted.iterrows():
        print(f"\n{row['Model']}:")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

        # Confusion Matrix
        cm = row['Confusion_Matrix']
        print(f"  Confusion Matrix:")
        print(f"    [[{cm[0][0]}, {cm[0][1]}]")
        print(f"     [{cm[1][0]}, {cm[1][1]}]]")

if __name__ == "__main__":
    # Example usage (would normally import from train.py)
    from train import run_training_pipeline

    models, X_train, X_test, y_train, y_test = run_training_pipeline()
    results_df = evaluate_all_models(models, X_test, y_test)
    print_evaluation_summary(results_df)
