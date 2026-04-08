"""
Model metrics computation module.

This module provides comprehensive evaluation metrics for classification models
including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from typing import Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')

    # ROC-AUC
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    else:
        metrics['roc_auc'] = np.nan

    return metrics

def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generate confusion matrix and related statistics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        cm_dict = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
    else:
        cm_dict = {'confusion_matrix': cm.tolist()}

    return cm_dict

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with classification report
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

def compute_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC curve data.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities for positive class

    Returns:
        Dictionary with ROC curve data
    """
    if len(np.unique(y_true)) != 2:
        return {'error': 'ROC curve requires binary classification'}

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': roc_auc
    }

def evaluate_model_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: np.ndarray = None) -> Dict[str, Any]:
    """
    Complete model evaluation combining all metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities

    Returns:
        Comprehensive evaluation results
    """
    logger.info("Computing model evaluation metrics...")

    results = {}

    # Basic metrics
    results['metrics'] = compute_classification_metrics(y_true, y_pred, y_proba)

    # Confusion matrix
    results['confusion_matrix'] = generate_confusion_matrix(y_true, y_pred)

    # Classification report
    results['classification_report'] = generate_classification_report(y_true, y_pred)

    # ROC curve (if applicable)
    if y_proba is not None:
        results['roc_curve'] = compute_roc_curve(y_true, y_proba)

    logger.info(f"Evaluation complete. Accuracy: {results['metrics']['accuracy']:.4f}")
    return results

def format_evaluation_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Format evaluation results into a DataFrame for saving.

    Args:
        results: Evaluation results dictionary

    Returns:
        Formatted DataFrame
    """
    # Flatten metrics
    df_data = {}
    for key, value in results['metrics'].items():
        df_data[f'metric_{key}'] = [value]

    # Add confusion matrix elements
    cm = results['confusion_matrix']
    if 'true_negatives' in cm:
        df_data['cm_true_negatives'] = [cm['true_negatives']]
        df_data['cm_false_positives'] = [cm['false_positives']]
        df_data['cm_false_negatives'] = [cm['false_negatives']]
        df_data['cm_true_positives'] = [cm['true_positives']]

    return pd.DataFrame(df_data)

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.metrics import accuracy_score

    # Dummy data
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.6, 0.3, 0.9])

    results = evaluate_model_predictions(y_true, y_pred, y_proba)
    print("Sample evaluation results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"Confusion Matrix: {results['confusion_matrix']['confusion_matrix']}")
