"""
Fairness analysis module for detecting bias in model predictions.

This module analyzes model fairness by comparing error rates, false positive/negative
rates across different subgroups to identify potential biases.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate fairness-related metrics for a subgroup.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with fairness metrics
    """
    if len(np.unique(y_true)) != 2:
        return {'error': 'Fairness metrics require binary classification'}

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total = len(y_true)
    positives = np.sum(y_true)
    negatives = total - positives

    metrics = {
        'total_samples': total,
        'positive_class_samples': positives,
        'negative_class_samples': negatives,
        'accuracy': (tp + tn) / total,
        'error_rate': (fp + fn) / total,
        'true_positive_rate': tp / positives if positives > 0 else 0,  # Sensitivity/Recall
        'false_positive_rate': fp / negatives if negatives > 0 else 0,
        'true_negative_rate': tn / negatives if negatives > 0 else 0,  # Specificity
        'false_negative_rate': fn / positives if positives > 0 else 0,
        'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'false_discovery_rate': fp / (tp + fp) if (tp + fp) > 0 else 0,
        'false_omission_rate': fn / (tn + fn) if (tn + fn) > 0 else 0
    }

    return metrics

def compare_fairness_across_groups(fairness_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Compare fairness metrics across different groups.

    Args:
        fairness_results: Dictionary with fairness metrics per group

    Returns:
        Dictionary with fairness comparisons and disparities
    """
    if len(fairness_results) < 2:
        return {'error': 'Need at least 2 groups for fairness comparison'}

    # Extract metrics for each group
    groups = list(fairness_results.keys())
    metrics_to_compare = [
        'error_rate', 'false_positive_rate', 'false_negative_rate',
        'true_positive_rate', 'true_negative_rate'
    ]

    comparisons = {}

    for metric in metrics_to_compare:
        values = [fairness_results[group][metric] for group in groups]
        max_val = max(values)
        min_val = min(values)
        disparity = max_val - min_val

        comparisons[metric] = {
            'values': dict(zip(groups, values)),
            'max_value': max_val,
            'min_value': min_val,
            'disparity': disparity,
            'disparity_percentage': (disparity / min_val * 100) if min_val > 0 else 0
        }

    return comparisons

def detect_bias_indicators(comparisons: Dict[str, Any], threshold: float = 0.1) -> Dict[str, Any]:
    """
    Detect potential bias indicators based on disparities.

    Args:
        comparisons: Results from compare_fairness_across_groups
        threshold: Threshold for flagging disparities (10% by default)

    Returns:
        Dictionary with bias detection results
    """
    bias_indicators = {}

    for metric, data in comparisons.items():
        disparity_pct = data['disparity_percentage']

        if disparity_pct > threshold * 100:  # Convert to percentage
            bias_indicators[metric] = {
                'flagged': True,
                'disparity_percentage': disparity_pct,
                'severity': 'high' if disparity_pct > 20 else 'moderate',
                'description': f"Significant disparity in {metric.replace('_', ' ')} across groups"
            }
        else:
            bias_indicators[metric] = {
                'flagged': False,
                'disparity_percentage': disparity_pct,
                'severity': 'low',
                'description': f"Acceptable disparity in {metric.replace('_', ' ')}"
            }

    return bias_indicators

def analyze_model_fairness(model, subgroup_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """
    Comprehensive fairness analysis across subgroups.

    Args:
        model: Trained model
        subgroup_data: Dictionary with subgroup data

    Returns:
        Comprehensive fairness analysis results
    """
    logger.info("Starting fairness analysis...")

    fairness_results = {}

    # Calculate fairness metrics for each subgroup
    for group_name, data in subgroup_data.items():
        if data['size'] == 0:
            continue

        X_group = data['X']
        y_group = data['y']

        # Get predictions
        y_pred = model.predict(X_group)

        # Calculate fairness metrics
        fairness_results[group_name] = calculate_fairness_metrics(y_group.values, y_pred)

    # Compare across groups
    comparisons = compare_fairness_across_groups(fairness_results)

    # Detect bias indicators
    bias_indicators = detect_bias_indicators(comparisons)

    results = {
        'fairness_metrics': fairness_results,
        'comparisons': comparisons,
        'bias_indicators': bias_indicators,
        'summary': {
            'total_groups_analyzed': len(fairness_results),
            'flagged_biases': sum(1 for b in bias_indicators.values() if b['flagged']),
            'most_concerning_metric': max(bias_indicators.items(), key=lambda x: x[1]['disparity_percentage'])[0]
        }
    }

    logger.info("Fairness analysis complete")
    return results

def print_fairness_report(results: Dict[str, Any]):
    """
    Print a comprehensive fairness report.

    Args:
        results: Results from analyze_model_fairness
    """
    print("\n" + "="*60)
    print("MODEL FAIRNESS ANALYSIS REPORT")
    print("="*60)

    # Summary
    summary = results['summary']
    print(f"\nGroups Analyzed: {summary['total_groups_analyzed']}")
    print(f"Potential Biases Flagged: {summary['flagged_biases']}")
    print(f"Most Concerning Metric: {summary['most_concerning_metric'].replace('_', ' ')}")

    # Bias indicators
    print(f"\nBIAS DETECTION RESULTS:")
    for metric, indicator in results['bias_indicators'].items():
        status = "⚠️  FLAGGED" if indicator['flagged'] else "✅ OK"
        print(f"  {metric.replace('_', ' ').title()}: {status}")
        print(".1f")
        if indicator['flagged']:
            print(f"    Severity: {indicator['severity'].upper()}")
            print(f"    {indicator['description']}")

    # Detailed comparisons
    print(f"\nDETAILED METRIC COMPARISONS:")
    for metric, comparison in results['comparisons'].items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        for group, value in comparison['values'].items():
            print(".4f")
        print(".4f")
        print(".1f")

def format_fairness_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Format fairness results into a DataFrame for saving.

    Args:
        results: Fairness analysis results

    Returns:
        Formatted DataFrame
    """
    rows = []

    for group, metrics in results['fairness_metrics'].items():
        row = {'subgroup': group}
        row.update(metrics)
        rows.append(row)

    # Add comparison data
    for metric, comparison in results['comparisons'].items():
        row = {
            'subgroup': f'{metric}_comparison',
            'max_disparity': comparison['disparity'],
            'disparity_percentage': comparison['disparity_percentage']
        }
        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Example usage
    from src.models.registry import load_best_model
    from src.evaluation.subgroup_analysis import load_evaluation_data, get_subgroup_data

    model = load_best_model()

    # Load data and get subgroup data
    features_df, cleaned_df = load_evaluation_data()
    target_col = "Employment_Rate_12_Months (%)"
    features_df[target_col] = (features_df[target_col] > 85).astype(int)

    X = features_df.drop(columns=[target_col])
    y = features_df[target_col]

    subgroup_data = get_subgroup_data(X, y, cleaned_df, 'Region')

    # Analyze fairness
    fairness_results = analyze_model_fairness(model, subgroup_data)
    print_fairness_report(fairness_results)
