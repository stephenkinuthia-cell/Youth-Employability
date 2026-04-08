"""
Subgroup analysis module for evaluating model performance across different groups.

This module analyzes how the model performs across different subgroups such as
regions, to identify potential disparities in performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_evaluation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the final features and original cleaned data for subgroup analysis.

    Returns:
        Tuple of (features_df, cleaned_df)
    """
    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
    cleaned_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cleaned_data.csv"

    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)

    logger.info(f"Loading cleaned data from {cleaned_path}")
    cleaned_df = pd.read_csv(cleaned_path)

    return features_df, cleaned_df

def get_subgroup_data(X: pd.DataFrame, y: pd.DataFrame, subgroups: pd.DataFrame,
                     subgroup_column: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split data by subgroup for analysis.

    Args:
        X: Feature matrix
        y: Target vector
        subgroups: DataFrame with subgroup information
        subgroup_column: Column name for subgrouping

    Returns:
        Dictionary with subgroup data
    """
    subgroup_data = {}

    unique_groups = subgroups[subgroup_column].unique()

    for group in unique_groups:
        mask = subgroups[subgroup_column] == group
        subgroup_data[str(group)] = {
            'X': X[mask],
            'y': y[mask],
            'size': len(mask[mask])
        }
        logger.info(f"Group '{group}': {len(mask[mask])} samples")

    return subgroup_data

def evaluate_subgroup_performance(model, subgroup_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """
    Evaluate model performance for each subgroup.

    Args:
        model: Trained model
        subgroup_data: Dictionary with subgroup data

    Returns:
        Dictionary with performance metrics per subgroup
    """
    from .metrics import evaluate_model_predictions

    results = {}

    for group_name, data in subgroup_data.items():
        if data['size'] == 0:
            logger.warning(f"Skipping empty group: {group_name}")
            continue

        logger.info(f"Evaluating group: {group_name} ({data['size']} samples)")

        X_group = data['X']
        y_group = data['y']

        # Get predictions
        y_pred = model.predict(X_group)
        y_proba = model.predict_proba(X_group)[:, 1] if hasattr(model, 'predict_proba') else None

        # Evaluate
        eval_results = evaluate_model_predictions(y_group.values, y_pred, y_proba)

        results[group_name] = {
            'sample_size': data['size'],
            'metrics': eval_results['metrics'],
            'confusion_matrix': eval_results['confusion_matrix']
        }

    return results

def compare_subgroup_performance(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare performance across subgroups.

    Args:
        results: Results from evaluate_subgroup_performance

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for group_name, group_results in results.items():
        row = {
            'subgroup': group_name,
            'sample_size': group_results['sample_size'],
            'accuracy': group_results['metrics']['accuracy'],
            'precision': group_results['metrics']['precision'],
            'recall': group_results['metrics']['recall'],
            'f1_score': group_results['metrics']['f1_score'],
            'roc_auc': group_results['metrics']['roc_auc']
        }

        # Add confusion matrix elements if available
        cm = group_results['confusion_matrix']
        if 'true_positives' in cm:
            row['true_positives'] = cm['true_positives']
            row['false_positives'] = cm['false_positives']
            row['true_negatives'] = cm['true_negatives']
            row['false_negatives'] = cm['false_negatives']

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df

def analyze_region_performance(model) -> pd.DataFrame:
    """
    Analyze model performance by region.

    Note: Gender analysis not possible as gender data is not available in the dataset.

    Args:
        model: Trained model

    Returns:
        DataFrame with region-wise performance
    """
    logger.info("Starting subgroup analysis by region...")

    # Load data
    features_df, cleaned_df = load_evaluation_data()

    # Prepare features and target
    target_col = "Employment_Rate_12_Months (%)"
    features_df[target_col] = (features_df[target_col] > 85).astype(int)  # Binarize as in training

    X = features_df.drop(columns=[target_col])
    y = features_df[target_col]

    # Get subgroup data (Region)
    subgroup_data = get_subgroup_data(X, y, cleaned_df, 'Region')

    # Evaluate performance
    results = evaluate_subgroup_performance(model, subgroup_data)

    # Compare results
    comparison_df = compare_subgroup_performance(results)

    logger.info("Subgroup analysis complete")
    return comparison_df

def print_subgroup_summary(comparison_df: pd.DataFrame):
    """
    Print a summary of subgroup performance.

    Args:
        comparison_df: DataFrame from compare_subgroup_performance
    """
    print("\n" + "="*60)
    print("SUBGROUP PERFORMANCE ANALYSIS")
    print("="*60)

    # Sort by accuracy
    df_sorted = comparison_df.sort_values('accuracy', ascending=False)

    for _, row in df_sorted.iterrows():
        print(f"\n{row['subgroup']}:")
        print(f"  Sample Size: {row['sample_size']}")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

        if 'true_positives' in row:
            print(f"  Confusion Matrix: TP={row['true_positives']}, FP={row['false_positives']}, "
                  f"TN={row['true_negatives']}, FN={row['false_negatives']}")

    # Performance disparities
    print(f"\nPerformance Range:")
    print(".4f")
    print(".4f")
    print(".4f")

if __name__ == "__main__":
    # Example usage
    from src.models.registry import load_best_model

    model = load_best_model()
    results_df = analyze_region_performance(model)
    print_subgroup_summary(results_df)
