"""
Select relevant features and remove redundancy.

This module performs feature selection through:
- Correlation analysis (remove highly correlated features)
- Feature importance (identify weak predictors)
- Low variance filtering (remove near-constant features)

Note: Feature importance is calculated using a simple model for scoring only,
NOT for model training or hyperparameter tuning.
"""

import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


def remove_low_variance_features(df: pd.DataFrame,
                                 threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with variance below threshold.
    
    Features with very low variance contribute little to model predictions.
    
    Args:
        df: Input DataFrame with numerical features.
        threshold: Variance threshold (default: 0.01).
    
    Returns:
        Tuple of (DataFrame with low-variance features removed, removed_columns).
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    
    # Get mask of features to keep
    mask = selector.get_support()
    removed = [col for col, keep in zip(df.columns, mask) if not keep]
    
    if removed:
        logger.info(f"Removed {len(removed)} low-variance features: {removed}")
    else:
        logger.info("No low-variance features removed")
    
    df_reduced = df.loc[:, mask]
    
    return df_reduced, removed


def remove_highly_correlated_features(df: pd.DataFrame,
                                      threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove one feature from each highly correlated pair.
    
    Highly correlated features provide redundant information.
    
    Args:
        df: Input DataFrame (numerical columns only).
        threshold: Correlation threshold (default: 0.95).
    
    Returns:
        Tuple of (DataFrame with one from each pair removed, removed_columns).
    """
    # Compute correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle to avoid duplicates
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find columns with correlation > threshold
    to_drop = [col for col in upper_triangle.columns 
               if any(upper_triangle[col] > threshold)]
    
    if to_drop:
        logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    else:
        logger.info("No highly correlated features removed")
    
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced, to_drop


def calculate_feature_importance(X: pd.DataFrame,
                                y: pd.Series,
                                model_type: str = 'rf',
                                n_estimators: int = 100,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Calculate feature importance using a lightweight model.
    
    Uses Random Forest for importance scoring. Importance indicates
    how much each feature contributes to reducing prediction error.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        model_type: Type of model ('rf' for Random Forest).
        n_estimators: Number of trees in forest.
        random_state: Random state for reproducibility.
    
    Returns:
        DataFrame with feature names and importance scores.
    """
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=10  # Keep shallow to avoid overfitting
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Fit model for importance scoring only
    model.fit(X, y)
    
    # Extract importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    logger.info(f"Calculated feature importance from {model_type} model")
    
    return importance


def select_features_by_importance(importance_df: pd.DataFrame,
                                  threshold: float = 0.01) -> List[str]:
    """
    Select features with importance above threshold.
    
    Args:
        importance_df: DataFrame from calculate_feature_importance().
        threshold: Importance threshold (default: 0.01).
    
    Returns:
        List of selected feature names.
    """
    selected = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
    
    logger.info(f"Selected {len(selected)} features with importance >= {threshold}")
    
    return selected


def perform_feature_selection(X: pd.DataFrame,
                             y: pd.Series,
                             exclude_cols: List[str] = None,
                             variance_threshold: float = 0.01,
                             correlation_threshold: float = 0.95,
                             importance_threshold: float = 0.005) -> Tuple[pd.DataFrame, dict]:
    """
    Orchestrate full feature selection pipeline.
    
    Steps:
    1. Remove low-variance features
    2. Remove highly correlated features
    3. Calculate feature importance
    4. Remove features below importance threshold
    
    Args:
        X: Feature matrix.
        y: Target vector.
        exclude_cols: Columns to always keep.
        variance_threshold: Low-variance threshold.
        correlation_threshold: Correlation threshold.
        importance_threshold: Importance threshold.
    
    Returns:
        Tuple of (selected_feature_matrix, selection_report).
    """
    if exclude_cols is None:
        exclude_cols = []
    
    logger.info("Starting feature selection pipeline")
    
    # Step 1: Low variance filtering
    X_var, removed_var = remove_low_variance_features(X, threshold=variance_threshold)
    
    # Step 2: Correlation filtering
    X_corr, removed_corr = remove_highly_correlated_features(X_var, threshold=correlation_threshold)
    
    # Step 3: Feature importance
    importance = calculate_feature_importance(X_corr, y)
    selected_features = select_features_by_importance(importance, threshold=importance_threshold)
    
    # Ensure excluded columns stay (if they exist)
    for col in exclude_cols:
        if col in X.columns and col not in selected_features:
            selected_features.append(col)
    
    # Final selection
    X_selected = X_corr[selected_features]
    
    # Report
    report = {
        'original_features': X.shape[1],
        'removed_variance': removed_var,
        'removed_correlation': removed_corr,
        'importance_threshold': importance_threshold,
        'final_features': len(selected_features),
        'selected_features': selected_features,
        'importance_scores': importance.to_dict('records')
    }
    
    logger.info(f"Feature selection complete: {X.shape[1]} → {X_selected.shape[1]} features")
    
    return X_selected, report


def print_selection_report(report: dict) -> None:
    """
    Pretty-print feature selection report.
    
    Args:
        report: Dictionary from perform_feature_selection().
    """
    print("\n" + "=" * 70)
    print("FEATURE SELECTION REPORT")
    print("=" * 70)
    
    print(f"\nOriginal Features: {report['original_features']}")
    print(f"Final Features: {report['final_features']}")
    print(f"Reduction: {report['original_features'] - report['final_features']} features removed "
          f"({100 * (1 - report['final_features'] / report['original_features']):.1f}%)")
    
    print(f"\nRemoved Features (Low Variance):")
    if report['removed_variance']:
        for col in report['removed_variance'][:5]:
            print(f"  - {col}")
        if len(report['removed_variance']) > 5:
            print(f"  ... and {len(report['removed_variance']) - 5} more")
    else:
        print("  None")
    
    print(f"\nRemoved Features (High Correlation):")
    if report['removed_correlation']:
        for col in report['removed_correlation'][:5]:
            print(f"  - {col}")
        if len(report['removed_correlation']) > 5:
            print(f"  ... and {len(report['removed_correlation']) - 5} more")
    else:
        print("  None")
    
    print(f"\nTop 10 Features by Importance:")
    for idx, feat in enumerate(report['importance_scores'][:10], 1):
        print(f"  {idx}. {feat['Feature']}: {feat['Importance']:.4f}")
    
    print(f"\nSelected Features ({len(report['selected_features'])} total):")
    for col in sorted(report['selected_features'])[:15]:
        print(f"  - {col}")
    if len(report['selected_features']) > 15:
        print(f"  ... and {len(report['selected_features']) - 15} more")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    from build_features import load_cleaned_data, build_all_features
    from transform_features import fit_and_transform
    
    df = load_cleaned_data()
    df_featured = build_all_features(df)
    
    # Extract target (assuming it exists)
    target_col = 'Employment_Rate_12_Months (%)'
    if target_col not in df_featured.columns:
        logger.warning(f"Target column {target_col} not found. Using None for feature selection.")
        y = None
    else:
        y = df_featured[target_col]
    
    # Transform features
    exclude = ['University_Name']
    df_transformed, _ = fit_and_transform(df_featured, exclude_cols=exclude)
    
    # Perform selection if target available
    if y is not None:
        X_selected, report = perform_feature_selection(
            df_transformed, y, exclude_cols=exclude
        )
        
        print_selection_report(report)
        
        print(f"\nSelected feature matrix shape: {X_selected.shape}")
        print(f"Sample of selected features:\n{X_selected.head()}")
