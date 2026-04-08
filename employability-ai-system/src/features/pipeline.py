"""
Feature engineering integration pipeline.

This module runs the complete feature engineering workflow:
1. Build features
2. Transform features  
3. Select features
4. Save final dataset
"""

import logging
from pathlib import Path
import pandas as pd

from build_features import load_cleaned_data, build_all_features
from transform_features import fit_and_transform
from select_features import perform_feature_selection, print_selection_report

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_features_output_path() -> Path:
    """Get path to final features output."""
    src_path = Path(__file__).parent.parent.parent
    return src_path / 'data' / 'processed' / 'final_features.csv'


def save_final_features(X: pd.DataFrame, 
                       y: pd.Series = None,
                       output_path: Path = None) -> Path:
    """
    Save final feature matrix and optionally target to CSV.
    
    Args:
        X: Feature matrix.
        y: Target vector (optional).
        output_path: Path to save file.
    
    Returns:
        Path: Path to saved file.
    """
    if output_path is None:
        output_path = get_features_output_path()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target if provided
    if y is not None:
        df_output = X.copy()
        df_output['Employment_Rate_12_Months (%)'] = y.values
    else:
        df_output = X.copy()
    
    df_output.to_csv(output_path, index=False)
    
    logger.info(f"Saved final features to {output_path}")
    logger.info(f"Output shape: {df_output.shape}")
    
    return output_path


def run_pipeline(validate: bool = True, exclude_cols: str = 'University_Name') -> tuple:
    """
    Run complete feature engineering pipeline.
    
    Args:
        validate: Print detailed reports.
        exclude_cols: High-cardinality columns to exclude.
    
    Returns:
        Tuple of (feature_matrix, target, output_path, report).
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)
    
    # Load and build features
    logger.info("\n[1/4] Loading and building features...")
    df = load_cleaned_data()
    df_featured = build_all_features(df)
    
    # Extract target
    target_col = 'Employment_Rate_12_Months (%)'
    y = df_featured[target_col].copy()
    
    # Transform features
    logger.info("\n[2/4] Transforming features...")
    exclude = [exclude_cols] if exclude_cols else []
    df_transformed, _ = fit_and_transform(df_featured, exclude_cols=exclude)
    
    # Select features
    logger.info("\n[3/4] Selecting relevant features...")
    X_selected, report = perform_feature_selection(
        df_transformed, y, 
        exclude_cols=exclude,
        importance_threshold=0.0001  # Lower threshold to keep more features
    )
    
    if validate:
        print_selection_report(report)
    
    # Save
    logger.info("\n[4/4] Saving final dataset...")
    output_path = save_final_features(X_selected, y)
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info(f"Features: {X_selected.shape[1]}, Samples: {X_selected.shape[0]}")
    logger.info("=" * 70 + "\n")
    
    return X_selected, y, output_path, report


if __name__ == "__main__":
    X, y, path, report = run_pipeline(validate=True)
    print(f"\nFinal dataset saved to: {path}")
    print(f"Shape: {X.shape}")
