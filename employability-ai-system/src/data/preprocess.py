"""
Preprocess and clean the dataset.

This module handles data cleaning steps:
- Removing duplicates
- Cleaning column names
- Handling missing values through imputation
- Saving the processed dataset

Note: Feature engineering and encoding are NOT performed here.
"""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.impute import SimpleImputer

from load_data import load_raw_data
from validate_data import validate_dataset, print_validation_report

logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by stripping whitespace.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.strip()
    logger.info(f"Cleaned column names")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_removed = n_before - n_after
    
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate rows")
    else:
        logger.info(f"No duplicate rows found")
    
    return df


def impute_numeric_columns(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in numeric columns using median/mean.
    
    Args:
        df: Input DataFrame.
        strategy: Imputation strategy ('median' or 'mean').
    
    Returns:
        DataFrame with numeric missing values imputed.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_cols:
        logger.info("No numeric columns to impute")
        return df
    
    # Check which columns have missing values
    cols_with_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
    
    if not cols_with_missing:
        logger.info("No missing values in numeric columns")
        return df
    
    imputer = SimpleImputer(strategy=strategy)
    df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
    
    logger.info(f"Imputed missing values in {len(cols_with_missing)} numeric columns "
                f"using {strategy}")
    
    return df


def impute_categorical_columns(df: pd.DataFrame, fill_value: str = 'Unknown') -> pd.DataFrame:
    """
    Impute missing values in categorical columns with a default value.
    
    Args:
        df: Input DataFrame.
        fill_value: Value to use for imputation (default: 'Unknown').
    
    Returns:
        DataFrame with categorical missing values imputed.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        logger.info("No categorical columns to impute")
        return df
    
    # Check which columns have missing values
    cols_with_missing = [col for col in categorical_cols if df[col].isnull().sum() > 0]
    
    if not cols_with_missing:
        logger.info("No missing values in categorical columns")
        return df
    
    for col in cols_with_missing:
        df[col] = df[col].fillna(fill_value)
    
    logger.info(f"Imputed missing values in {len(cols_with_missing)} categorical columns "
                f"with '{fill_value}'")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to clean the dataset.
    
    Steps:
    1. Clean column names
    2. Remove duplicate rows
    3. Impute numeric missing values
    4. Impute categorical missing values
    
    Args:
        df: Raw input DataFrame.
    
    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Starting preprocessing pipeline")
    
    # Step 1: Clean column names
    df = clean_column_names(df)
    
    # Step 2: Remove duplicates
    df = remove_duplicate_rows(df)
    
    # Step 3: Impute numeric columns
    df = impute_numeric_columns(df, strategy='median')
    
    # Step 4: Impute categorical columns
    df = impute_categorical_columns(df, fill_value='Unknown')
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    
    return df


def get_processed_data_path() -> Path:
    """
    Get the path where the processed data should be saved.
    
    Returns:
        Path: Path to data/processed/ directory.
    """
    # Determine the project root
    src_path = Path(__file__).parent.parent.parent
    processed_dir = src_path / 'data' / 'processed'
    return processed_dir


def save_processed_data(df: pd.DataFrame, filename: str = 'cleaned_data.csv') -> Path:
    """
    Save the processed dataset to data/processed/ directory.
    
    Args:
        df: Cleaned DataFrame.
        filename: Output filename (default: 'cleaned_data.csv').
    
    Returns:
        Path: Path to the saved file.
    """
    processed_dir = get_processed_data_path()
    
    # Create directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / filename
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path}")
    logger.info(f"Output file size: {output_path.stat().st_size} bytes")
    
    return output_path


def run_pipeline(validate: bool = True) -> Tuple[pd.DataFrame, Path]:
    """
    Run the complete data pipeline: load → validate → preprocess → save.
    
    Args:
        validate: If True, print validation report before preprocessing.
    
    Returns:
        Tuple of (cleaned DataFrame, path to saved file).
    """
    logger.info("\n" + "=" * 70)
    logger.info("Starting Data Pipeline: LOAD → VALIDATE → PREPROCESS → SAVE")
    logger.info("=" * 70)
    
    # Step 1: Load raw data
    logger.info("\n[1/4] Loading raw data...")
    df = load_raw_data()
    
    # Step 2: Validate data
    if validate:
        logger.info("\n[2/4] Validating raw data...")
        report = validate_dataset(df)
        print_validation_report(report)
    else:
        logger.info("\n[2/4] Skipping validation")
    
    # Step 3: Preprocess data
    logger.info("\n[3/4] Preprocessing data...")
    df_clean = preprocess_data(df)
    
    # Step 4: Save processed data
    logger.info("\n[4/4] Saving processed data...")
    output_path = save_processed_data(df_clean)
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 70 + "\n")
    
    return df_clean, output_path


if __name__ == "__main__":
    # When run directly, execute the full pipeline
    df_processed, save_path = run_pipeline(validate=True)
    
    # Display summary of the processed data
    print("\nProcessed Data Summary:")
    print(f"Shape: {df_processed.shape}")
    print(f"\nData Types:\n{df_processed.dtypes}")
    print(f"\nFirst 5 rows:")
    print(df_processed.head())
    print(f"\nMissing values:\n{df_processed.isnull().sum().sum()} total")
