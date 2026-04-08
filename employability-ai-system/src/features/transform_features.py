"""
Transform features through encoding and scaling.

This module handles:
- One-hot encoding of categorical variables
- Standard scaling of numerical variables
- Consistent, reproducible transformations via sklearn pipelines

Note: NO feature selection here (that's in select_features.py).
"""

import logging
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def identify_feature_types(df: pd.DataFrame, 
                          target_col: str = 'Employment_Rate_12_Months (%)',
                          exclude_cols: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical features automatically.
    
    Args:
        df: Input DataFrame.
        target_col: Target column to exclude.
        exclude_cols: Additional columns to exclude (e.g., identifiers).
    
    Returns:
        Tuple of (categorical_columns, numerical_columns).
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Columns to always exclude
    exclude_list = [target_col] + exclude_cols
    
    # Get all columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_list]
    
    # Separate by dtype
    categorical = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = df[feature_cols].select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
    
    logger.info(f"Identified {len(categorical)} categorical and {len(numerical)} numerical features")
    
    return categorical, numerical


def build_preprocessor(categorical_cols: List[str], 
                       numerical_cols: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing pipeline using sklearn ColumnTransformer.
    
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
    
    Returns:
        ColumnTransformer with categorical and numerical pipelines.
    """
    # Categorical transformer: one-hot encode, drop first to avoid multicollinearity
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Numerical transformer: standardize
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    logger.info(f"Built preprocessor for {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")
    
    return preprocessor


def fit_and_transform(df: pd.DataFrame, 
                      target_col: str = 'Employment_Rate_12_Months (%)',
                      exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Fit preprocessor on data and transform features.
    
    Args:
        df: Input DataFrame with raw features.
        target_col: Target column to exclude.
        exclude_cols: Additional columns to exclude.
    
    Returns:
        Tuple of (transformed_dataframe, fitted_preprocessor).
    """
    if exclude_cols is None:
        exclude_cols = []
    
    logger.info("Starting feature transformation pipeline")
    
    # Identify feature types
    categorical_cols, numerical_cols = identify_feature_types(df, target_col, exclude_cols)
    
    # Build preprocessor
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
    # Fit and transform
    features_to_transform = [col for col in df.columns 
                            if col != target_col and col not in exclude_cols]
    X = df[features_to_transform]
    
    X_transformed = preprocessor.fit_transform(X)
    
    logger.info(f"Transformed features shape: {X_transformed.shape}")
    
    # Get feature names after transformation
    feature_names = get_feature_names_after_transform(
        preprocessor, categorical_cols, numerical_cols
    )
    
    # Convert to DataFrame for interpretability
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    
    logger.info(f"Final transformed shape: {df_transformed.shape}")
    
    return df_transformed, preprocessor


def get_feature_names_after_transform(preprocessor: ColumnTransformer,
                                      categorical_cols: List[str],
                                      numerical_cols: List[str]) -> List[str]:
    """
    Get feature names after one-hot encoding and scaling.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        categorical_cols: List of original categorical columns.
        numerical_cols: List of original numerical columns.
    
    Returns:
        List of feature names after transformation.
    """
    feature_names = []
    
    # Numerical features keep their names
    feature_names.extend(numerical_cols)
    
    # Categorical features get one-hot encoded names
    for idx, cat_col in enumerate(categorical_cols):
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categories = ohe.categories_[idx]
        # Drop first category to avoid multicollinearity
        transformed_names = [f"{cat_col}_{cat}" for cat in categories[1:]]
        feature_names.extend(transformed_names)
    
    return feature_names


def transform_new_data(df: pd.DataFrame, 
                       preprocessor: ColumnTransformer,
                       target_col: str = 'Employment_Rate_12_Months (%)',
                       exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Transform new data using a fitted preprocessor.
    
    This is useful for applying the same transformations to test/validation data.
    
    Args:
        df: New DataFrame to transform.
        preprocessor: Fitted ColumnTransformer.
        target_col: Target column to exclude.
        exclude_cols: Additional columns to exclude.
    
    Returns:
        Transformed DataFrame.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    features_to_transform = [col for col in df.columns 
                            if col != target_col and col not in exclude_cols]
    X = df[features_to_transform]
    
    X_transformed = preprocessor.transform(X)
    
    # Get feature names to maintain consistency
    categorical_cols, numerical_cols = identify_feature_types(df, target_col, exclude_cols)
    feature_names = get_feature_names_after_transform(
        preprocessor, categorical_cols, numerical_cols
    )
    
    df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    
    return df_transformed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    from build_features import load_cleaned_data, build_all_features
    
    df = load_cleaned_data()
    df_featured = build_all_features(df)
    
    # Exclude University_Name as it's too high-cardinality for one-hot encoding
    exclude = ['University_Name']
    
    df_transformed, preprocessor = fit_and_transform(
        df_featured, 
        exclude_cols=exclude
    )
    
    print("\nTransformed features sample:")
    print(df_transformed.head())
    print(f"\nFinal shape: {df_transformed.shape}")
    print(f"Feature names: {df_transformed.columns.tolist()[:10]}... ({len(df_transformed.columns)} total)")
