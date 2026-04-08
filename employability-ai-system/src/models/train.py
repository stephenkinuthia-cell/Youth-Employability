"""
Training module for baseline machine learning models.

This module loads the processed features dataset, binarizes the target variable
for classification, performs train-test split, and trains baseline models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "final_features.csv"
TARGET_COLUMN = "Employment_Rate_12_Months (%)"
THRESHOLD = 85.0  # Binarize: 1 if > 85%, 0 otherwise

def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the final features dataset.

    Args:
        data_path: Path to the CSV file

    Returns:
        DataFrame with features and target
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df

def binarize_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN, threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Binarize the continuous target variable for classification.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        threshold: Threshold for binarization (1 if > threshold, 0 otherwise)

    Returns:
        DataFrame with binarized target
    """
    df = df.copy()
    df[target_col] = (df[target_col] > threshold).astype(int)
    logger.info(f"Binarized target: {df[target_col].value_counts().to_dict()}")
    return df

def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """
    Split DataFrame into features (X) and target (y).

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        X: Feature matrix
        y: Target vector
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Perform stratified train-test split.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random state for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_baseline_models(X_train, y_train, random_state: int = RANDOM_STATE):
    """
    Train baseline models: Logistic Regression, Random Forest, Gradient Boosting.

    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random state for reproducibility

    Returns:
        Dictionary of trained models
    """
    models = {}

    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr

    # Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=random_state, n_estimators=100)
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb

    logger.info(f"Trained {len(models)} baseline models")
    return models

def run_training_pipeline():
    """
    Run the complete training pipeline.

    Returns:
        models: Dictionary of trained models
        X_train, X_test, y_train, y_test: Split data
    """
    # Load and preprocess data
    df = load_data()
    df = binarize_target(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train models
    models = train_baseline_models(X_train, y_train)

    return models, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    models, X_train, X_test, y_train, y_test = run_training_pipeline()
    print(f"Trained models: {list(models.keys())}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
