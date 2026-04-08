"""
Prediction module for making predictions with trained models.

This module loads the best trained model and provides functions for
making predictions and obtaining prediction probabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_best_model():
    """
    Load the best trained model.

    Returns:
        Loaded model object
    """
    from .registry import load_best_model as _load_best_model

    try:
        model = _load_best_model()
        logger.info("Best model loaded successfully")
        return model
    except FileNotFoundError as e:
        logger.error(f"Best model not found: {e}")
        raise

def predict(model, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Make predictions using the trained model.

    Args:
        model: Trained model
        X: Feature matrix

    Returns:
        Predictions array
    """
    logger.info(f"Making predictions on {X.shape[0]} samples")
    predictions = model.predict(X)
    return predictions

def predict_proba(model, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Get prediction probabilities using the trained model.

    Args:
        model: Trained model
        X: Feature matrix

    Returns:
        Prediction probabilities array (shape: n_samples, n_classes)
    """
    if not hasattr(model, 'predict_proba'):
        raise AttributeError(f"Model {type(model).__name__} does not support predict_proba")

    logger.info(f"Getting prediction probabilities for {X.shape[0]} samples")
    probabilities = model.predict_proba(X)
    return probabilities

def predict_with_proba(model, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions and get probabilities in one call.

    Args:
        model: Trained model
        X: Feature matrix

    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = predict(model, X)
    probabilities = predict_proba(model, X)
    return predictions, probabilities

def format_predictions(predictions: np.ndarray, probabilities: np.ndarray = None) -> pd.DataFrame:
    """
    Format predictions into a readable DataFrame.

    Args:
        predictions: Prediction array
        probabilities: Probability array (optional)

    Returns:
        DataFrame with predictions and probabilities
    """
    df = pd.DataFrame({
        'Prediction': predictions,
        'Prediction_Label': ['High_Employability' if p == 1 else 'Low_Employability' for p in predictions]
    })

    if probabilities is not None:
        # For binary classification, probabilities[:, 1] is probability of positive class
        if probabilities.shape[1] == 2:
            df['Probability_High_Employability'] = probabilities[:, 1]
            df['Probability_Low_Employability'] = probabilities[:, 0]
        else:
            # Multi-class case
            for i in range(probabilities.shape[1]):
                df[f'Probability_Class_{i}'] = probabilities[:, i]

    return df

def make_predictions(X: Union[pd.DataFrame, np.ndarray], include_proba: bool = True) -> pd.DataFrame:
    """
    Complete prediction pipeline: load model, make predictions, format results.

    Args:
        X: Feature matrix
        include_proba: Whether to include prediction probabilities

    Returns:
        DataFrame with predictions and optional probabilities
    """
    model = load_best_model()

    if include_proba:
        predictions, probabilities = predict_with_proba(model, X)
        return format_predictions(predictions, probabilities)
    else:
        predictions = predict(model, X)
        return format_predictions(predictions)

def get_prediction_summary(predictions_df: pd.DataFrame) -> dict:
    """
    Get summary statistics of predictions.

    Args:
        predictions_df: DataFrame from format_predictions

    Returns:
        Dictionary with prediction summary
    """
    summary = {
        'total_predictions': len(predictions_df),
        'high_employability_count': (predictions_df['Prediction'] == 1).sum(),
        'low_employability_count': (predictions_df['Prediction'] == 0).sum(),
        'high_employability_percentage': (predictions_df['Prediction'] == 1).mean() * 100,
    }

    if 'Probability_High_Employability' in predictions_df.columns:
        summary['avg_probability_high'] = predictions_df['Probability_High_Employability'].mean()
        summary['avg_probability_low'] = predictions_df['Probability_Low_Employability'].mean()

    return summary

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create dummy data for testing (should match feature dimensions)
    # In real usage, this would be actual feature data
    n_samples = 10
    n_features = 102  # Based on our feature engineering
    X_dummy = np.random.randn(n_samples, n_features)

    try:
        predictions_df = make_predictions(X_dummy, include_proba=True)
        print("Predictions:")
        print(predictions_df.head())

        summary = get_prediction_summary(predictions_df)
        print("\nPrediction Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    except FileNotFoundError:
        print("Best model not found. Please train and save a model first.")
        print("Run the training pipeline to create models/trained/best_model.pkl")
