"""
Model registry for storing and retrieving trained models.

This module provides functionality to store trained models and retrieve
the best performing model based on evaluation criteria.
"""

import joblib
from pathlib import Path
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "trained"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

def ensure_models_dir():
    """Ensure the models directory exists."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, model_name: str, filepath: Path = None):
    """
    Save a trained model to disk.

    Args:
        model: Trained model object
        model_name: Name of the model
        filepath: Optional custom filepath
    """
    ensure_models_dir()

    if filepath is None:
        filepath = MODELS_DIR / f"{model_name}.pkl"

    joblib.dump(model, filepath)
    logger.info(f"Saved model '{model_name}' to {filepath}")

def save_best_model(model, model_name: str, filepath: Path = BEST_MODEL_PATH):
    """
    Save the best performing model.

    Args:
        model: Best trained model
        model_name: Name of the best model
        filepath: Path to save the model
    """
    ensure_models_dir()
    joblib.dump(model, filepath)
    logger.info(f"Saved best model '{model_name}' to {filepath}")

def load_model(model_name: str) -> Any:
    """
    Load a specific model from disk.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model object
    """
    filepath = MODELS_DIR / f"{model_name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {filepath}")

    model = joblib.load(filepath)
    logger.info(f"Loaded model '{model_name}' from {filepath}")
    return model

def load_best_model(filepath: Path = BEST_MODEL_PATH) -> Any:
    """
    Load the best performing model.

    Args:
        filepath: Path to the best model file

    Returns:
        Loaded best model object
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Best model not found at {filepath}")

    model = joblib.load(filepath)
    logger.info(f"Loaded best model from {filepath}")
    return model

def save_all_models(models: Dict[str, Any]):
    """
    Save all models in a dictionary.

    Args:
        models: Dictionary of model_name -> model_object
    """
    for model_name, model in models.items():
        save_model(model, model_name)

def get_model_info(model) -> Dict[str, Any]:
    """
    Get basic information about a model.

    Args:
        model: Trained model object

    Returns:
        Dictionary with model information
    """
    info = {
        'type': type(model).__name__,
        'module': type(model).__module__,
    }

    # Add model-specific parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        # Only include important parameters
        important_params = {}
        if 'n_estimators' in params:
            important_params['n_estimators'] = params['n_estimators']
        if 'max_depth' in params:
            important_params['max_depth'] = params['max_depth']
        if 'C' in params:
            important_params['C'] = params['C']
        if 'learning_rate' in params:
            important_params['learning_rate'] = params['learning_rate']

        info['parameters'] = important_params

    return info

def list_saved_models() -> list:
    """
    List all saved model files.

    Returns:
        List of model filenames
    """
    ensure_models_dir()
    return [f.stem for f in MODELS_DIR.glob("*.pkl")]

if __name__ == "__main__":
    # Example usage
    print(f"Models directory: {MODELS_DIR}")
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Saved models: {list_saved_models()}")
