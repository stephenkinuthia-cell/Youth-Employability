"""
Hyperparameter tuning module using GridSearchCV.

This module performs hyperparameter tuning for all baseline models using
GridSearchCV with 5-fold cross-validation, optimizing for ROC-AUC score.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameter grids (simplified for faster execution)
PARAM_GRIDS = {
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['liblinear']
    },
    'RandomForest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    }
}

def tune_model(model, param_grid: dict, X_train, y_train, cv: int = 5, scoring='roc_auc'):
    """
    Tune a single model using GridSearchCV.

    Args:
        model: Base model to tune
        param_grid: Hyperparameter grid
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Best estimator from grid search
    """
    logger.info(f"Tuning {model.__class__.__name__} with {len(param_grid)} parameters...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(".4f")

    return grid_search.best_estimator_

def tune_all_models(models: dict, X_train, y_train, param_grids: dict = PARAM_GRIDS):
    """
    Tune all models using their respective parameter grids.

    Args:
        models: Dictionary of base models
        X_train: Training features
        y_train: Training target
        param_grids: Dictionary of parameter grids

    Returns:
        Dictionary of tuned models
    """
    tuned_models = {}

    for model_name, model in models.items():
        if model_name in param_grids:
            tuned_model = tune_model(model, param_grids[model_name], X_train, y_train)
            tuned_models[model_name] = tuned_model
        else:
            logger.warning(f"No parameter grid found for {model_name}, using base model")
            tuned_models[model_name] = model

    logger.info(f"Tuned {len(tuned_models)} models")
    return tuned_models

def get_base_models(random_state: int = 42):
    """
    Get base models for tuning.

    Args:
        random_state: Random state for reproducibility

    Returns:
        Dictionary of base models
    """
    return {
        'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state)
    }

if __name__ == "__main__":
    # Example usage
    from train import run_training_pipeline

    # Get training data
    models, X_train, X_test, y_train, y_test = run_training_pipeline()

    # Tune models
    tuned_models = tune_all_models(models, X_train, y_train)

    print(f"Tuned models: {list(tuned_models.keys())}")

    # Evaluate tuned models
    from evaluate import evaluate_all_models, print_evaluation_summary

    tuned_results = evaluate_all_models(tuned_models, X_test, y_test)
    print("\nTUNED MODELS EVALUATION:")
    print_evaluation_summary(tuned_results)
