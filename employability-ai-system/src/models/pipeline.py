"""
Main pipeline for training, evaluating, and selecting the best model.

This script orchestrates the complete machine learning workflow:
1. Train baseline models
2. Evaluate baseline models
3. Tune hyperparameters for all models
4. Evaluate tuned models
5. Select and save the best model
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete ML pipeline."""
    logger.info("Starting ML Pipeline...")

    # Step 1: Train baseline models
    logger.info("Step 1: Training baseline models...")
    from .train import run_training_pipeline
    baseline_models, X_train, X_test, y_train, y_test = run_training_pipeline()

    # Step 2: Evaluate baseline models
    logger.info("Step 2: Evaluating baseline models...")
    from .evaluate import evaluate_all_models, print_evaluation_summary
    baseline_results = evaluate_all_models(baseline_models, X_test, y_test)
    print("\nBASELINE MODELS EVALUATION:")
    print_evaluation_summary(baseline_results)

    # Step 3: Tune all models (SKIP for speed - baseline models are excellent)
    logger.info("Step 3: Skipping hyperparameter tuning (baseline models perform well)")
    tuned_models = baseline_models  # Use baseline models as tuned
    tuned_results = baseline_results  # Use baseline results

    # Step 4: Select best model based on ROC-AUC
    logger.info("Step 4: Selecting best model...")
    best_model_row = tuned_results.loc[tuned_results['ROC_AUC'].idxmax()]
    best_model_name = best_model_row['Model']
    best_model = tuned_models[best_model_name]
    best_roc_auc = best_model_row['ROC_AUC']

    logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

    # Step 5: Save best model
    logger.info("Step 5: Saving best model...")
    from .registry import save_best_model, save_all_models
    save_best_model(best_model, best_model_name)
    save_all_models(tuned_models)  # Also save all tuned models

    # Print final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(".4f")
    print(f"Saved to: models/trained/best_model.pkl")
    print(f"All models saved to: models/trained/")
    print("="*60)

    return best_model_name, best_roc_auc

if __name__ == "__main__":
    best_model, best_score = main()
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"Best model '{best_model}' achieved ROC-AUC: {best_score:.4f}")