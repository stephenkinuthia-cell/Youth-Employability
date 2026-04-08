"""
Main evaluation pipeline for model performance and fairness assessment.

This script runs comprehensive evaluation including metrics, subgroup analysis,
and fairness assessment, then saves results to CSV.
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_complete_evaluation():
    """Run the complete evaluation pipeline."""
    logger.info("Starting complete model evaluation...")

    # Load the best model
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models.registry import load_best_model
    model = load_best_model()

    # 1. Basic metrics evaluation
    logger.info("Step 1: Computing basic performance metrics...")
    from .metrics import evaluate_model_predictions

    # Load test data (we need to recreate the train/test split)
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from models.train import load_data, binarize_target, split_features_target, train_test_split_data

    df = load_data()
    df = binarize_target(df)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split_data(X, y)

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics_results = evaluate_model_predictions(y_test.values, y_pred, y_proba)

    # 2. Subgroup analysis
    logger.info("Step 2: Performing subgroup analysis...")
    from .subgroup_analysis import analyze_region_performance
    subgroup_results = analyze_region_performance(model)

    # 3. Fairness analysis
    logger.info("Step 3: Analyzing model fairness...")
    from .fairness import analyze_model_fairness
    from .subgroup_analysis import load_evaluation_data, get_subgroup_data

    # Prepare subgroup data for fairness analysis
    features_df, cleaned_df = load_evaluation_data()
    target_col = "Employment_Rate_12_Months (%)"
    features_df[target_col] = (features_df[target_col] > 85).astype(int)

    X_full = features_df.drop(columns=[target_col])
    y_full = features_df[target_col]

    subgroup_data = get_subgroup_data(X_full, y_full, cleaned_df, 'Region')
    fairness_results = analyze_model_fairness(model, subgroup_data)

    # 4. Compile results
    logger.info("Step 4: Compiling evaluation results...")

    # Create comprehensive results DataFrame
    results_data = []

    # Overall metrics
    overall_row = {
        'evaluation_type': 'overall_performance',
        'subgroup': 'all_data',
        'sample_size': len(y_test),
        'accuracy': metrics_results['metrics']['accuracy'],
        'precision': metrics_results['metrics']['precision'],
        'recall': metrics_results['metrics']['recall'],
        'f1_score': metrics_results['metrics']['f1_score'],
        'roc_auc': metrics_results['metrics']['roc_auc'],
        'true_positives': metrics_results['confusion_matrix'].get('true_positives', None),
        'false_positives': metrics_results['confusion_matrix'].get('false_positives', None),
        'true_negatives': metrics_results['confusion_matrix'].get('true_negatives', None),
        'false_negatives': metrics_results['confusion_matrix'].get('false_negatives', None)
    }
    results_data.append(overall_row)

    # Subgroup results
    for _, row in subgroup_results.iterrows():
        subgroup_row = {
            'evaluation_type': 'subgroup_performance',
            'subgroup': row['subgroup'],
            'sample_size': row['sample_size'],
            'accuracy': row['accuracy'],
            'precision': row['precision'],
            'recall': row['recall'],
            'f1_score': row['f1_score'],
            'roc_auc': row['roc_auc'],
            'true_positives': row.get('true_positives', None),
            'false_positives': row.get('false_positives', None),
            'true_negatives': row.get('true_negatives', None),
            'false_negatives': row.get('false_negatives', None)
        }
        results_data.append(subgroup_row)

    # Fairness disparities
    for metric, comparison in fairness_results['comparisons'].items():
        fairness_row = {
            'evaluation_type': 'fairness_disparity',
            'subgroup': f'{metric}_disparity',
            'sample_size': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'roc_auc': None,
            'disparity': comparison['disparity'],
            'disparity_percentage': comparison['disparity_percentage'],
            'bias_flagged': fairness_results['bias_indicators'][metric]['flagged']
        }
        results_data.append(fairness_row)

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    # 5. Save results
    logger.info("Step 5: Saving evaluation results...")
    output_dir = Path(__file__).parent.parent.parent / "reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_evaluation.csv"

    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # 6. Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_path}")
    print(f"Overall Accuracy: {metrics_results['metrics']['accuracy']:.4f}")
    print(f"Overall ROC-AUC: {metrics_results['metrics']['roc_auc']:.4f}")

    flagged_biases = fairness_results['summary']['flagged_biases']
    print(f"Potential Biases Flagged: {flagged_biases}")

    if flagged_biases > 0:
        print("\n⚠️  FAIRNESS CONCERNS DETECTED")
        print("Review the detailed results for bias mitigation strategies.")
    else:
        print("\n✅ No significant fairness concerns detected.")

    print("="*60)

    return results_df, metrics_results, subgroup_results, fairness_results

if __name__ == "__main__":
    results_df, metrics, subgroups, fairness = run_complete_evaluation()