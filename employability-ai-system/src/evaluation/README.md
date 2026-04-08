# Model Evaluation and Fairness Documentation

## Overview

This evaluation module provides comprehensive assessment of model performance and fairness across different subgroups. The system evaluates the trained GradientBoosting model on multiple dimensions to ensure reliability and identify potential biases.

**Output**: `reports/results/model_evaluation.csv`

---

## Module Architecture

```
src/evaluation/
├── __init__.py           # Package initialization
├── metrics.py            # Performance metrics computation
├── subgroup_analysis.py  # Performance by subgroups
├── fairness.py           # Bias detection and fairness analysis
└── pipeline.py           # Complete evaluation orchestration
```

---

## Module 1: metrics.py

**Purpose**: Compute standard classification metrics and confusion matrices.

### Metrics Computed

1. **Basic Metrics**
   - Accuracy: Overall correct predictions
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - F1-Score: Harmonic mean of precision and recall

2. **Advanced Metrics**
   - ROC-AUC: Area under the receiver operating characteristic curve
   - Macro-averaged metrics: Unweighted average across classes

3. **Confusion Matrix Analysis**
   - True Positives, False Positives, True Negatives, False Negatives
   - Derived rates: Specificity, Sensitivity, False positive/negative rates

### Key Functions

- `compute_classification_metrics()`: Calculate all performance metrics
- `generate_confusion_matrix()`: Create confusion matrix with statistics
- `evaluate_model_predictions()`: Complete evaluation pipeline

### Usage

```python
from src.evaluation.metrics import evaluate_model_predictions

results = evaluate_model_predictions(y_true, y_pred, y_proba)
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
```

---

## Module 2: subgroup_analysis.py

**Purpose**: Evaluate model performance across different demographic groups.

### Analysis Performed

**Available Subgroups**:

- **Region**: North America, Europe, Asia-Pacific, Middle East & Africa, Latin America
- **Note**: Gender analysis not available (gender data not present in dataset)

**Metrics by Subgroup**:

- All standard classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix elements
- Sample sizes per group

### Key Functions

- `analyze_region_performance()`: Complete regional performance analysis
- `evaluate_subgroup_performance()`: Evaluate model on specific subgroups
- `compare_subgroup_performance()`: Compare metrics across groups

### Current Results Summary

| Region               | Sample Size | Accuracy | ROC-AUC |
| -------------------- | ----------- | -------- | ------- |
| Latin America        | 679         | 96.32%   | 99.17%  |
| Asia-Pacific         | 650         | 96.00%   | 99.16%  |
| Europe               | 722         | 95.29%   | 98.80%  |
| Middle East & Africa | 714         | 95.24%   | 99.14%  |
| North America        | 735         | 94.97%   | 98.57%  |

### Usage

```python
from src.evaluation.subgroup_analysis import analyze_region_performance

model = load_best_model()
results_df = analyze_region_performance(model)
print(results_df)
```

---

## Module 3: fairness.py

**Purpose**: Detect potential biases by comparing error rates across subgroups.

### Fairness Metrics Analyzed

1. **Error Rates**
   - Overall error rate: (FP + FN) / Total
   - False positive rate: FP / (FP + TN)
   - False negative rate: FN / (FN + TP)

2. **Performance Disparities**
   - Absolute disparity: max - min across groups
   - Percentage disparity: (max - min) / min \* 100

3. **Bias Detection**
   - Flags disparities > 10% as potential concerns
   - Categorizes severity: low/moderate/high

### Current Fairness Assessment

**⚠️ Biases Detected**: 3 out of 5 metrics flagged

| Metric              | Disparity | Severity | Status     |
| ------------------- | --------- | -------- | ---------- |
| Error Rate          | 36.7%     | High     | ⚠️ FLAGGED |
| False Positive Rate | 96.5%     | High     | ⚠️ FLAGGED |
| False Negative Rate | 90.1%     | High     | ⚠️ FLAGGED |
| True Positive Rate  | 1.4%      | Low      | ✅ OK      |
| True Negative Rate  | 9.9%      | Low      | ✅ OK      |

### Key Functions

- `analyze_model_fairness()`: Complete fairness analysis
- `calculate_fairness_metrics()`: Compute fairness metrics for subgroups
- `detect_bias_indicators()`: Identify concerning disparities

### Usage

```python
from src.evaluation.fairness import analyze_model_fairness

fairness_results = analyze_model_fairness(model, subgroup_data)
print_fairness_report(fairness_results)
```

---

## Module 4: pipeline.py

**Purpose**: Orchestrate complete evaluation workflow.

### Evaluation Pipeline

```
1. Load trained model
2. Compute overall performance metrics
3. Analyze subgroup performance (by region)
4. Assess model fairness (bias detection)
5. Compile comprehensive results
6. Save to CSV: reports/results/model_evaluation.csv
```

### Output Format

The results CSV contains:

- **Overall Performance**: Model metrics on full test set
- **Subgroup Performance**: Metrics broken down by region
- **Fairness Disparities**: Bias indicators and disparity measurements

### Key Functions

- `run_complete_evaluation()`: Execute full evaluation pipeline

### Usage

```python
from src.evaluation.pipeline import run_complete_evaluation

results_df, metrics, subgroups, fairness = run_complete_evaluation()
print("Evaluation complete! Results saved to reports/results/model_evaluation.csv")
```

---

## Overall Model Performance

**Test Set Results** (700 samples):

- **Accuracy**: 91.57%
- **ROC-AUC**: 97.14%
- **Precision**: 91.39%
- **Recall**: 91.57%
- **F1-Score**: 91.42%

**Confusion Matrix**:

```
[[123, 37],  # TN, FP
 [22, 518]]   # FN, TP
```

---

## Fairness Concerns Identified

### ⚠️ High Priority Issues

1. **Error Rate Disparity**: 36.7% difference between best and worst performing regions
2. **False Positive Rate Disparity**: 96.5% difference - significant variation in false alarms
3. **False Negative Rate Disparity**: 90.1% difference - concerning variation in missed detections

### ✅ Acceptable Performance

- True Positive Rate disparity: 1.4% (good consistency)
- True Negative Rate disparity: 9.9% (reasonable variation)

---

## Recommendations

### Immediate Actions

1. **Investigate Regional Disparities**
   - Analyze why Latin America/Asia-Pacific perform better than North America
   - Check for data quality issues or feature representation gaps

2. **Bias Mitigation Strategies**
   - Consider reweighting training samples by region
   - Implement fairness constraints during model training
   - Add fairness-aware evaluation to future model iterations

3. **Additional Monitoring**
   - Track these metrics in production
   - Set up alerts for disparity thresholds
   - Regular fairness audits

### Model Strengths

- **Excellent Overall Performance**: 97.14% ROC-AUC
- **Strong Regional Performance**: All regions >94.97% accuracy
- **Good Calibration**: Consistent true positive rates across groups

---

## Running the Evaluation

### Option 1: Complete Pipeline

```bash
cd src/evaluation
python pipeline.py
```

### Option 2: From Project Root

```python
from run_evaluation import *
# Results saved to reports/results/model_evaluation.csv
```

### Option 3: Individual Components

```python
# Load model
from src.models.registry import load_best_model
model = load_best_model()

# Individual analyses
from src.evaluation.metrics import evaluate_model_predictions
from src.evaluation.subgroup_analysis import analyze_region_performance
from src.evaluation.fairness import analyze_model_fairness
```

---

## Data Sources

- **Model**: `models/trained/best_model.pkl` (GradientBoostingClassifier)
- **Features**: `data/processed/final_features.csv`
- **Subgroup Info**: `data/processed/cleaned_data.csv` (Region column)
- **Output**: `reports/results/model_evaluation.csv`

---

## Limitations

1. **Gender Analysis Unavailable**: Gender data not present in original dataset
2. **Regional Grouping**: Analysis limited to 5 major regions
3. **Binary Classification**: Fairness metrics designed for binary outcomes
4. **Sample Size Variation**: Regions have different sample sizes (650-735)

---

## Future Enhancements

1. **Additional Subgroups**: Include more demographic variables when available
2. **Intersectional Analysis**: Analyze combinations of subgroups
3. **Fairness Constraints**: Implement fairness-aware training
4. **Temporal Analysis**: Track fairness over time
5. **Explainability**: Add feature importance analysis by subgroup
