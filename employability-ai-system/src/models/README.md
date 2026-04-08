# Models Package Documentation

## Overview

This package implements a complete machine learning pipeline for employability prediction, including training, evaluation, hyperparameter tuning, model registry, and prediction capabilities.

**Target**: Binary classification (High Employability vs Low Employability)

- **High Employability**: Employment Rate > 85%
- **Low Employability**: Employment Rate ≤ 85%

**Input**: `data/processed/final_features.csv` (3,500 samples × 102 features)
**Output**: `models/trained/best_model.pkl` (GradientBoostingClassifier)

---

## Pipeline Architecture

```
Data Loading → Target Binarization → Train/Test Split → Model Training → Evaluation → Model Selection → Save Best Model
```

---

## Module 1: train.py

**Purpose**: Load data, preprocess target, split data, and train baseline models.

### Key Functions

- `load_data()`: Load final_features.csv
- `binarize_target()`: Convert continuous employment rate to binary (threshold=85%)
- `train_test_split_data()`: Stratified split (80/20)
- `train_baseline_models()`: Train LogisticRegression, RandomForest, GradientBoosting
- `run_training_pipeline()`: Complete workflow

### Usage

```python
from src.models.train import run_training_pipeline

models, X_train, X_test, y_train, y_test = run_training_pipeline()
```

---

## Module 2: evaluate.py

**Purpose**: Evaluate models using comprehensive classification metrics.

### Metrics Calculated

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (primary metric for selection)
- **Confusion Matrix**: Detailed prediction breakdown

### Key Functions

- `evaluate_model()`: Evaluate single model
- `evaluate_all_models()`: Evaluate all models, return DataFrame
- `print_evaluation_summary()`: Pretty-print results

### Example Output

```
MODEL EVALUATION SUMMARY
============================================================
GradientBoosting:
  Accuracy: 0.9157
  Precision: 0.9201
  Recall: 0.9157
  F1_Score: 0.9157
  ROC_AUC: 0.9714
  Confusion Matrix:
    [[123, 37]
     [22, 518]]
```

---

## Module 3: tune.py

**Purpose**: Perform hyperparameter tuning using GridSearchCV.

### Tuning Strategy

- **Algorithm**: GridSearchCV with 5-fold cross-validation
- **Scoring**: ROC-AUC (optimized metric)
- **Models Tuned**: All baseline models
- **Parallel**: Uses all available CPU cores

### Parameter Grids

```python
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
```

### Usage

```python
from src.models.tune import tune_all_models

tuned_models = tune_all_models(baseline_models, X_train, y_train)
```

---

## Module 4: registry.py

**Purpose**: Store and retrieve trained models.

### Features

- **Save Models**: Persist models to disk using joblib
- **Load Models**: Retrieve specific or best model
- **Model Info**: Get metadata about saved models
- **Directory Management**: Automatic creation of models/trained/

### Key Functions

- `save_model()`: Save individual model
- `save_best_model()`: Save best performing model
- `load_model()`: Load specific model
- `load_best_model()`: Load best model
- `list_saved_models()`: List all saved models

### Usage

```python
from src.models.registry import save_best_model, load_best_model

# Save
save_best_model(best_model, "GradientBoosting")

# Load
model = load_best_model()
```

---

## Module 5: predict.py

**Purpose**: Make predictions using the trained best model.

### Features

- **Single Predictions**: Predict class labels
- **Probabilities**: Get prediction probabilities
- **Batch Predictions**: Handle multiple samples
- **Formatted Output**: Readable DataFrame results

### Key Functions

- `load_best_model()`: Load the best model
- `predict()`: Make predictions
- `predict_proba()`: Get probabilities
- `make_predictions()`: Complete prediction pipeline
- `format_predictions()`: Format results nicely

### Usage

```python
from src.models.predict import make_predictions

# Make predictions on new data
results = make_predictions(X_new, include_proba=True)
print(results)
```

### Example Output

```
   Prediction Prediction_Label  Probability_High_Employability  Probability_Low_Employability
0           1   High_Employability                      0.987654                      0.012346
1           0    Low_Employability                      0.123456                      0.876544
```

---

## Module 6: pipeline.py

**Purpose**: Orchestrate the complete ML workflow.

### Pipeline Steps

1. **Train** baseline models
2. **Evaluate** baseline models
3. **Tune** hyperparameters (optional, skipped for speed)
4. **Select** best model by ROC-AUC
5. **Save** best model to models/trained/best_model.pkl

### Usage

```python
from src.models.pipeline import main

best_model_name, best_score = main()
print(f"Best: {best_model_name} (ROC-AUC: {best_score:.4f})")
```

---

## Model Performance

### Baseline Results (Test Set)

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC    |
| ------------------ | -------- | --------- | ------ | -------- | ---------- |
| GradientBoosting   | 0.9157   | 0.9201    | 0.9157 | 0.9157   | **0.9714** |
| LogisticRegression | 0.9129   | 0.9130    | 0.9129 | 0.9129   | 0.9689     |
| RandomForest       | 0.9100   | 0.9101    | 0.9100 | 0.9100   | 0.9534     |

### Best Model Selected

**GradientBoostingClassifier** with ROC-AUC: 0.9714

---

## File Structure

```
src/models/
├── __init__.py          # Package initialization
├── train.py             # Data loading, preprocessing, model training
├── evaluate.py          # Model evaluation metrics
├── tune.py              # Hyperparameter tuning
├── predict.py           # Prediction interface
├── registry.py          # Model storage/retrieval
├── pipeline.py          # Complete workflow orchestration
└── README.md            # This documentation

models/trained/
├── best_model.pkl       # Best performing model (GradientBoosting)
├── GradientBoosting.pkl # Individual saved models
├── LogisticRegression.pkl
└── RandomForest.pkl
```

---

## Key Design Decisions

### 1. Target Binarization

- **Threshold**: 85% employment rate
- **Rationale**: Clear business interpretation (high vs low employability)
- **Class Distribution**: ~77% high employability, ~23% low employability

### 2. Model Selection

- **Primary Metric**: ROC-AUC (handles class imbalance well)
- **Secondary**: F1-Score (balances precision/recall)
- **Winner**: GradientBoosting (best ROC-AUC: 0.9714)

### 3. Evaluation Strategy

- **Stratified Split**: Maintains class distribution
- **Comprehensive Metrics**: Accuracy + Precision + Recall + F1 + ROC-AUC
- **Confusion Matrix**: Detailed error analysis

### 4. Hyperparameter Tuning

- **Skipped for Speed**: Baseline models performed excellently
- **Available**: GridSearchCV implementation ready if needed
- **Scalable**: Parameter grids can be expanded

### 5. Model Persistence

- **Format**: joblib (faster than pickle for sklearn)
- **Naming**: Descriptive filenames
- **Registry**: Centralized model management

---

## Usage Examples

### Complete Pipeline

```python
# Run full pipeline
from src.models.pipeline import main
best_model, score = main()

# Make predictions
from src.models.predict import make_predictions
results = make_predictions(new_data)
```

### Individual Components

```python
# Train only
from src.models.train import run_training_pipeline
models, X_train, X_test, y_train, y_test = run_training_pipeline()

# Evaluate only
from src.models.evaluate import evaluate_all_models
results = evaluate_all_models(models, X_test, y_test)

# Save/load models
from src.models.registry import save_best_model, load_best_model
save_best_model(models['GradientBoosting'], 'GradientBoosting')
model = load_best_model()
```

---

## Reproducibility

- **Random State**: 42 (fixed for all operations)
- **Stratified Split**: Maintains class proportions
- **Sklearn Pipelines**: Consistent preprocessing
- **Joblib**: Deterministic model saving/loading

---

## Troubleshooting

### ModuleNotFoundError

**Solution**: Run from project root or add to PYTHONPATH

### Memory Issues

**Solution**: Reduce n_estimators in RandomForest/GradientBoosting

### Poor Performance

**Solution**: Check data preprocessing, try different hyperparameters

### Model Not Found

**Solution**: Run training pipeline first to create models/trained/best_model.pkl

---

## Next Steps

After model training:

1. **Model Validation**: Cross-validation on full dataset
2. **Feature Importance**: Analyze which features drive predictions
3. **Model Interpretability**: SHAP/LIME explanations
4. **Deployment**: Containerize model for production
5. **Monitoring**: Set up performance tracking

---

## Glossary

- **ROC-AUC**: Area Under Receiver Operating Characteristic curve
- **Stratified Split**: Train/test split maintaining class proportions
- **Hyperparameter Tuning**: Optimizing model configuration parameters
- **Baseline Model**: Initial model without tuning
- **Confusion Matrix**: Table showing prediction vs actual results
