# Feature Engineering Pipeline Documentation

## Overview

This feature engineering pipeline transforms raw cleaned data into a model-ready dataset through three modular stages: feature building, feature transformation, and feature selection.

**Output**: `data/processed/final_features.csv` (3,500 samples × 103 features)

---

## Pipeline Architecture

```
Cleaned Data (18 features)
    ↓
BUILD FEATURES (derive new features)
    18 → 26 features
    ↓
TRANSFORM FEATURES (encode & scale)
    26 → 179 features (after one-hot encoding)
    ↓
SELECT FEATURES (remove redundancy)
    179 → 102 features (final)
    ↓
SAVE to final_features.csv
```

---

## Module 1: build_features.py

**Purpose**: Create meaningful, interpretable derived features from raw variables.

### Features Created

1. **Interaction Features** (3 features)
   - `Demand_x_Reputation`: Skill demand × Employer reputation
   - `Demand_x_Remote`: Skill demand × Remote availability
   - `Reputation_x_Remote`: Reputation × Remote availability

2. **Aggregation Features** (1 feature)
   - `Unique_Skills_Count`: Count of unique required skills

3. **Time-Based Features** (2 features)
   - `Years_Since_Graduation`: Career stage proxy (Year - Graduation_Year)
   - `Career_Stage`: Categorical (Early/Mid/Late) based on years

4. **Salary-Based Features** (1 feature)
   - `Salary_Per_Year_Experience`: Starting salary / years experience

5. **Location Features** (1 feature)
   - `Is_US`: Binary indicator for USA

### Key Functions

- `load_cleaned_data()`: Load from data/processed/cleaned_data.csv
- `build_all_features()`: Orchestrate all feature creation steps
- `create_*_features()`: Individual feature builders

### Usage

```python
from build_features import load_cleaned_data, build_all_features

df = load_cleaned_data()
df_featured = build_all_features(df)
```

---

## Module 2: transform_features.py

**Purpose**: Standardize and encode features for machine learning models.

### Transformations

1. **One-Hot Encoding** (Categorical variables)
   - Converts categorical variables to binary features
   - Drop first category to avoid multicollinearity
   - Example: `Degree_Level` → `Degree_Level_Master`, `Degree_Level_PhD`

2. **Standard Scaling** (Numerical variables)
   - Centers around 0, scales to unit variance
   - Ensures all features have comparable ranges
   - Prevents large-valued features from dominating learning

3. **Handled Features**
   - **Categorical**: Country, Region, Degree_Level, Field_of_Study, Top_Industry, Job_Role, Skill_1/2/3, Career_Stage
   - **Numerical**: All numerical columns and interaction features

### Key Functions

- `identify_feature_types()`: Auto-detect categorical vs numerical
- `build_preprocessor()`: Create sklearn ColumnTransformer
- `fit_and_transform()`: Fit on data and transform
- `transform_new_data()`: Transform new data using fitted preprocessor
- `get_feature_names_after_transform()`: Get transformed feature names

### Usage

```python
from transform_features import fit_and_transform

df_transformed, preprocessor = fit_and_transform(
    df_featured,
    exclude_cols=['University_Name']  # Too high-cardinality
)

# Later, on new data:
from transform_features import transform_new_data
df_new_transformed = transform_new_data(df_new, preprocessor)
```

---

## Module 3: select_features.py

**Purpose**: Identify and remove redundant/irrelevant features.

### Feature Selection Strategy

1. **Low-Variance Filtering** (Removes constant features)
   - Removes features with variance < 0.01
   - These contribute minimally to predictions

2. **Correlation Filtering** (Removes redundancy)
   - Identifies pairs with |correlation| > 0.95
   - Keeps one feature from each pair
   - Avoids multicollinearity issues

3. **Feature Importance Filtering** (Keeps predictive features)
   - Uses Random Forest to score feature importance
   - Keeps features with importance ≥ 0.0001
   - Lightweight model (max_depth=10) used for scoring only

### Selection Report includes

- Count of removed features (by type)
- Feature importance scores (all features)
- Final selected features (names only)

### Key Functions

- `remove_low_variance_features()`: Filter near-constant features
- `remove_highly_correlated_features()`: Remove redundant pairs
- `calculate_feature_importance()`: Score features with RF
- `perform_feature_selection()`: Orchestrate full selection
- `print_selection_report()`: Pretty-print detailed report

### Example Report

```
FEATURE SELECTION REPORT
======================================================================
Original Features: 179
Final Features: 102
Reduction: 77 features removed (43.0%)

Removed Features (Low Variance):
  - Unique_Skills_Count
  - Years_Since_Graduation

Removed Features (High Correlation):
  - Year
  - Demand_x_Remote
  - ... (8 more)

Top 10 Features by Importance:
  1. Employment_Rate_6_Months (%): 0.4521
  2. Average_Starting_Salary_USD: 0.1234
  ...
```

### Usage

```python
from select_features import perform_feature_selection, print_selection_report

X_selected, report = perform_feature_selection(
    X_transformed, y,
    importance_threshold=0.0001
)

print_selection_report(report)
```

---

## Module 4: pipeline.py

**Purpose**: Orchestrate the complete workflow from loading to saving.

### Pipeline Flow

```
1. Load cleaned data
2. Build derived features
3. Transform features (encode + scale)
4. Select relevant features
5. Save final dataset
```

### Key Functions

- `run_pipeline()`: Execute complete pipeline
- `save_final_features()`: Write to CSV with target
- `get_features_output_path()`: Get output path

### Usage

```python
from pipeline import run_pipeline

X, y, output_path, report = run_pipeline(validate=True)

print(f"Output: {output_path}")
print(f"Shape: {X.shape}")
print(f"Features: {len(report['selected_features'])}")
```

---

## Running the Pipeline

### Option 1: Full Pipeline with Validation

```bash
cd src/features
python pipeline.py
```

### Option 2: From Python Script

```python
from src.features.pipeline import run_pipeline

X_final, y_final, output_path, report = run_pipeline(validate=True)
print(f"Model-ready dataset: {output_path}")
```

### Option 3: Step-by-Step

```python
from src.features.build_features import load_cleaned_data, build_all_features
from src.features.transform_features import fit_and_transform
from src.features.select_features import perform_feature_selection

# Step 1
df = load_cleaned_data()
df = build_all_features(df)

# Step 2
df_trans, prep = fit_and_transform(df, exclude_cols=['University_Name'])

# Step 3
X, report = perform_feature_selection(df_trans, df['Employment_Rate_12_Months (%)'])
```

---

## Output Dataset

**Location**: `data/processed/final_features.csv`

### Structure

| Property       | Value                         |
| -------------- | ----------------------------- |
| Rows           | 3,500                         |
| Columns        | 103                           |
| Features       | 102                           |
| Target         | Employment_Rate_12_Months (%) |
| Missing Values | 0                             |
| File Size      | 1.9 MB                        |

### Feature Types

- **Numerical** (scaled): Salary, demand scores, reputation, etc.
- **Categorical** (one-hot encoded): Skills, jobs, fields, locations, etc.
- **Target**: Employment_Rate_12_Months (%)

### Feature Summary

```
Employment_Rate_6_Months (%)          (numerical, scaled)
Average_Starting_Salary_USD            (numerical, scaled)
Reputation_x_Remote                    (interaction, scaled)
Employer_Reputation_Score (1–100)      (numerical, scaled)
Demand_x_Reputation                    (interaction, scaled)
Skill_Demand_Score (1–100)             (numerical, scaled)
Remote_Work_Availability (%)           (numerical, scaled)
Graduation_Year                        (numerical, scaled)
Degree_Level_Master                    (categorical, one-hot)
Degree_Level_PhD                       (categorical, one-hot)
Skill_1_Public Policy                  (categorical, one-hot)
Skill_2_Data Visualization             (categorical, one-hot)
... (92 more features)
Employment_Rate_12_Months (%)          (TARGET - not scaled)
```

---

## Key Design Decisions

### 1. Feature Building

- **Interaction features**: Capture combined effects of related variables
- **Domain-driven**: All features are interpretable and business-meaningful
- **No encoding here**: Keeps this stage simple and generalized

### 2. Feature Transformation

- **StandardScaler**: Centers features, handles different ranges
- **OneHotEncoder** with drop='first': Avoids dummy variable trap
- **ColumnTransformer**: Applies different transforms to different types
- **Reproducible**: Fitted transformer can be applied to new data

### 3. Feature Selection

- **Multiple filters**: Variance + Correlation + Importance
- **Lightweight model**: RF with max_depth=10 (not for production)
- **Low threshold**: 0.0001 importance to avoid over-filtering
- **Interpretability**: Removed features documented in report

---

## Important Notes

### ✅ What This Pipeline Does

- Creates 8 new derived features from 18 raw features
- One-hot encodes 10 categorical variables (179 binary features)
- Identifies and removes 77 redundant/low-value features
- Produces a clean, scaled, interpretable 102-feature dataset
- No target leakage
- Reproducible transformations via sklearn pipelines
- Comprehensive audit trail (selection reports)

### ❌ What This Pipeline Does NOT Do

- Model training (only lightweight feature importance)
- Hyperparameter tuning
- Cross-validation
- Feature interactions beyond products/sums
- Target variable modification
- Data imputation (assumes clean data)

---

## Reproducibility

All transformations are saved and documented:

1. **ColumnTransformer** from `transform_features.py` can be pickled and reused
2. **Feature names** are preserved through all stages
3. **Selection criteria** are logged and reported
4. **Random states** are fixed (42) for consistent results

---

## Troubleshooting

### Pipeline Runs But Selects Too Few Features

**Solution**: Lower the `importance_threshold` in `run_pipeline()` (default: 0.0001)

### ModuleNotFoundError

**Solution**: Run from `src/features/` directory or add to PYTHONPATH

### Memory Issues with Large Datasets

**Solution**: Reduce number of one-hot categories by excluding high-cardinality columns

### Features Differ Between Runs

**Solution**: Check that `random_state=42` is used in all model components

---

## Next Steps

After this feature engineering, you can:

1. **Split data**: Use for train/validation/test split
2. **Train models**: Use final_features.csv as input
3. **Evaluate**: Compare model performance
4. **Hyperparameter tune**: Optimize on final feature set

---

## Glossary

- **Interaction feature**: Product or combination of two existing features
- **One-hot encoding**: Convert categorical to binary features
- **StandardScaling**: Normalize features to mean=0, std=1
- **Feature importance**: Relative contribution of feature to predictions
- **Target leakage**: Using information that wouldn't be available at prediction time
