# Data Pipeline Documentation

## Overview

This data pipeline provides a clean, modular approach to preprocessing the graduate employability dataset. It follows the ETL (Extract-Transform-Load) pattern with quality validation built in.

## Pipeline Architecture

```
Raw Data (CSV)
    ↓
load_data.py
    ↓
Raw DataFrame
    ↓
validate_data.py (no modifications)
    ↓
Validation Report
    ↓
preprocess.py (cleaning & imputation)
    ↓
Cleaned DataFrame
    ↓
Save to data/processed/cleaned_data.csv
```

## Module Descriptions

### 1. `load_data.py`
**Purpose**: Load raw dataset into memory.

**Key Functions**:
- `get_raw_data_path()`: Returns the path to raw data using pathlib
- `load_raw_data()`: Loads CSV and logs basic stats

**Key Features**:
- Uses pathlib for cross-platform compatibility
- No hardcoded paths
- Logs dataset shape and columns
- Raises FileNotFoundError if file missing

**Usage**:
```python
from load_data import load_raw_data

df = load_raw_data()
```

### 2. `validate_data.py`
**Purpose**: Assess data quality without modifying the dataset.

**Key Functions**:
- `check_missing_values()`: Count nulls per column
- `check_duplicates()`: Count duplicate rows
- `check_data_types()`: Get column dtypes
- `check_salary_sanity()`: Flag negative salaries
- `check_employment_rate_sanity()`: Flag rates outside 0-100%
- `validate_dataset()`: Run all checks, return report dict
- `print_validation_report()`: Pretty-print the report

**Key Features**:
- No data modification
- Combines multiple sanity checks
- Human-readable output formatting
- Returns structured report dictionary

**Usage**:
```python
from load_data import load_raw_data
from validate_data import validate_dataset, print_validation_report

df = load_raw_data()
report = validate_dataset(df)
print_validation_report(report)
```

### 3. `preprocess.py`
**Purpose**: Clean and prepare data for modeling.

**Key Functions**:
- `clean_column_names()`: Strip whitespace from column names
- `remove_duplicate_rows()`: Drop exact duplicates
- `impute_numeric_columns()`: Fill numeric nulls with median
- `impute_categorical_columns()`: Fill categorical nulls with 'Unknown'
- `preprocess_data()`: Apply all cleaning steps in sequence
- `save_processed_data()`: Write CSV to data/processed/
- `run_pipeline()`: Full pipeline from load to save

**Key Features**:
- Modular, reusable cleaning functions
- Skips steps if not needed (e.g., no duplicates)
- Uses scikit-learn SimpleImputer for robustness
- Creates output directory if missing
- Returns both DataFrame and file path

**Preprocessing Steps**:
1. Clean column names (strip whitespace)
2. Remove duplicate rows
3. Impute numeric columns (median strategy)
4. Impute categorical columns ('Unknown' strategy)

**Usage**:
```python
from preprocess import run_pipeline

df_clean, output_path = run_pipeline(validate=True)
```

## Running the Pipeline

### Option 1: Run from src/data directory
```bash
cd src/data
python preprocess.py
```

### Option 2: Run from project root
```bash
cd src/data
python preprocess.py
```

### Option 3: Import in your script
```python
from src.data.preprocess import run_pipeline

df_clean, output_path = run_pipeline(validate=True)
print(f"Data saved to: {output_path}")
```

## Output

The pipeline generates:
- **cleaned_data.csv**: Cleaned dataset ready for feature engineering
- **Console logs**: Step-by-step progress and validation reports

### File Location
```
data/
├── raw/
│   └── global_graduate_employability_index.csv
└── processed/
    └── cleaned_data.csv  ← Output file
```

## Data Quality Checks

The validation report provides:
- **Missing Values**: Count per column
- **Duplicates**: Total duplicate rows
- **Data Types**: Type of each column
- **Salary Sanity**: Flags negative salaries
- **Employment Rate Sanity**: Flags rates outside 0-100%

### Example Validation Report
```
======================================================================
DATA VALIDATION REPORT
======================================================================

Basic Info:
  Total Rows: 3500
  Total Columns: 18

Missing Values:
  None

Duplicate Rows: 0

Salary Sanity Checks:
  Average_Starting_Salary_USD: ⚠ FAIL (1 negative values)

Employment Rate Sanity Checks:
  Employment_Rate_12_Months (%): ✓ PASS
    Range: 68.00 - 100.00 (expected 0-100)
    Out of range: 0
```

## Preprocessing Details

### Missing Value Imputation
- **Numeric**: Median imputation (robust to outliers)
- **Categorical**: Fill with 'Unknown' (preserves categories)

### Column Name Cleaning
- Strips leading/trailing whitespace
- Does NOT lowercase (preserves column semantics)

### Duplicate Removal
- Exact row duplicates removed
- Logs number of rows removed

## Notes

- **No feature engineering** in this module (saved for later)
- **No encoding** (categorical variables remain as-is)
- **No data leakage** considerations (safe for preprocessing)
- **Reproducible**: Same input always produces same output
- **Logging**: Detailed logs aid debugging

## Next Steps

After running this pipeline, the cleaned dataset is ready for:
1. **Exploratory Data Analysis** (02_feature_engineering.ipynb)
2. **Feature Engineering** (src/features/)
3. **Model Training** (03_modeling.ipynb)

## Troubleshooting

### FileNotFoundError
- Ensure `data/raw/global_graduate_employability_index.csv` exists
- Check file path and permissions

### ModuleNotFoundError
- Run from `src/data` directory: `cd src/data && python preprocess.py`
- Or add src/ to PYTHONPATH

### Path Issues
- Pipeline uses pathlib for cross-platform compatibility
- Should work on Windows, macOS, Linux without modification

## Dependencies
- pandas
- scikit-learn (for imputation)
- logging (standard library)
- pathlib (standard library)

