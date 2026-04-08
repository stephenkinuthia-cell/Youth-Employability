"""
Validate the raw dataset for quality issues.

This module performs data validation checks without modifying the data.
It detects missing values, duplicates, data types, and basic sanity issues.
"""

import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for missing values in each column.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary mapping column names to missing value counts.
    """
    missing = df.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()
    return missing_dict


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Check for duplicate rows in the dataset.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Count of duplicate rows.
    """
    duplicate_count = df.duplicated().sum()
    return duplicate_count


def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get the data types of all columns.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary mapping column names to their data types.
    """
    return df.dtypes.astype(str).to_dict()


def check_salary_sanity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform sanity checks on salary columns (no negative values).
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary with sanity check results.
    """
    results = {}
    
    # Check for negative salary values
    salary_columns = [col for col in df.columns if 'salary' in col.lower()]
    
    for col in salary_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            negative_count = (df[col] < 0).sum()
            results[col] = {
                'has_negatives': negative_count > 0,
                'negative_count': int(negative_count)
            }
    
    return results


def check_employment_rate_sanity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform sanity checks on employment rate columns (should be 0-100).
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary with sanity check results.
    """
    results = {}
    
    # Check for employment rate columns
    rate_columns = [col for col in df.columns if 'employment_rate' in col.lower()
                   or 'rate' in col.lower() and '%' in col.lower()]
    
    for col in rate_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
            results[col] = {
                'valid_range': '0-100',
                'out_of_range_count': int(out_of_range),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    
    return results


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all validation checks on the dataset.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary containing all validation results.
    """
    validation_report = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'missing_values': check_missing_values(df),
        'duplicate_rows': check_duplicates(df),
        'data_types': check_data_types(df),
        'salary_sanity': check_salary_sanity(df),
        'employment_rate_sanity': check_employment_rate_sanity(df)
    }
    
    return validation_report


def print_validation_report(report: Dict[str, Any]) -> None:
    """
    Pretty-print the validation report to stdout.
    
    Args:
        report: Validation report dictionary from validate_dataset().
    """
    print("\n" + "=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\nBasic Info:")
    print(f"  Total Rows: {report['total_rows']}")
    print(f"  Total Columns: {report['total_columns']}")
    
    print(f"\nMissing Values:")
    if report['missing_values']:
        for col, count in report['missing_values'].items():
            print(f"  {col}: {count}")
    else:
        print("  None")
    
    print(f"\nDuplicate Rows: {report['duplicate_rows']}")
    
    print(f"\nSalary Sanity Checks:")
    if report['salary_sanity']:
        for col, result in report['salary_sanity'].items():
            status = "⚠ FAIL" if result['has_negatives'] else "✓ PASS"
            print(f"  {col}: {status} ({result['negative_count']} negative values)")
    else:
        print("  No salary columns found")
    
    print(f"\nEmployment Rate Sanity Checks:")
    if report['employment_rate_sanity']:
        for col, result in report['employment_rate_sanity'].items():
            status = "⚠ FAIL" if result['out_of_range_count'] > 0 else "✓ PASS"
            print(f"  {col}: {status}")
            print(f"    Range: {result['min']:.2f} - {result['max']:.2f} (expected 0-100)")
            print(f"    Out of range: {result['out_of_range_count']}")
    else:
        print("  No employment rate columns found")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test validation when run directly
    from load_data import load_raw_data
    
    df = load_raw_data()
    report = validate_dataset(df)
    print_validation_report(report)
