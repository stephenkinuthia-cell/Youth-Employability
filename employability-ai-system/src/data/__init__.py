"""
Data pipeline module for loading, validating, and preprocessing the dataset.

This package provides three core modules:
- load_data: Load raw data from CSV
- validate_data: Validate data quality
- preprocess: Clean and prepare data for modeling

Usage:
    from src.data.preprocess import run_pipeline
    df_clean, output_path = run_pipeline(validate=True)
"""
