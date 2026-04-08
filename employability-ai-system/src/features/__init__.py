"""
Feature engineering modules for the ML pipeline.

Modules:
- build_features: Create derived features from raw data
- transform_features: Encode and scale features
- select_features: Remove redundant features
- pipeline: Orchestrate the complete workflow

Usage:
    from src.features.pipeline import run_pipeline
    
    X, y, output_path, report = run_pipeline(validate=True)
"""
