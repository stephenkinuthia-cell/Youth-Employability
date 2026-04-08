#!/usr/bin/env python3
"""
Employability AI System - Main Entry Point

This script provides an overview of available commands and quick access
to the main functionality of the employability prediction system.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Employability AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:

Data Processing:
  python src/data/preprocess.py    # Run data cleaning pipeline

Model Training:
  python quick_train.py           # Train and evaluate models

Evaluation:
  python run_evaluation.py        # Run model evaluation & fairness analysis

Recommendations:
  python test_recommendations.py  # Single user recommendations
  python run_recommendations.py   # Batch recommendations (20 users)

Examples:
  python test_recommendations.py  # Quick test with user_0
  python run_recommendations.py   # Generate recommendations for sample users
  python run_evaluation.py        # Evaluate model performance
        """
    )

    parser.add_argument(
        '--version', action='version', version='Employability AI System v1.0'
    )

    parser.add_argument(
        '--status', action='store_true',
        help='Show current project status'
    )

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if len(sys.argv) == 1:
        parser.print_help()

def show_status():
    """Show current project status"""
    print("=" * 60)
    print("EMPLOYABILITY AI SYSTEM - STATUS")
    print("=" * 60)

    # Get project root
    project_root = Path(__file__).parent

    # Check if key files exist
    checks = [
        ("Raw data", project_root / "data" / "raw" / "global_graduate_employability_index.csv"),
        ("Processed features", project_root / "data" / "processed" / "final_features.csv"),
        ("Trained model", project_root / "models" / "trained" / "best_model.pkl"),
        ("Evaluation results", project_root / "reports" / "results" / "model_evaluation.csv"),
        ("Recommendation modules", project_root / "src" / "recommendation"),
    ]

    for name, path in checks:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path.relative_to(project_root) if path.exists() else path.name}")

    print("\n" + "=" * 60)
    print("QUICK START COMMANDS:")
    print("=" * 60)
    print("1. Test single user recommendations:")
    print("   python test_recommendations.py")
    print()
    print("2. Generate batch recommendations:")
    print("   python run_recommendations.py")
    print()
    print("3. Run model evaluation:")
    print("   python run_evaluation.py")
    print()
    print("4. Train new models:")
    print("   python quick_train.py")

if __name__ == "__main__":
    main()
