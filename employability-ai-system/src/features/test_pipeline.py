#!/usr/bin/env python
"""Quick test of the feature engineering pipeline."""

from pipeline import run_pipeline

print("\nRunning feature engineering pipeline...\n")
X, y, path, report = run_pipeline(validate=True)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Final dataset shape: {X.shape}")
print(f"Saved to: {path}")
print(f"Total selected features: {len(report['selected_features'])}")

print(f"\nTop 15 features by importance:")
for i, feat in enumerate(report['importance_scores'][:15], 1):
    print(f"  {i:2d}. {feat['Feature']:<40s} {feat['Importance']:.6f}")

print("\n" + "=" * 70)
