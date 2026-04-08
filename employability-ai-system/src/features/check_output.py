import pandas as pd
from pathlib import Path

# Navigate to project root (3 levels up from src/features)
path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'final_features.csv'
df = pd.read_csv(path)

print(f"Shape: {df.shape}")
print(f"Features: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nFirst 5 column names:")
for i, col in enumerate(df.columns[:5], 1):
    print(f"  {i}. {col}")
print(f"  ... ({len(df.columns) - 5} more)")

print(f"\nTarget column: {df.columns[-1]}")
print(f"Target mean: {df.iloc[:, -1].mean():.2f}")
print(f"Target std:  {df.iloc[:, -1].std():.2f}")
print(f"\nFile location: {path}")
print(f"File size: {path.stat().st_size / 1024:.1f} KB")
