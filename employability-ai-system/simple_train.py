"""
Simple script to train and save a basic GradientBoosting model.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data_path = Path(__file__).parent / "data" / "processed" / "final_features.csv"
df = pd.read_csv(data_path)
logger.info(f"Loaded {len(df)} samples")

# Binarize target
target_col = "Employment_Rate_12_Months (%)"
df[target_col] = (df[target_col] > 85).astype(int)

# Split
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = GradientBoostingClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
logger.info(".4f")

# Save model
models_dir = Path("models/trained")
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "best_model.pkl"
joblib.dump(model, model_path)
logger.info(f"Model saved to {model_path}")

print("✅ Model trained and saved successfully!")