"""
Test the saved model.
"""

from src.models.predict import make_predictions
import numpy as np

# Create dummy data (102 features as in our dataset)
X = np.random.randn(5, 102)

# Make predictions
result = make_predictions(X)
print('Predictions shape:', result.shape)
print(result)