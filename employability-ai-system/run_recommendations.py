"""
Run recommendation pipeline for sample users.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommendation.pipeline import run_recommendation_pipeline

if __name__ == "__main__":
    # Run for 20 sample users (quick demo)
    print("Running recommendation pipeline for sample users...")
    recommendations_df, summaries_df = run_recommendation_pipeline(sample_size=20)