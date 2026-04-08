"""
Run evaluation pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.pipeline import run_complete_evaluation

if __name__ == "__main__":
    run_complete_evaluation()