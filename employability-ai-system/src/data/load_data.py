"""
Load raw data from the data/raw/ directory.

This module handles loading the raw CSV file into a pandas DataFrame
with proper path handling and logging.
"""

import logging
from pathlib import Path
import pandas as pd

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_raw_data_path() -> Path:
    """
    Get the path to the raw data file.
    
    Returns:
        Path: Absolute path to the raw CSV file.
    """
    # Determine the project root by finding the src directory
    src_path = Path(__file__).parent.parent.parent
    raw_data_path = src_path / 'data' / 'raw' / 'global_graduate_employability_index.csv'
    return raw_data_path


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw dataset from the data/raw/ directory.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    
    Raises:
        FileNotFoundError: If the raw data file does not exist.
    """
    raw_path = get_raw_data_path()
    
    # Check if file exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    
    # Load the CSV file
    df = pd.read_csv(raw_path)
    
    # Log basic information about the loaded dataset
    logger.info(f"Loaded data from {raw_path}")
    logger.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {', '.join(df.columns.tolist())}")
    
    return df


if __name__ == "__main__":
    # When run directly, load and display basic info
    data = load_raw_data()
    print("\nFirst few rows:")
    print(data.head())
    print(f"\nData types:\n{data.dtypes}")
