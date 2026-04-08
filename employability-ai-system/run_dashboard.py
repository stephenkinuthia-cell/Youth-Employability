#!/usr/bin/env python3
"""
Script to run the Employability AI Streamlit dashboard.
"""

import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("Starting Employability AI Dashboard...")
    print("Dashboard will be available at: http://localhost:8502")

    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "src" / "dashboard" / "app.py"),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--server.port", "8502"
    ])