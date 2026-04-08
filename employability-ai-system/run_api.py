#!/usr/bin/env python3
"""
Script to run the Employability AI API server.
"""

import sys
from pathlib import Path
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("Starting Employability AI API server...")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/api/v1/health")

    # Import the app directly
    from api.app import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )