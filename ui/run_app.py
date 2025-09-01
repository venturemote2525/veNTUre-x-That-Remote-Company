#!/usr/bin/env python3
"""
Launch script for the Food Analysis Streamlit App
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    app_path = Path(__file__).parent / "food_analysis_app.py"
    
    print("🍎 Starting Food Analysis Streamlit App...")
    print(f"📁 App location: {app_path}")
    print("🌐 Access the app at: http://localhost:8501")
    print("⚡ Models will load automatically on first use")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down the app...")
    except Exception as e:
        print(f"❌ Error running the app: {e}")

if __name__ == "__main__":
    main()