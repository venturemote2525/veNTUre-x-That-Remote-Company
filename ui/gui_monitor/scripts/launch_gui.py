#!/usr/bin/env python3
"""
Launch script for RTX 3060 Training Monitor GUI
"""
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import and launch main dashboard
from main_dashboard import main

if __name__ == "__main__":
    print("Launching RTX 3060 Training Monitor...")
    print("Real-time GPU monitoring and training progress")
    print("=" * 50)
    main()