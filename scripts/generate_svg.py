#!/usr/bin/env python3
"""
Convenience CLI wrapper for SVG generation
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.generate_svg import main

if __name__ == "__main__":
    main()
