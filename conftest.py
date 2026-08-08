"""
Ensures the repository root is on sys.path so test modules can
`import losses` (and other top-level modules) regardless of how pytest
is invoked (plain `pytest`, `python -m pytest`, or from a different cwd).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
