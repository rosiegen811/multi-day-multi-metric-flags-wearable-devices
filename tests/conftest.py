"""Shared pytest configuration for the Wearable Flag Pipeline test suite.

Ensures the repository root (which holds the ``data``/``flag``/``eval``/
``app``/``export`` packages) is importable when tests run from any working
directory.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
