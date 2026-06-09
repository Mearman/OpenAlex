"""Pytest configuration: ensure the repo root is importable.

Inserts the project root (parent of this tests/ directory) at the front of
sys.path so ``import sync`` resolves without needing an editable install.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
