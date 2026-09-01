"""
Root conftest.py — ensures the project root is on sys.path so that
`backend` and `training` packages are importable regardless of where
pytest is launched from.
"""
import sys
from pathlib import Path

# Insert project root at the front of the path (idempotent)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
