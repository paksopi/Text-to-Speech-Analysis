"""Pytest configuration: make src/ and src/eval/ importable without needing
the package installed, matching the sys.path tricks the scripts themselves use.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
for _p in (SRC, SRC / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
