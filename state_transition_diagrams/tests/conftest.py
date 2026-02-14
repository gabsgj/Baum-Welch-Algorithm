"""
conftest.py
===========
pytest configuration — ensures the library is on ``sys.path``
when running tests from the ``state_transition_diagrams/`` root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the parent of state_transition_diagrams/ so the package is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Also ensure demo/ is importable for test_demo_app
_lib_root = Path(__file__).resolve().parent
if str(_lib_root) not in sys.path:
    sys.path.insert(0, str(_lib_root))
