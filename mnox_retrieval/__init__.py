"""Bridge package for running from source tree without editable install.

This file makes `python -m mnox_retrieval.cli ...` work when users run directly
from the repository root (e.g., D:\\MOE) by pointing package discovery to
`src/mnox_retrieval`.
"""

from __future__ import annotations

from pathlib import Path

_pkg_root = Path(__file__).resolve().parent
_src_pkg = _pkg_root.parent / "src" / "mnox_retrieval"

# Let Python resolve submodules (e.g. mnox_retrieval.cli) from src layout.
__path__ = [str(_src_pkg)]
