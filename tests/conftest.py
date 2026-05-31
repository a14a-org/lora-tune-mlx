"""Shared test fixtures and module loading helpers.

The data-preprocessing helpers live in ``data/preprocess_for_qwen.py``. The
``data`` directory is not a Python package, and importing the module by name
would also be shadowed by the top-level ``data`` directory on some setups, so we
load it directly from its file path. The pure functions exercised by the test
suite depend only on the standard library (``json``), which keeps the unit tests
free of the heavy ML dependencies (mlx, transformers, ...).
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PREPROCESS_PATH = _REPO_ROOT / "data" / "preprocess_for_qwen.py"


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def preprocess() -> ModuleType:
    """The ``preprocess_for_qwen`` module loaded from its file path."""
    return _load_module_from_path("preprocess_for_qwen", _PREPROCESS_PATH)
