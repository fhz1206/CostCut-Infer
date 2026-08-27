"""tests.conftest — 共享 pytest fixtures（注入 Python 路径 + 清理临时目录）。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 把 python/ 加入 sys.path（pytest 在仓库根跑时不自动加）
_ROOT = Path(__file__).resolve().parents[1]
_PYTHON = _ROOT / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))


@pytest.fixture
def tmp_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="ccut-test-"))
    try:
        yield d
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _ROOT


@pytest.fixture(scope="session")
def python_dir() -> Path:
    return _PYTHON


@pytest.fixture(scope="session")
def python_exe() -> str:
    """测试环境 Python 解释器（用户要求 Python 3.14.5）。"""
    return sys.executable
