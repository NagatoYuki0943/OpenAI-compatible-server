from pathlib import Path

import pytest

from openai_compatible_server.build import build_wheel


def test_build_requires_pyproject(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        build_wheel(tmp_path, tmp_path / "dist")
