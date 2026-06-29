from __future__ import annotations

from pathlib import Path

import pytest

from openai_compatible_server.backends.base import BaseModelBackend
from openai_compatible_server.backends.factory import create_model_backend
from openai_compatible_server.config import Settings
from openai_compatible_server.init_backend import backend_template, create_backend_file


def test_create_backend_file_in_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    destination = create_backend_file(Path("custom_backend.py"))
    backend = create_model_backend(
        Settings(
            model_id="custom-model",
            model_backend_class="./custom_backend.py:CustomModelBackend",
            log_dir=tmp_path / "logs",
        )
    )

    assert destination == (tmp_path / "custom_backend.py").resolve()
    assert destination.read_text(encoding="utf-8") == backend_template()
    assert isinstance(backend, BaseModelBackend)
    assert type(backend).__name__ == "CustomModelBackend"
    assert backend.model_id == "custom-model"


def test_create_backend_file_does_not_overwrite_without_force(tmp_path: Path) -> None:
    destination = tmp_path / "custom_backend.py"
    destination.write_text("# existing backend\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_backend_file(destination)

    assert destination.read_text(encoding="utf-8") == "# existing backend\n"


def test_create_backend_file_can_overwrite_with_force(tmp_path: Path) -> None:
    destination = tmp_path / "custom_backend.py"
    destination.write_text("# existing backend\n", encoding="utf-8")

    create_backend_file(destination, force=True)

    assert destination.read_text(encoding="utf-8") == backend_template()
