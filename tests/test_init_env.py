from __future__ import annotations

from pathlib import Path

import pytest

from openai_compatible_server.config import Settings
from openai_compatible_server.init_env import create_env_file, env_template


def test_create_env_file_in_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    destination = create_env_file(Path(".env"))
    settings = Settings.from_env(environ={})

    assert destination == (tmp_path / ".env").resolve()
    assert destination.read_text(encoding="utf-8") == env_template()
    assert settings.env_file == destination
    assert settings.model_id == "demo-multimodal-model"
    assert settings.port == 8000


def test_create_env_file_does_not_overwrite_without_force(tmp_path: Path) -> None:
    destination = tmp_path / ".env"
    destination.write_text("MODEL_ID=existing-model\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_env_file(destination)

    assert destination.read_text(encoding="utf-8") == "MODEL_ID=existing-model\n"


def test_create_env_file_can_overwrite_with_force(tmp_path: Path) -> None:
    destination = tmp_path / ".env"
    destination.write_text("MODEL_ID=existing-model\n", encoding="utf-8")

    create_env_file(destination, force=True)

    assert destination.read_text(encoding="utf-8") == env_template()


def test_packaged_template_matches_project_example() -> None:
    project_template = Path(__file__).parents[1] / ".env.example"

    assert project_template.read_text(encoding="utf-8") == env_template()
