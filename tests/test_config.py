from __future__ import annotations

from pathlib import Path

from openai_compatible.config import Settings, read_configuration


def test_settings_load_from_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HOST=127.0.0.1",
                "PORT=9000",
                "RELOAD=true",
                "API_KEY=dotenv-secret",
                "MODEL_ID=dotenv-model",
                "MODEL_MAX_CONCURRENCY=3",
                "LOG_LEVEL=debug",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(env_file, environ={})

    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.reload is True
    assert settings.api_key == "dotenv-secret"
    assert settings.model_id == "dotenv-model"
    assert settings.model_max_concurrency == 3
    assert settings.log_level == "DEBUG"
    assert settings.env_file == env_file.resolve()


def test_system_environment_overrides_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / "server.env"
    env_file.write_text("PORT=8000\nMODEL_ID=dotenv-model\n", encoding="utf-8")

    settings = Settings.from_env(
        env_file,
        environ={"PORT": "9100", "MODEL_ID": "system-model"},
    )

    assert settings.port == 9100
    assert settings.model_id == "system-model"


def test_env_file_can_be_selected_through_environment(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("MODEL_ID=selected-model\n", encoding="utf-8")

    values = read_configuration(environ={"ENV_FILE": str(env_file)})

    assert values["MODEL_ID"] == "selected-model"
    assert values["ENV_FILE"] == str(env_file.resolve())
