from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def read_configuration(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read .env values, then apply system environment overrides."""
    system_env = dict(os.environ if environ is None else environ)
    configured_path = env_file or system_env.get("ENV_FILE") or ".env"
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    values.update(system_env)
    values["ENV_FILE"] = str(path.resolve())
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    api_key: str | None = None

    model_id: str = "demo-multimodal-model"
    model_backend_class: str | None = None
    model_max_concurrency: int = 1
    model_stream_chunk_size: int = 12

    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    log_rotation: str = "00:00"
    log_retention: str = "14 days"
    env_file: Path = Path(".env")

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        values = read_configuration(env_file, environ=environ)
        return cls(
            host=values.get("HOST", "0.0.0.0"),
            port=int(values.get("PORT", "8000")),
            reload=_as_bool(values.get("RELOAD", "false")),
            api_key=values.get("API_KEY") or None,
            model_id=values.get("MODEL_ID", "demo-multimodal-model"),
            model_backend_class=values.get("MODEL_BACKEND_CLASS") or None,
            model_max_concurrency=int(values.get("MODEL_MAX_CONCURRENCY", "1")),
            model_stream_chunk_size=int(values.get("MODEL_STREAM_CHUNK_SIZE", "12")),
            log_dir=Path(values.get("LOG_DIR", "logs")).expanduser().resolve(),
            log_level=values.get("LOG_LEVEL", "INFO").upper(),
            log_rotation=values.get("LOG_ROTATION", "00:00"),
            log_retention=values.get("LOG_RETENTION", "14 days"),
            env_file=Path(values["ENV_FILE"]),
        )
