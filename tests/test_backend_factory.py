from __future__ import annotations

from pathlib import Path

import pytest

from openai_compatible_server.backends import OCSGenerationRequest
from openai_compatible_server.backends.custom import CustomModelBackend
from openai_compatible_server.backends.factory import create_model_backend
from openai_compatible_server.config import Settings


def _settings(tmp_path: Path, backend_class: str) -> Settings:
    return Settings(
        model_id="factory-test-model",
        model_backend_class=backend_class,
        log_dir=tmp_path / "logs",
    )


def test_factory_loads_installed_module_backend(tmp_path: Path) -> None:
    backend = create_model_backend(
        _settings(
            tmp_path,
            "openai_compatible_server.backends.custom:CustomModelBackend",
        )
    )

    assert isinstance(backend, CustomModelBackend)
    assert backend.model_id == "factory-test-model"


def test_factory_loads_backend_from_python_file(tmp_path: Path) -> None:
    backend_file = tmp_path / "my_backend.py"
    backend_file.write_text(
        "\n".join(
            [
                "from typing import Any",
                "from openai_compatible_server.backends import (",
                "    BaseModelBackend, OCSGenerationRequest, OCSGenerationResult",
                ")",
                "",
                "class FileBackend(BaseModelBackend):",
                "    def load_model(self) -> Any:",
                "        return {'ready': True}",
                "",
                "    def generate(self, request: OCSGenerationRequest):",
                "        return [",
                "            OCSGenerationResult(content='file') for _ in range(request.n)",
                "        ]",
            ]
        ),
        encoding="utf-8",
    )

    backend = create_model_backend(_settings(tmp_path, f"{backend_file}:FileBackend"))

    assert type(backend).__name__ == "FileBackend"
    assert backend.model_id == "factory-test-model"


def test_factory_loads_sibling_modules_for_backend_file(tmp_path: Path) -> None:
    (tmp_path / "modeling_qwen3vl.py").write_text(
        "MODEL_NAME = 'qwen3vl-sibling'\n",
        encoding="utf-8",
    )
    backend_file = tmp_path / "qwen3vl_backend.py"
    backend_file.write_text(
        "\n".join(
            [
                "from typing import Any",
                "from modeling_qwen3vl import MODEL_NAME",
                "from openai_compatible_server.backends import (",
                "    BaseModelBackend, OCSGenerationRequest, OCSGenerationResult",
                ")",
                "",
                "class Qwen3VLBackend(BaseModelBackend):",
                "    def load_model(self) -> Any:",
                "        return {'ready': True}",
                "",
                "    def generate(self, request: OCSGenerationRequest):",
                "        return [OCSGenerationResult(content=MODEL_NAME)]",
            ]
        ),
        encoding="utf-8",
    )

    backend = create_model_backend(_settings(tmp_path, f"{backend_file}:Qwen3VLBackend"))

    assert type(backend).__name__ == "Qwen3VLBackend"


def test_factory_error_explains_supported_backend_paths(tmp_path: Path) -> None:
    with pytest.raises(ModuleNotFoundError, match=r"\.py file path"):
        create_model_backend(_settings(tmp_path, "missing_package.backend:Backend"))


async def test_custom_backend_stream_generate_yields_model_chunks(tmp_path: Path) -> None:
    backend = create_model_backend(
        _settings(
            tmp_path,
            "openai_compatible_server.backends.custom:CustomModelBackend",
        )
    )
    await backend.load()
    request = OCSGenerationRequest(
        model="factory-test-model",
        messages=[{"role": "user", "content": "hello"}],
        sampling_params={},
    )

    chunks = [chunk async for chunk in backend.stream_generate(request)]

    assert "".join(chunk.content or "" for chunk in chunks) == "Custom model stream response."
    assert chunks[-1].finish_reason == "stop"
