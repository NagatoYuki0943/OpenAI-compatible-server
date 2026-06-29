from __future__ import annotations

from typing import Any

import pytest

from openai_compatible_server.backends import (
    BaseModelBackend,
    OCSGenerationRequest,
    OCSGenerationResult,
    OCSModelMetadata,
    OCSReasoningMetadata,
)
from openai_compatible_server.config import Settings


class StubBackend(BaseModelBackend):
    generation_defaults = {
        "max_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.0,
        "min_tokens": 2,
        "stop_token_ids": [99],
        "bad_words": ["blocked"],
        "include_stop_str_in_output": True,
        "ignore_eos": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    model_metadata = OCSModelMetadata(
        name="Test Model",
        capabilities=("reasoning", "image-recognition", "function-call"),
        input_modalities=("text", "image"),
        output_modalities=("text",),
        reasoning=OCSReasoningMetadata(
            supported_efforts=("low", "medium", "high"),
            default_effort="medium",
        ),
        context_window=8192,
        max_output_tokens=1024,
    )

    def __init__(self, model_id: str = "test-model") -> None:
        super().__init__(model_id, max_concurrency=2, stream_chunk_size=5)
        self.load_count = 0
        self.unload_count = 0
        self.requests: list[OCSGenerationRequest] = []

    def load_model(self) -> dict[str, Any]:
        self.load_count += 1
        return {"ready": True}

    def generate(self, request: OCSGenerationRequest) -> list[OCSGenerationResult]:
        self.requests.append(request)
        return [
            OCSGenerationResult(
                content=f"answer-{index}",
                reasoning_content=f"reason-{index}",
                prompt_tokens=3,
                completion_tokens=6,
                reasoning_tokens=2,
            )
            for index in range(request.n)
        ]

    def unload_model(self) -> None:
        self.unload_count += 1


@pytest.fixture
def backend() -> StubBackend:
    return StubBackend()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        api_key=None,
        model_id="test-model",
        log_dir=tmp_path / "logs",
        log_level="DEBUG",
    )
