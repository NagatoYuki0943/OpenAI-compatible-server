"""Runnable custom backend example included in the installed package."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from openai_compatible_server.backends.base import (
    BaseModelBackend,
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelMetadata,
    ReasoningMetadata,
)

_STREAM_END = object()


def _next_stream_item(iterator: Iterator[str]) -> str | object:
    return next(iterator, _STREAM_END)


class ExampleStringModel:
    """Tiny stand-in for a model whose generate method supports stream=True."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        **sampling_params: Any,
    ) -> str | Iterator[str]:
        content = (
            "Custom model response. "
            f"messages={len(messages)}, "
            f"temperature={sampling_params.get('temperature')}"
        )
        if not stream:
            return content
        return iter(("Custom ", "model ", "stream ", "response."))


class CustomModelBackend(BaseModelBackend):
    generation_defaults = {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "stop_token_ids": [],
        "bad_words": [],
        "include_stop_str_in_output": False,
        "ignore_eos": False,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }
    model_metadata = ModelMetadata(
        name="My Custom Model",
        description="Example custom multimodal reasoning model.",
        capabilities=("reasoning", "image-recognition", "function-call"),
        input_modalities=("text", "image"),
        output_modalities=("text",),
        supports_streaming=True,
        reasoning=ReasoningMetadata(
            supported_efforts=("low", "medium", "high"),
            default_effort="medium",
            min_thinking_tokens=0,
            max_thinking_tokens=8192,
        ),
        context_window=32_768,
        max_output_tokens=4096,
    )

    def load_model(self) -> Any:
        # Replace this with tokenizer/model/pipeline loading.
        return ExampleStringModel()

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for _ in range(request.n):
            content = self.model.generate(
                request.messages,
                stream=False,
                **request.sampling_params,
            )
            if not isinstance(content, str):
                raise TypeError("model.generate(stream=False) must return str")
            results.append(
                GenerationResult(
                    content=content,
                    reasoning_content="Optional reasoning output",
                )
            )
        return results

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        self._ensure_loaded()
        request = self.with_generation_defaults(request)
        async with self._inference_semaphore:
            for index in range(request.n):
                stream = await asyncio.to_thread(
                    self.model.generate,
                    request.messages,
                    stream=True,
                    **request.sampling_params,
                )
                if isinstance(stream, str):
                    raise TypeError("model.generate(stream=True) must return an iterator of str")

                iterator = iter(stream)
                while True:
                    item = await asyncio.to_thread(_next_stream_item, iterator)
                    if item is _STREAM_END:
                        break
                    if not isinstance(item, str):
                        raise TypeError("model.generate(stream=True) must yield str")
                    yield GenerationChunk(index=index, content=item)

                yield GenerationChunk(index=index, finish_reason="stop")

    def unload_model(self) -> None:
        # Release GPU memory or other external resources here.
        return None
