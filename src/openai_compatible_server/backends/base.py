from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal

from loguru import logger

ModelCapability = Literal[
    "function-call",
    "reasoning",
    "image-recognition",
    "image-generation",
    "audio-recognition",
    "audio-generation",
    "embedding",
    "rerank",
    "audio-transcript",
    "video-recognition",
    "video-generation",
    "structured-output",
    "file-input",
    "web-search",
    "code-execution",
    "file-search",
    "computer-use",
]
Modality = Literal["text", "image", "audio", "video", "vector"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "auto"]


@dataclass(frozen=True, slots=True)
class OCSReasoningMetadata:
    type: str = "effort"
    supported_efforts: tuple[ReasoningEffort, ...] = ()
    default_effort: ReasoningEffort | None = None
    min_thinking_tokens: int | None = None
    max_thinking_tokens: int | None = None
    interleaved: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "supportedEfforts": list(self.supported_efforts),
            "interleaved": self.interleaved,
        }
        if self.default_effort is not None:
            result["defaultEffort"] = self.default_effort
        if self.min_thinking_tokens is not None or self.max_thinking_tokens is not None:
            limits: dict[str, int] = {}
            if self.min_thinking_tokens is not None:
                limits["min"] = self.min_thinking_tokens
            if self.max_thinking_tokens is not None:
                limits["max"] = self.max_thinking_tokens
            result["thinkingTokenLimits"] = limits
        return result


@dataclass(frozen=True, slots=True)
class OCSModelMetadata:
    name: str | None = None
    description: str | None = None
    owned_by: str = "local"
    created: int = 0
    capabilities: tuple[ModelCapability, ...] = ()
    input_modalities: tuple[Modality, ...] = ("text",)
    output_modalities: tuple[Modality, ...] = ("text",)
    supports_streaming: bool = True
    reasoning: OCSReasoningMetadata | None = None
    context_window: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_model_card(self, model_id: str) -> dict[str, Any]:
        card: dict[str, Any] = {
            "id": model_id,
            "object": "model",
            "created": self.created,
            "owned_by": self.owned_by,
            "name": self.name or model_id,
            "capabilities": list(self.capabilities),
            # Include both naming styles because OpenAI-compatible clients vary.
            "input_modalities": list(self.input_modalities),
            "inputModalities": list(self.input_modalities),
            "output_modalities": list(self.output_modalities),
            "outputModalities": list(self.output_modalities),
            "supports_streaming": self.supports_streaming,
            "supportsStreaming": self.supports_streaming,
        }
        if self.description is not None:
            card["description"] = self.description
        if self.reasoning is not None:
            card["reasoning"] = self.reasoning.to_dict()
        for snake_name, camel_name, value in (
            ("context_window", "contextWindow", self.context_window),
            ("max_input_tokens", "maxInputTokens", self.max_input_tokens),
            ("max_output_tokens", "maxOutputTokens", self.max_output_tokens),
        ):
            if value is not None:
                card[snake_name] = value
                card[camel_name] = value
        card.update(self.extra)
        return card


@dataclass(slots=True)
class OCSGenerationRequest:
    model: str
    messages: list[dict[str, Any]]
    sampling_params: dict[str, Any]
    n: int = 1
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCSGenerationResult:
    content: str = ""
    reasoning_content: str | None = None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    logprobs: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCSGenerationChunk:
    index: int
    content: str | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    logprobs: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModelBackend(ABC):
    model_metadata: OCSModelMetadata | None = None
    generation_defaults: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        model_id: str,
        *,
        max_concurrency: int = 1,
        stream_chunk_size: int = 12,
    ) -> None:
        self.model_id = model_id
        self.stream_chunk_size = stream_chunk_size
        self.model: Any = None
        self._loaded = False
        self._lifecycle_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_metadata(self) -> OCSModelMetadata:
        """Return metadata advertised by ``GET /v1/models``."""
        return self.model_metadata or OCSModelMetadata()

    def model_card(self) -> dict[str, Any]:
        return self.get_model_metadata().to_model_card(self.model_id)

    async def load(self) -> None:
        async with self._lifecycle_lock:
            if self._loaded:
                return
            started_at = time.perf_counter()
            logger.info("Loading model | backend={} | model={}", type(self).__name__, self.model_id)
            try:
                self.model = await asyncio.to_thread(self.load_model)
                self._loaded = True
            except Exception:
                logger.exception(
                    "Model loading failed | backend={} | model={}",
                    type(self).__name__,
                    self.model_id,
                )
                raise
            logger.info(
                "Model loaded | backend={} | model={} | elapsed_ms={:.2f}",
                type(self).__name__,
                self.model_id,
                (time.perf_counter() - started_at) * 1000,
            )

    async def unload(self) -> None:
        async with self._lifecycle_lock:
            if not self._loaded:
                return
            started_at = time.perf_counter()
            try:
                await asyncio.to_thread(self.unload_model)
            except Exception:
                logger.exception(
                    "Model cleanup failed | backend={} | model={}",
                    type(self).__name__,
                    self.model_id,
                )
                raise
            finally:
                self.model = None
                self._loaded = False
            logger.info(
                "Model unloaded | backend={} | model={} | elapsed_ms={:.2f}",
                type(self).__name__,
                self.model_id,
                (time.perf_counter() - started_at) * 1000,
            )

    async def infer(self, request: OCSGenerationRequest) -> list[OCSGenerationResult]:
        self._ensure_loaded()
        request = self.with_generation_defaults(request)
        started_at = time.perf_counter()
        logger.info(
            "Inference started | backend={} | model={} | choices={} | messages={}",
            type(self).__name__,
            request.model,
            request.n,
            len(request.messages),
        )
        try:
            async with self._inference_semaphore:
                results = await asyncio.to_thread(self.generate, request)
            if len(results) != request.n:
                raise RuntimeError(
                    f"Backend returned {len(results)} result(s), expected {request.n}"
                )
        except Exception:
            logger.exception(
                "Inference failed | backend={} | model={}",
                type(self).__name__,
                request.model,
            )
            raise
        logger.info(
            "Inference finished | backend={} | model={} | choices={} | elapsed_ms={:.2f}",
            type(self).__name__,
            request.model,
            len(results),
            (time.perf_counter() - started_at) * 1000,
        )
        return results

    def with_generation_defaults(self, request: OCSGenerationRequest) -> OCSGenerationRequest:
        """Return a request with model defaults overridden by explicit client parameters."""
        sampling_params = {
            **self.generation_defaults,
            **request.sampling_params,
        }
        min_tokens = sampling_params.get("min_tokens")
        max_tokens = sampling_params.get("max_tokens")
        if min_tokens is not None and max_tokens is not None and min_tokens > max_tokens:
            raise ValueError("min_tokens cannot exceed the maximum output token count")
        return replace(request, sampling_params=sampling_params)

    async def stream_generate(
        self, request: OCSGenerationRequest
    ) -> AsyncIterator[OCSGenerationChunk]:
        results = await self.infer(request)
        prompt_tokens = next(
            (item.prompt_tokens for item in results if item.prompt_tokens is not None),
            None,
        )
        completion_tokens = (
            sum(item.completion_tokens for item in results)
            if all(item.completion_tokens is not None for item in results)
            else None
        )
        reasoning_tokens = (
            sum(item.reasoning_tokens for item in results)
            if all(item.reasoning_tokens is not None for item in results)
            else None
        )
        for index, result in enumerate(results):
            if result.reasoning_content:
                for start in range(0, len(result.reasoning_content), self.stream_chunk_size):
                    yield OCSGenerationChunk(
                        index=index,
                        reasoning_content=result.reasoning_content[
                            start : start + self.stream_chunk_size
                        ],
                    )
            for start in range(0, len(result.content), self.stream_chunk_size):
                yield OCSGenerationChunk(
                    index=index,
                    content=result.content[start : start + self.stream_chunk_size],
                )
            chunk_prompt_tokens = None
            chunk_completion_tokens = None
            chunk_reasoning_tokens = None
            if (
                index == len(results) - 1
                and prompt_tokens is not None
                and completion_tokens is not None
                and reasoning_tokens is not None
            ):
                chunk_prompt_tokens = prompt_tokens
                chunk_completion_tokens = completion_tokens
                chunk_reasoning_tokens = reasoning_tokens
            yield OCSGenerationChunk(
                index=index,
                finish_reason=result.finish_reason,
                tool_calls=result.tool_calls,
                logprobs=result.logprobs,
                prompt_tokens=chunk_prompt_tokens,
                completion_tokens=chunk_completion_tokens,
                reasoning_tokens=chunk_reasoning_tokens,
            )

    @abstractmethod
    def load_model(self) -> Any:
        """Load tokenizer/model resources and return the model object."""

    @abstractmethod
    def generate(self, request: OCSGenerationRequest) -> list[OCSGenerationResult]:
        """Run synchronous generation in a worker thread."""

    def unload_model(self) -> None:
        """Release model resources. Override for GPU cleanup."""
        return None

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model backend is not loaded")
