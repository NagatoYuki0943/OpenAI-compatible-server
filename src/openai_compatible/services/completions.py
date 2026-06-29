from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from openai_compatible.backends import (
    BaseModelBackend,
    GenerationRequest,
    GenerationResult,
)
from openai_compatible.schemas import ChatMessage, ChatRequest


class CompletionService:
    def __init__(self, backend: BaseModelBackend) -> None:
        self.backend = backend

    async def complete(self, request: ChatRequest, request_id: str | None = None) -> dict[str, Any]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        backend_request = self.backend.with_generation_defaults(
            _backend_request(request, request_id)
        )
        results = await self.backend.infer(backend_request)
        choices = []
        for index, result in enumerate(results):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": result.content,
                "refusal": None,
                "reasoning_content": result.reasoning_content,
            }
            if result.tool_calls is not None:
                message["tool_calls"] = result.tool_calls
            message.update(result.extra)
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "logprobs": result.logprobs,
                    "finish_reason": result.finish_reason,
                }
            )
        completion = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": choices,
            "usage": _usage_for(request, results),
            "system_fingerprint": "fp_demo_openai_compatible",
            "service_tier": request.service_tier,
        }
        logger.debug(
            "Completion built | completion_id={} | choices={} | usage={}",
            completion_id,
            len(choices),
            completion["usage"],
        )
        return completion

    async def stream(
        self, request: ChatRequest, request_id: str | None = None
    ) -> AsyncIterator[str]:
        started_at = time.perf_counter()
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        common = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "fp_demo_openai_compatible",
            "service_tier": request.service_tier,
        }
        outputs = ["" for _ in range(request.n)]
        reasoning_outputs = ["" for _ in range(request.n)]
        finished = [False for _ in range(request.n)]
        backend_usage: dict[str, Any] | None = None
        try:
            for index in range(request.n):
                yield _sse(
                    {
                        **common,
                        "choices": [
                            {
                                "index": index,
                                "delta": {"role": "assistant", "content": ""},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                )

            backend_request = self.backend.with_generation_defaults(
                _backend_request(request, request_id)
            )
            async for chunk in self.backend.stream_generate(backend_request):
                if not 0 <= chunk.index < request.n:
                    raise RuntimeError(f"Backend returned invalid choice index {chunk.index}")
                delta: dict[str, Any] = dict(chunk.extra)
                if chunk.content is not None:
                    delta["content"] = chunk.content
                    outputs[chunk.index] += chunk.content
                if chunk.reasoning_content is not None:
                    delta["reasoning_content"] = chunk.reasoning_content
                    reasoning_outputs[chunk.index] += chunk.reasoning_content
                if chunk.tool_calls is not None:
                    delta["tool_calls"] = chunk.tool_calls
                if chunk.finish_reason is not None:
                    finished[chunk.index] = True
                if chunk.usage is not None:
                    backend_usage = chunk.usage
                yield _sse(
                    {
                        **common,
                        "choices": [
                            {
                                "index": chunk.index,
                                "delta": delta,
                                "logprobs": chunk.logprobs,
                                "finish_reason": chunk.finish_reason,
                            }
                        ],
                    }
                )

            for index, is_finished in enumerate(finished):
                if not is_finished:
                    yield _sse(
                        {
                            **common,
                            "choices": [
                                {
                                    "index": index,
                                    "delta": {},
                                    "logprobs": None,
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )

            if request.stream_options and request.stream_options.include_usage:
                results = [
                    GenerationResult(
                        content=outputs[index],
                        reasoning_content=reasoning_outputs[index],
                    )
                    for index in range(request.n)
                ]
                yield _sse(
                    {
                        **common,
                        "choices": [],
                        "usage": backend_usage or _usage_for(request, results),
                    }
                )
            yield _sse("[DONE]")
        except asyncio.CancelledError:
            logger.warning("Stream cancelled by client | completion_id={}", completion_id)
            raise
        except Exception:
            logger.exception("Stream failed | completion_id={}", completion_id)
            raise
        finally:
            logger.info(
                "Stream finished | completion_id={} | elapsed_ms={:.2f}",
                completion_id,
                (time.perf_counter() - started_at) * 1000,
            )


def text_and_media(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    texts: list[str] = []
    media: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            texts.append(message.content)
            continue
        for part in message.content or []:
            if part.type == "text" and part.text:
                texts.append(part.text)
            elif part.type in {
                "image_url",
                "input_audio",
                "video_url",
                "input_video",
                "file",
            }:
                media.append(part.type)
    return "\n".join(texts), media


def _backend_request(request: ChatRequest, request_id: str | None) -> GenerationRequest:
    return GenerationRequest(
        model=request.model,
        messages=[message.model_dump(exclude_none=True) for message in request.messages],
        sampling_params=request.sampling_params(),
        n=request.n,
        request_id=request_id,
        metadata={
            "user": request.user,
            "service_tier": request.service_tier,
            **(request.metadata or {}),
        },
    )


def _usage_for(request: ChatRequest, results: list[GenerationResult]) -> dict[str, Any]:
    prompt_text, _ = text_and_media(request.messages)
    prompt_tokens = next(
        (item.prompt_tokens for item in results if item.prompt_tokens is not None),
        _estimate_tokens(prompt_text),
    )
    reasoning_tokens = sum(
        item.reasoning_tokens
        if item.reasoning_tokens is not None
        else _estimate_tokens(item.reasoning_content or "")
        for item in results
    )
    completion_tokens = sum(
        item.completion_tokens
        if item.completion_tokens is not None
        else _estimate_tokens(item.content) + _estimate_tokens(item.reasoning_content or "")
        for item in results
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _sse(data: dict[str, Any] | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"
