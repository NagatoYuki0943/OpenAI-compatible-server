from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def as_data_url(path: str) -> str:
    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_media_url(value: str) -> str:
    if "://" in value or value.startswith("data:"):
        return value
    return as_data_url(value)


def build_messages(
    image: str | None = None,
    video: str | None = None,
    prompt: str = "请描述收到的文本和多媒体输入。",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": resolve_media_url(image), "detail": "auto"},
            }
        )
    if video:
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": resolve_media_url(video), "fps": 1.0},
            }
        )
    return [
        {"role": "developer", "content": "你是一个支持多模态输入的助手。"},
        {"role": "user", "content": content},
    ]


def build_request(
    *,
    model: str,
    image: str | None = None,
    video: str | None = None,
    stream: bool = False,
    prompt: str = "请描述收到的文本和多媒体输入。",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    min_p: float = 0.05,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    min_tokens: int = 1,
    thinking_token_budget: int = 256,
    reasoning_effort: str = "medium",
    seed: int | None = 42,
    n: int = 1,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_messages(image, video, prompt),
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "n": n,
        "min_tokens": min_tokens,
        "thinking_token_budget": thinking_token_budget,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    }
