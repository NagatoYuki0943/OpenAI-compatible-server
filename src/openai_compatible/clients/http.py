from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

import aiohttp
import httpx
import requests

from openai_compatible.clients.common import build_request


def iter_sse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


async def iter_async_sse_lines(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    async for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


def requests_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 120,
) -> Iterator[dict[str, Any]]:
    with requests.post(
        url,
        json=data,
        headers=headers,
        timeout=timeout,
        stream=data.get("stream", False),
    ) as response:
        response.raise_for_status()
        if not data.get("stream"):
            yield response.json()
            return
        yield from iter_sse_lines(response.iter_lines(decode_unicode=True))


def httpx_sync_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 120,
) -> Iterator[dict[str, Any]]:
    with httpx.Client(timeout=timeout) as client:
        if not data.get("stream"):
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            yield response.json()
            return
        with client.stream("POST", url, json=data, headers=headers) as response:
            response.raise_for_status()
            yield from iter_sse_lines(response.iter_lines())


async def httpx_async_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 120,
) -> AsyncIterator[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        if not data.get("stream"):
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            yield response.json()
            return
        async with client.stream("POST", url, json=data, headers=headers) as response:
            response.raise_for_status()
            async for event in iter_async_sse_lines(response.aiter_lines()):
                yield event


async def aiohttp_async_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 120,
) -> AsyncIterator[dict[str, Any]]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with (
        aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session,
        session.post(url, json=data) as response,
    ):
        response.raise_for_status()
        if not data.get("stream"):
            yield await response.json()
            return
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                for event in iter_sse_lines([line]):
                    yield event
                if line.strip() == "data: [DONE]":
                    return


async def _run_async(
    implementation: str,
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> None:
    function = aiohttp_async_chat if implementation == "aiohttp" else httpx_async_chat
    async for event in function(url, data, headers, timeout):
        print(json.dumps(event, ensure_ascii=False, indent=2))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the API using a generic HTTP client.")
    parser.add_argument(
        "--client",
        choices=["requests", "httpx", "httpx-async", "aiohttp"],
        default="httpx-async",
        help="HTTP implementation (default: httpx-async)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("CHAT_URL", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "I AM AN API KEY"))
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "demo-multimodal-model"))
    parser.add_argument("--prompt", default="请描述收到的文本和多媒体输入。")
    parser.add_argument("--image", help="Image URL, data URI, or local path")
    parser.add_argument("--video", help="Video URL, data URI, or local path")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--min-tokens", type=int, default=1)
    parser.add_argument("--thinking-token-budget", type=int, default=256)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-n", type=int, default=1, help="Number of choices")
    parser.add_argument("--stream", action="store_true", help="Use SSE streaming")
    return parser


def main() -> None:
    args = create_parser().parse_args()

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    data = build_request(
        model=args.model,
        image=args.image,
        video=args.video,
        stream=args.stream,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        min_tokens=args.min_tokens,
        thinking_token_budget=args.thinking_token_budget,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        n=args.n,
    )

    if args.client == "requests":
        events = requests_chat(args.url, data, headers, args.timeout)
    elif args.client == "httpx":
        events = httpx_sync_chat(args.url, data, headers, args.timeout)
    else:
        asyncio.run(_run_async(args.client, args.url, data, headers, args.timeout))
        return
    for event in events:
        print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
