from __future__ import annotations

import argparse
import os
from typing import Any

from openai import OpenAI

from openai_compatible.clients.common import build_messages


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the API using the OpenAI SDK.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
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

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    common: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(args.image, args.video, args.prompt),
        "max_completion_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "n": args.n,
        "stream": args.stream,
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "min_tokens": args.min_tokens,
            "thinking_token_budget": args.thinking_token_budget,
        },
    }

    if not args.stream:
        response = client.chat.completions.create(**common)
        print(response.model_dump_json(indent=2))
        return

    stream = client.chat.completions.create(
        **common,
        stream_options={"include_usage": True},
    )
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    for chunk in stream:
        if chunk.usage:
            print(f"\nusage: {chunk.usage}")
        for choice in chunk.choices:
            reasoning = getattr(choice.delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            if choice.delta.content:
                answer_parts.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)
    print(f"\n\nreasoning: {''.join(reasoning_parts)}")
    print(f"answer: {''.join(answer_parts)}")


if __name__ == "__main__":
    main()
