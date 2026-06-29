from __future__ import annotations

from typing import Any

from openai_compatible_server.backends.base import (
    BaseModelBackend,
    OCSGenerationRequest,
    OCSGenerationResult,
    OCSModelMetadata,
    OCSReasoningMetadata,
)


class DemoModelBackend(BaseModelBackend):
    generation_defaults = {
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
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
    model_metadata = OCSModelMetadata(
        name="Demo Multimodal Model",
        description="Demonstration backend for the OpenAI-compatible server.",
        capabilities=(
            "reasoning",
            "image-recognition",
            "audio-recognition",
            "video-recognition",
            "file-input",
        ),
        input_modalities=("text", "image", "audio", "video"),
        output_modalities=("text",),
        supports_streaming=True,
        reasoning=OCSReasoningMetadata(
            type="effort",
            supported_efforts=("none", "minimal", "low", "medium", "high"),
            default_effort="medium",
            min_thinking_tokens=0,
        ),
        context_window=128_000,
        max_output_tokens=16_384,
    )

    def load_model(self) -> dict[str, str]:
        return {"model_id": self.model_id, "type": "demo"}

    def generate(self, request: OCSGenerationRequest) -> list[OCSGenerationResult]:
        text, media = _extract_inputs(request.messages)
        media_text = f" Media received: {', '.join(media)}." if media else ""
        results = []
        for index in range(request.n):
            reasoning = (
                f"Validated {len(request.messages)} message(s), extracted "
                f"{len(text)} text character(s), and applied sampling parameters."
            )
            content = (
                f"Demo completion {index + 1}: received your request."
                f"{media_text} Last text: {text[-200:] or '(none)'}"
            )
            results.append(
                OCSGenerationResult(
                    content=content,
                    reasoning_content=reasoning,
                    prompt_tokens=_estimate_tokens(text),
                    completion_tokens=_estimate_tokens(content) + _estimate_tokens(reasoning),
                    reasoning_tokens=_estimate_tokens(reasoning),
                )
            )
        return results


def _extract_inputs(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    texts: list[str] = []
    media: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content or []:
            part_type = part.get("type")
            if part_type == "text" and part.get("text"):
                texts.append(part["text"])
            elif part_type in {
                "image_url",
                "input_audio",
                "video_url",
                "input_video",
                "file",
            }:
                media.append(part_type)
    return "\n".join(texts), media


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
