"""Runnable custom backend example included in the installed package."""

from typing import Any

from openai_compatible.backends.base import (
    BaseModelBackend,
    GenerationRequest,
    GenerationResult,
    ModelMetadata,
    ReasoningMetadata,
)


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
        return {"ready": True}

    def infer(self, request: GenerationRequest) -> list[GenerationResult]:
        # Convert request.messages and pass request.sampling_params to your model.
        return [
            GenerationResult(
                content=f"Custom model response {index + 1}",
                reasoning_content="Optional reasoning output",
            )
            for index in range(request.n)
        ]

    def unload_model(self) -> None:
        # Release GPU memory or other external resources here.
        return None
