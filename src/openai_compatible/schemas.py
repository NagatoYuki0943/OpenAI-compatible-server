from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["developer", "system", "user", "assistant", "tool", "function"]


class OpenAIModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ImageURL(OpenAIModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class AudioData(OpenAIModel):
    data: str
    format: Literal["wav", "mp3"]


class VideoURL(OpenAIModel):
    """Non-standard URL/data-URI video extension."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"
    fps: float | None = Field(default=None, gt=0)


class VideoData(OpenAIModel):
    """Non-standard base64 video extension."""

    data: str
    format: str = "mp4"
    fps: float | None = Field(default=None, gt=0)


class ContentPart(OpenAIModel):
    type: Literal[
        "text",
        "image_url",
        "input_audio",
        "video_url",
        "input_video",
        "file",
        "refusal",
    ]
    text: str | None = None
    image_url: ImageURL | None = None
    input_audio: AudioData | None = None
    video_url: VideoURL | None = None
    input_video: VideoData | None = None
    file: dict[str, Any] | None = None
    refusal: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> ContentPart:
        payloads = {
            "text": self.text,
            "image_url": self.image_url,
            "input_audio": self.input_audio,
            "video_url": self.video_url,
            "input_video": self.input_video,
            "file": self.file,
            "refusal": self.refusal,
        }
        if payloads[self.type] is None:
            raise ValueError(f"{self.type!r} content part is missing its payload")
        return self


class FunctionCall(OpenAIModel):
    name: str
    arguments: str


class ToolCall(OpenAIModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(OpenAIModel):
    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None

    @model_validator(mode="after")
    def validate_message(self) -> ChatMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.content is None and not self.tool_calls and not self.function_call:
            raise ValueError("a message needs content, tool_calls, or function_call")
        return self


class StreamOptions(OpenAIModel):
    include_usage: bool = False
    include_obfuscation: bool = False


class ChatRequest(OpenAIModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1, le=128)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    logit_bias: dict[str, float] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    modalities: list[Literal["text", "audio"]] | None = None
    audio: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None
    user: str | None = None
    service_tier: str | None = None

    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    min_tokens: int | None = Field(default=None, ge=0)
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    allowed_token_ids: list[int] | None = Field(default=None, min_length=1)
    include_stop_str_in_output: bool | None = None
    ignore_eos: bool | None = None
    skip_special_tokens: bool | None = None
    spaces_between_special_tokens: bool | None = None
    prompt_logprobs: int | None = Field(default=None, ge=-1)
    thinking_token_budget: int | None = Field(default=None, ge=0)
    structured_outputs: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_request(self) -> ChatRequest:
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options can only be used with stream=true")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        if self.modalities and "audio" in self.modalities and self.audio is None:
            raise ValueError("audio output parameters are required for audio modality")
        max_output = self.max_completion_tokens or self.max_tokens
        if max_output is not None and self.min_tokens is not None and self.min_tokens > max_output:
            raise ValueError("min_tokens cannot exceed the maximum output token count")
        return self

    def sampling_params(self) -> dict[str, Any]:
        names = {
            "n",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "stop",
            "stop_token_ids",
            "bad_words",
            "allowed_token_ids",
            "include_stop_str_in_output",
            "ignore_eos",
            "min_tokens",
            "logprobs",
            "prompt_logprobs",
            "logit_bias",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "thinking_token_budget",
            "structured_outputs",
        }
        data = self.model_dump()
        result = {name: data[name] for name in names if data[name] is not None}
        max_tokens = self.max_completion_tokens or self.max_tokens
        if max_tokens is not None:
            result["max_tokens"] = max_tokens
        return result
