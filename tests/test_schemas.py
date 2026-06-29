import pytest
from pydantic import ValidationError

from openai_compatible_server.schemas import ChatRequest


def test_sampling_params_support_openai_and_vllm_extensions() -> None:
    request = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=64,
        temperature=0.5,
        top_k=20,
        min_p=0.1,
        repetition_penalty=1.0,
        stop_token_ids=[1, 2],
    )

    params = request.sampling_params()

    assert params["max_tokens"] == 64
    assert params["temperature"] == 0.5
    assert params["top_k"] == 20
    assert params["min_p"] == 0.1
    assert params["stop_token_ids"] == [1, 2]


def test_model_specific_sampling_params_are_omitted_when_unspecified() -> None:
    request = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    params = request.sampling_params()

    for name in (
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "min_tokens",
        "stop_token_ids",
        "bad_words",
        "allowed_token_ids",
        "include_stop_str_in_output",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    ):
        assert name not in params


def test_sampling_params_preserve_explicit_zero_and_false_values() -> None:
    request = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        min_tokens=0,
        include_stop_str_in_output=False,
        ignore_eos=False,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    params = request.sampling_params()

    assert params["min_tokens"] == 0
    assert params["include_stop_str_in_output"] is False
    assert params["ignore_eos"] is False
    assert params["skip_special_tokens"] is False
    assert params["spaces_between_special_tokens"] is False


def test_multimodal_and_video_extension_validate() -> None:
    request = ChatRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4", "fps": 1},
                    },
                ],
            }
        ],
    )

    assert len(request.messages[0].content) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "test-model",
            "messages": [{"role": "tool", "content": "result"}],
        },
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream_options": {"include_usage": True},
        },
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "top_logprobs": 2,
        },
    ],
)
def test_invalid_requests_are_rejected(payload) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)
