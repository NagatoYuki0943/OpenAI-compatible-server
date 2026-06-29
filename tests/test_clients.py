from openai_compatible_server.clients.common import build_request
from openai_compatible_server.clients.http import create_parser as create_http_parser
from openai_compatible_server.clients.http import iter_sse_lines
from openai_compatible_server.clients.openai_sdk import create_parser as create_sdk_parser


def test_sse_parser_stops_at_done() -> None:
    lines = [
        'data: {"choices": [{"index": 0}]}',
        "",
        "data: [DONE]",
        'data: {"ignored": true}',
    ]

    assert list(iter_sse_lines(lines)) == [{"choices": [{"index": 0}]}]


def test_request_builder_supports_local_media(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    request = build_request(
        model="test-model",
        image=str(image),
        video="https://example.com/video.mp4",
        stream=True,
    )
    content = request["messages"][1]["content"]

    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["video_url"]["url"] == "https://example.com/video.mp4"
    assert request["stream_options"]["include_usage"] is True


def test_http_client_parser_supports_generation_options() -> None:
    args = create_http_parser().parse_args(
        [
            "--client",
            "requests",
            "--url",
            "http://localhost/v1/chat/completions",
            "--model",
            "model-a",
            "--prompt",
            "hello",
            "--top-k",
            "20",
            "--max-tokens",
            "64",
            "--frequency-penalty",
            "0.2",
            "--presence-penalty",
            "0.3",
            "--min-tokens",
            "4",
            "--thinking-token-budget",
            "128",
            "--reasoning-effort",
            "high",
            "--stream",
        ]
    )

    assert args.client == "requests"
    assert args.model == "model-a"
    assert args.prompt == "hello"
    assert args.top_k == 20
    assert args.max_tokens == 64
    assert args.frequency_penalty == 0.2
    assert args.presence_penalty == 0.3
    assert args.min_tokens == 4
    assert args.thinking_token_budget == 128
    assert args.reasoning_effort == "high"
    assert args.stream is True


def test_openai_sdk_parser_uses_clear_base_url_option() -> None:
    args = create_sdk_parser().parse_args(
        [
            "--base-url",
            "http://localhost/v1",
            "--api-key",
            "secret",
            "--model",
            "model-b",
            "-n",
            "2",
            "--frequency-penalty",
            "-0.1",
            "--presence-penalty",
            "0.4",
            "--min-tokens",
            "3",
            "--thinking-token-budget",
            "64",
            "--reasoning-effort",
            "medium",
        ]
    )

    assert args.base_url == "http://localhost/v1"
    assert args.api_key == "secret"
    assert args.model == "model-b"
    assert args.n == 2
    assert args.frequency_penalty == -0.1
    assert args.presence_penalty == 0.4
    assert args.min_tokens == 3
    assert args.thinking_token_budget == 64
    assert args.reasoning_effort == "medium"
