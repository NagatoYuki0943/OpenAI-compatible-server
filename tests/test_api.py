from __future__ import annotations

import json

from fastapi.testclient import TestClient

from openai_compatible_server.api import create_app


def test_health_models_and_backend_lifecycle(settings, backend) -> None:
    app = create_app(settings, backend)

    with TestClient(app) as client:
        health = client.get("/health", headers={"X-Request-ID": "health-id"})
        models = client.get("/v1/models")

        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert health.headers["X-Request-ID"] == "health-id"
        model = models.json()["data"][0]
        assert model["id"] == "test-model"
        assert model["capabilities"] == [
            "reasoning",
            "image-recognition",
            "function-call",
        ]
        assert model["input_modalities"] == ["text", "image"]
        assert model["inputModalities"] == ["text", "image"]
        assert model["supports_streaming"] is True
        assert model["supportsStreaming"] is True
        assert model["reasoning"]["supportedEfforts"] == ["low", "medium", "high"]
        assert model["reasoning"]["defaultEffort"] == "medium"
        assert model["context_window"] == 8192
        assert model["contextWindow"] == 8192
        assert backend.load_count == 1

    assert backend.unload_count == 1


def test_non_streaming_multimodal_completion(settings, backend) -> None:
    app = create_app(settings, backend)
    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ],
        "n": 2,
        "top_k": 7,
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "chat-id"},
            json=payload,
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["choices"]) == 2
    assert body["choices"][0]["message"]["reasoning_content"] == "reason-0"
    assert body["usage"]["completion_tokens"] == 12
    assert backend.requests[0].sampling_params["top_k"] == 7
    assert backend.requests[0].sampling_params["temperature"] == 0.8
    assert backend.requests[0].sampling_params["max_tokens"] == 256
    assert backend.requests[0].sampling_params["repetition_penalty"] == 1.0
    assert backend.requests[0].sampling_params["min_tokens"] == 2
    assert backend.requests[0].sampling_params["stop_token_ids"] == [99]
    assert backend.requests[0].request_id == "chat-id"


def test_streaming_completion_reassembles_content(settings, backend) -> None:
    app = create_app(settings, backend)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    answer = ""
    reasoning = ""
    usage = None
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        if not event["choices"]:
            usage = event["usage"]
            continue
        delta = event["choices"][0]["delta"]
        answer += delta.get("content", "")
        reasoning += delta.get("reasoning_content", "")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert answer == "answer-0"
    assert reasoning == "reason-0"
    assert usage["completion_tokens"] == 6
    assert "data: [DONE]" in response.text


def test_authentication_and_validation_errors(tmp_path, backend) -> None:
    from openai_compatible_server.config import Settings

    app = create_app(
        Settings(api_key="secret", model_id="test-model", log_dir=tmp_path / "logs"),
        backend,
    )
    with TestClient(app) as client:
        unauthorized = client.get("/v1/models")
        invalid = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "test-model", "messages": []},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "invalid_api_key"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
