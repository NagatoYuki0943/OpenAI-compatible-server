from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from openai_compatible_server.backends import BaseModelBackend, create_model_backend
from openai_compatible_server.config import Settings
from openai_compatible_server.logging import configure_logging
from openai_compatible_server.schemas import ChatRequest
from openai_compatible_server.services import CompletionService
from openai_compatible_server.services.completions import text_and_media


def create_app(
    settings: Settings | None = None,
    backend: BaseModelBackend | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings)
    backend = backend or create_model_backend(settings)
    service = CompletionService(backend)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Server starting | env_file={} | log_dir={} | level={} | rotation={} | retention={}",
            settings.env_file,
            settings.log_dir,
            settings.log_level,
            settings.log_rotation,
            settings.log_retention,
        )
        await backend.load()
        try:
            yield
        finally:
            await backend.unload()
            logger.info("Server stopping")
            await logger.complete()

    app = FastAPI(
        title="OpenAI-compatible multimodal server",
        version="0.1.0",
        description="Chat Completions-compatible API with common vLLM extensions.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_backend = backend
    app.state.completion_service = service

    async def require_api_key(
        authorization: str | None = Header(default=None),
    ) -> None:
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            logger.warning("API authentication failed")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )

    @app.middleware("http")
    async def request_logging(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"
        with logger.contextualize(request_id=request_id):
            logger.info(
                "Request started | method={} | path={} | client={}",
                request.method,
                request.url.path,
                client_host,
            )
            try:
                response = await call_next(request)
            except Exception:
                logger.exception(
                    "Request failed | method={} | path={} | elapsed_ms={:.2f}",
                    request.method,
                    request.url.path,
                    (time.perf_counter() - started_at) * 1000,
                )
                raise
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "Request finished | method={} | path={} | status={} | elapsed_ms={:.2f}",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started_at) * 1000,
            )
            return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning(
            "HTTP error | method={} | path={} | status={}",
            request.method,
            request.url.path,
            exc.status_code,
        )
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            body = exc.detail
        else:
            body = _error_body(str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(
            "Validation error | method={} | path={} | errors={}",
            request.method,
            request.url.path,
            len(exc.errors()),
        )
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "Request validation failed",
                code="validation_error",
                details=exc.errors(),
            ),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": backend.is_loaded,
            "model": backend.model_id,
        }

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [backend.model_card()],
        }

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_api_key)],
        response_model=None,
    )
    async def chat(chat_request: ChatRequest, request: Request) -> JSONResponse | StreamingResponse:
        prompt_text, media = text_and_media(chat_request.messages)
        logger.info(
            "Chat requested | model={} | stream={} | messages={} | text_chars={} "
            "| media={} | n={} | max_tokens={} | temperature={} | top_p={} | top_k={}",
            chat_request.model,
            chat_request.stream,
            len(chat_request.messages),
            len(prompt_text),
            media or "none",
            chat_request.n,
            chat_request.max_completion_tokens or chat_request.max_tokens,
            chat_request.temperature,
            chat_request.top_p,
            chat_request.top_k,
        )
        logger.debug("Sampling parameters | params={}", chat_request.sampling_params())
        logger.info(
            "Chat payload | messages={} | sampling_params={}",
            [message.model_dump(exclude_none=True) for message in chat_request.messages],
            chat_request.sampling_params(),
        )
        request_id = request.state.request_id
        if chat_request.stream:
            return StreamingResponse(
                service.stream(chat_request, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        completion = await service.complete(chat_request, request_id)
        logger.info(
            "Chat completed | completion_id={} | model={} | usage={}",
            completion["id"],
            chat_request.model,
            completion["usage"],
        )
        return JSONResponse(content=completion)

    return app


def _error_body(
    message: str,
    *,
    code: str | None = None,
    details: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": message,
        "type": "invalid_request_error",
        "param": None,
        "code": code,
    }
    if details is not None:
        error["details"] = details
    return {"error": error}
