from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from vidore_rag.api.engine import RetrievalEngine
from vidore_rag.api.models import (
    HealthResponse,
    PageResponse,
    QueryListResponse,
    RetrievalTraceResponse,
    RetrieveRequest,
    ValidationRequest,
    ValidationResponse,
)

LOGGER = logging.getLogger("vidore_rag.api")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = REPOSITORY_ROOT / "deployment/artifacts"


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests per minute must be positive")
        self._limit = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60
        with self._lock:
            requests = self._requests[client]
            while requests and requests[0] < cutoff:
                requests.popleft()
            if len(requests) >= self._limit:
                return False
            requests.append(now)
            return True


def create_app(
    *,
    engine: RetrievalEngine | None = None,
    artifact_dir: Path | None = None,
) -> FastAPI:
    resolved_artifact_dir = artifact_dir or Path(
        os.getenv("VIDORE_RAG_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_engine = engine or RetrievalEngine(resolved_artifact_dir)
        if os.getenv("VIDORE_RAG_PRELOAD_DENSE", "false").casefold() == "true":
            active_engine.preload_dense()
        application.state.engine = active_engine
        yield

    application = FastAPI(
        title="ViDoRe RAG Research API",
        version="0.2.0",
        description=(
            "Inspectable retrieval, context construction, and grounded output validation "
            "over the frozen ViDoRe V3 HR corpus."
        ),
        lifespan=lifespan,
    )
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://visual-document-rag.vercel.app",
    ]
    configured_origins = os.getenv("VIDORE_RAG_CORS_ORIGINS", "")
    origins.extend(value.strip() for value in configured_origins.split(",") if value.strip())
    application.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(set(origins)),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    limiter = RequestRateLimiter(int(os.getenv("VIDORE_RAG_REQUESTS_PER_MINUTE", "60")))

    @application.middleware("http")
    async def request_observability(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        client = request.client.host if request.client else "unknown"
        if request.url.path != "/api/v1/health" and not limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={"detail": "request rate limit exceeded", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            json.dumps(
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(latency_ms, 3),
                }
            )
        )
        return response

    def current_engine(request: Request) -> RetrievalEngine:
        return request.app.state.engine  # type: ignore[no-any-return]

    @application.get("/api/v1/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return current_engine(request).health()

    @application.get("/api/v1/queries", response_model=QueryListResponse)
    def queries(request: Request) -> QueryListResponse:
        return current_engine(request).list_queries()

    @application.get("/api/v1/pages/{page_id:path}", response_model=PageResponse)
    def page(page_id: str, request: Request) -> PageResponse:
        try:
            return current_engine(request).get_page(page_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="page not found") from exc

    @application.post("/api/v1/retrieve", response_model=RetrievalTraceResponse)
    def retrieve(payload: RetrieveRequest, request: Request) -> RetrievalTraceResponse:
        try:
            return current_engine(request).retrieve(payload)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"retrieval dependency unavailable: {exc}",
            ) from exc

    @application.post("/api/v1/validate", response_model=ValidationResponse)
    def validate(payload: ValidationRequest, request: Request) -> ValidationResponse:
        try:
            return current_engine(request).validate(payload.trace_id, payload.answer)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="trace not found or expired; run retrieval again",
            ) from exc

    return application


app = create_app()
