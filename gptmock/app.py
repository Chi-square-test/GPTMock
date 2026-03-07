from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gptmock.core.settings import Settings
from gptmock.routers.health import router as health_router
from gptmock.routers.ollama import router as ollama_router
from gptmock.routers.openai import router as openai_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage a single httpx.AsyncClient for the application lifetime."""
    app.state.http_client = httpx.AsyncClient(timeout=300.0)
    try:
        yield
    finally:
        await app.state.http_client.aclose()

def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Optional Settings instance. If None, creates from environment.

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        from gptmock.core.dependencies import get_settings
        settings = get_settings()

    # Create FastAPI app with lifespan
    app = FastAPI(
        title="gptmock",
        description="OpenAI & Ollama compatible API powered by ChatGPT",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(openai_router)
    app.include_router(ollama_router)

    return app
