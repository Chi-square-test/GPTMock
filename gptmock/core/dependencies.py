from __future__ import annotations

from functools import lru_cache

import httpx
from fastapi import Request

from .settings import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_http_client(request: Request) -> httpx.AsyncClient:
    """FastAPI dependency — returns the lifespan-managed httpx client."""
    return request.app.state.http_client
