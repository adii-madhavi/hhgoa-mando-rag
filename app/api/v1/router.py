"""Combined v1 API router, mounted at /api by app/main.py."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import meta, text, voice

api_router = APIRouter()
api_router.include_router(meta.router)
api_router.include_router(text.router)
api_router.include_router(voice.router)
