"""
Health check endpoints.
"""
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    version: str
    timestamp: datetime


router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint to verify API is running.

    Returns:
        Dict containing status, version, and current timestamp.
    """
    return {
        "status": "ok",
        "version": settings.VERSION,
        "timestamp": datetime.now(UTC),
    }
