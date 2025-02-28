"""
Main API router that includes all sub-routers.
"""
from fastapi import APIRouter

from app.api.v1.routers.agents import router as agents_router
from app.api.v1.routers.health import router as health_router

# Create main API router
router = APIRouter(prefix="/v1")

# Include sub-routers
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(agents_router, prefix="/agents", tags=["Agents"])
