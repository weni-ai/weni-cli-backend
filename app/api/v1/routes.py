"""
Main API router that includes all sub-routers.
"""

from fastapi import APIRouter

from app.api.v1.routers.agents import router as agents_router
from app.api.v1.routers.health import router as health_router
from app.api.v1.routers.permissions import router as permissions_router
from app.api.v1.routers.runs import router as runs_router

# Create main API router
router = APIRouter(prefix="/v1")

# Include sub-routers
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(agents_router, prefix="/agents", tags=["Agents"])
router.include_router(runs_router, prefix="/runs", tags=["Runs"])
router.include_router(permissions_router, prefix="/permissions", tags=["Permissions"])
