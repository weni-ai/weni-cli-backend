"""
Main API router that includes all sub-routers.
"""

from fastapi import APIRouter

from app.api.v1.routers.agents import router as agents_router
from app.api.v1.routers.health import router as health_router
from app.api.v1.routers.permissions import router as permissions_router
from app.api.v1.routers.passive_runs import router as passive_runs_router
from app.api.v1.routers.active_runs import router as active_runs_router
from app.api.v1.routers.tool_logs import router as tool_logs_router

# Create main API router
router = APIRouter(prefix="/v1")

# Include sub-routers
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(agents_router, prefix="/agents", tags=["Agents"])
router.include_router(passive_runs_router, prefix="/passive_runs", tags=["Passive Runs"])
router.include_router(active_runs_router, prefix="/active_runs", tags=["Active Runs"])
router.include_router(permissions_router, prefix="/permissions", tags=["Permissions"])
router.include_router(tool_logs_router, prefix="/tool-logs", tags=["Tool Logs"])
