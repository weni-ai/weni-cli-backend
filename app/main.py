"""
Main FastAPI application.
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as api_v1_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup logic (e.g., database connections, caches)
    yield
    # Shutdown logic


def create_application() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        docs_url=settings.DOCS_URL,
        lifespan=lifespan,
    )

    # Configure CORS to allow requests from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow any origin
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )

    # Include routers
    app.include_router(api_v1_router, prefix=settings.API_PREFIX)

    return app


app = create_application()
