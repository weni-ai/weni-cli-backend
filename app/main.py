"""
Main FastAPI application.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.middlewares import AuthorizationMiddleware, VersionCheckMiddleware
from app.api.v1.routes import router as api_v1_router
from app.core.config import settings

sentry_sdk.init(
    dsn=settings.SENTRY_DSN,
    traces_sample_rate=1.0,
    send_default_pii=True,
    integrations=[
        StarletteIntegration(
            transaction_style="endpoint",
            failed_request_status_codes={*range(401, 599)},
        ),
        FastApiIntegration(
            transaction_style="endpoint",
            failed_request_status_codes={*range(401, 599)},
        ),
    ],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # pragma: no cover
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

    authorization_middleware = AuthorizationMiddleware()
    app.add_middleware(BaseHTTPMiddleware, dispatch=authorization_middleware)

    version_check_middleware = VersionCheckMiddleware()
    app.add_middleware(BaseHTTPMiddleware, dispatch=version_check_middleware)

    # Include routers
    app.include_router(api_v1_router, prefix=settings.API_PREFIX)

    return app


app = create_application()
