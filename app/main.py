"""
Main FastAPI application.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.middlewares import AuthorizationMiddleware, VersionCheckMiddleware
from app.api.v1.routes import router as api_v1_router
from app.core.config import settings

logger = logging.getLogger(__name__)

SENTRY_ENABLED = bool(settings.SENTRY_DSN)

if SENTRY_ENABLED:
    logger.info(
        "Sentry enabled | environment=%s | traces_sample_rate=%s | profiles_sample_rate=%s | debug=%s",
        settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
        settings.SENTRY_TRACES_SAMPLE_RATE,
        settings.SENTRY_PROFILES_SAMPLE_RATE,
        settings.SENTRY_DEBUG,
    )
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
        release=settings.SENTRY_RELEASE,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
        send_default_pii=True,
        debug=settings.SENTRY_DEBUG,
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
else:
    logger.info("Sentry disabled (empty SENTRY_DSN)")


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

    if SENTRY_ENABLED:
        # Ensure Sentry wraps the ASGI app so requests become transactions in APM.
        # sentry-sdk's type hints don't match Starlette's middleware protocol perfectly; runtime is OK.
        app.add_middleware(cast(Any, SentryAsgiMiddleware))

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
