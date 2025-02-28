"""
Server entry point.
"""

import uvicorn

from app.core.config import settings


def start() -> None:
    """
    Start the API server using uvicorn.
    """
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL,
    )


if __name__ == "__main__":
    start()
