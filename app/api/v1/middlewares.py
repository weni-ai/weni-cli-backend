import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status

from app.clients.weni_client import WeniClient

logger = logging.getLogger(__name__)

NO_AUTH_ENDPOINTS = [
    "/api/v1/health",
]


class AuthorizationMiddleware:
    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path in NO_AUTH_ENDPOINTS:
            return await call_next(request)

        logger.info(f"Verifying project token for request: {request}")
        weni_client = WeniClient(request.headers.get("Authorization", ""), request.headers.get("X-Project-Uuid", ""))

        try:
            response = weni_client.check_authorization()

            logger.info(f"Response: {response.json()}")

            if response.status_code != status.HTTP_200_OK:
                return Response(status_code=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.error(f"Error verifying project token: {e}")
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)

        return await call_next(request)
