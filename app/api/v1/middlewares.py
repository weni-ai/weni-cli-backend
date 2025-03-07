import logging
from collections.abc import Awaitable, Callable

import requests
from fastapi import Request, Response, status

from app.clients.connect_client import ConnectClient

logger = logging.getLogger(__name__)

NO_AUTH_ENDPOINTS = ["/api/v1/health", "/api/v1/health/"]


class AuthorizationMiddleware:
    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path in NO_AUTH_ENDPOINTS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        project_uuid = request.headers.get("X-Project-Uuid")

        if not auth_header or not project_uuid:
            return Response(
                status_code=status.HTTP_400_BAD_REQUEST, content="Missing Authorization or X-Project-Uuid header"
            )

        connect_client = ConnectClient(
            auth_header,
            project_uuid,
        )

        try:
            response = connect_client.check_authorization()
            if response.status_code != status.HTTP_200_OK:
                return Response(status_code=status.HTTP_401_UNAUTHORIZED)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during authorization check: {e}")
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.exception(f"Unexpected error during authorization check: {e}")
            return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return await call_next(request)
