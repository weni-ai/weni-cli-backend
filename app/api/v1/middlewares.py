import logging
from collections.abc import Awaitable, Callable

import requests
from fastapi import Request, Response, status
from packaging.version import InvalidVersion, Version

from app.clients.connect_client import ConnectClient
from app.core.config import settings

logger = logging.getLogger(__name__)

NO_AUTH_ENDPOINTS = ["/api/v1/health", "/api/v1/health/", "/api/v1/permissions/verify"]
NO_VERSION_CHECK_ENDPOINTS = ["/api/v1/health", "/api/v1/health/"]
ACCEPTABLE_ROLES = [2, 3, 4] # 2 = contributor, 3 = moderator, 4 = support


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
            response_data = response.json()
            
            if response.status_code != status.HTTP_200_OK:
                return Response(status_code=status.HTTP_401_UNAUTHORIZED)

            if "project_authorization" in response_data:
                user_role = response_data["project_authorization"]
                if request.method == "POST" and user_role not in ACCEPTABLE_ROLES:
                    logger.info(f"User with role {user_role} attempted to push. Access denied.")
                    return Response(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content="Your role does not have permission to push agents"
                    )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error during authorization check: {e}")
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.exception(f"Unexpected error during authorization check: {e}")
            return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return await call_next(request)


class VersionCheckMiddleware:
    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path in NO_VERSION_CHECK_ENDPOINTS:
            return await call_next(request)

        cli_version_header = request.headers.get("X-CLI-Version")

        if not cli_version_header:
            return Response(
                status_code=status.HTTP_426_UPGRADE_REQUIRED,
                content="Upgrade required, please update your CLI to the latest version",
            )

        try:
            # Parse versions and compare them
            if not self._is_version_compatible(cli_version_header, settings.CLI_MINIMUM_VERSION):
                return Response(
                    status_code=status.HTTP_426_UPGRADE_REQUIRED,
                    content=(
                        f"Upgrade required. Minimum version: {settings.CLI_MINIMUM_VERSION}, "
                        f"your version: {cli_version_header}"
                    ),
                )
        except ValueError as e:
            logger.error(f"Error during version check: {e} - cli_version_header: {cli_version_header}")
            return Response(
                status_code=status.HTTP_400_BAD_REQUEST, content=f"Invalid version format: {cli_version_header}"
            )

        return await call_next(request)

    def _is_version_compatible(self, client_version: str, minimum_version: str) -> bool:
        """
        Check if client version is compatible with minimum required version using PEP 0440 standards.

        Args:
            client_version: Version string in format "a.b.c"
            minimum_version: Minimum required version in format "a.b.c"

        Returns:
            True if client version is greater than or equal to minimum version
        """
        try:
            client_ver = Version(client_version)
            minimum_ver = Version(minimum_version)
            return client_ver >= minimum_ver
        except InvalidVersion as e:
            raise ValueError("Invalid version format") from e
