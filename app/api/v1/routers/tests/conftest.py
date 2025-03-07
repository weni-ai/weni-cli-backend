"""
Shared fixtures for API router tests.
"""

from collections.abc import Awaitable, Callable

import pytest
from fastapi import Request, Response, status
from pytest_mock import MockerFixture

from app.api.v1.middlewares import NO_AUTH_ENDPOINTS


@pytest.fixture(scope="function")
def mock_auth_middleware(mocker: MockerFixture) -> None:
    """
    Mock the authorization middleware to allow test requests to pass through.

    This fixture mocks the AuthorizationMiddleware class to bypass the
    authorization check completely and always pass through to the next handler.
    """

    # Mock the __call__ method of the middleware to always call_next
    async def mock_call(  # type: ignore
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in NO_AUTH_ENDPOINTS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        project_uuid = request.headers.get("X-Project-Uuid")

        if not auth_header or not project_uuid:
            print("Missing Authorization or X-Project-Uuid header", request.url.path)
            return Response(
                status_code=status.HTTP_400_BAD_REQUEST, content="Missing Authorization or X-Project-Uuid header"
            )

        return await call_next(request)

    # Apply the mock to the middleware
    mocker.patch("app.api.v1.middlewares.AuthorizationMiddleware.__call__", mock_call)
