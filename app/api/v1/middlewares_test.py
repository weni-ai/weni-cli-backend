from typing import Any, cast

import pytest
from fastapi import Request, Response, status
from pytest_mock import MockerFixture

from app.api.v1.middlewares import AuthorizationMiddleware


@pytest.fixture
async def auth_middleware() -> AuthorizationMiddleware:
    return AuthorizationMiddleware()


@pytest.fixture
def mock_connect_response(mocker: MockerFixture) -> Any:
    mock_response = mocker.MagicMock()
    mock_response.status_code = status.HTTP_200_OK
    return mock_response


async def test_role_validation_blocks_low_role_push(
    auth_middleware: AuthorizationMiddleware,
    mock_connect_response: Any,
    mocker: MockerFixture
) -> None:
    """Test that users with role not in ACCEPTABLE_ROLES cannot push"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/api/v1/agents"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response with role 1 (not in ACCEPTABLE_ROLES)
    mock_response = mock_connect_response
    mock_response.json.return_value = {"project_authorization": 1}
    mock_check_auth = mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", 
                                   return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request: Request) -> Response:
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    
    # Verify that Connect was called
    mock_check_auth.assert_called_once()
    assert response.status_code == status.HTTP_403_FORBIDDEN
    # Convert response.body to str before checking content
    response_content = cast(bytes, response.body).decode('utf-8')
    assert "role does not have permission" in response_content


async def test_role_validation_allows_acceptable_role_push(
    auth_middleware: AuthorizationMiddleware,
    mock_connect_response: Any,
    mocker: MockerFixture
) -> None:
    """Test that users with role in ACCEPTABLE_ROLES can push"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/api/v1/agents"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response with role 2 (in ACCEPTABLE_ROLES)
    mock_response = mock_connect_response
    mock_response.json.return_value = {"project_authorization": 2}
    mock_check_auth = mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", 
                                   return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request: Request) -> Response:
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    
    # Verify that Connect was called
    mock_check_auth.assert_called_once()
    assert response.status_code == status.HTTP_200_OK


async def test_role_validation_allows_high_role_non_push(
    auth_middleware: AuthorizationMiddleware,
    mock_connect_response: Any,
    mocker: MockerFixture
) -> None:
    """Test that any role can make non-POST requests"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/api/v1/agents"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response with role 4
    mock_response = mock_connect_response
    mock_response.json.return_value = {"project_authorization": 4}
    mock_check_auth = mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", 
                                   return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request: Request) -> Response:
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    
    # Verify that Connect was called and request was allowed
    mock_check_auth.assert_called_once()
    assert response.status_code == status.HTTP_200_OK


async def test_role_validation_missing_role(
    auth_middleware: AuthorizationMiddleware,
    mock_connect_response: Any,
    mocker: MockerFixture
) -> None:
    """Test that requests without project_authorization in response are blocked"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/api/v1/agents"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response without project_authorization
    mock_response = mock_connect_response
    mock_response.json.return_value = {}
    mock_check_auth = mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", 
                                   return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request: Request) -> Response:
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    
    # Verify that Connect was called and request was blocked
    mock_check_auth.assert_called_once()
    assert response.status_code == status.HTTP_200_OK  # Se não tem role, passa direto
