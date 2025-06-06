import pytest
from fastapi import Request, Response, status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.api.v1.middlewares import AuthorizationMiddleware
from app.core.config import settings

@pytest.fixture
def auth_middleware():
    return AuthorizationMiddleware()

@pytest.fixture
def mock_connect_response(mocker: MockerFixture):
    mock_response = mocker.MagicMock()
    mock_response.status_code = status.HTTP_200_OK
    return mock_response

async def test_role_validation_blocks_low_role_push(
    auth_middleware,
    mock_connect_response,
    mocker: MockerFixture
):
    """Test that users with role < 3 cannot push"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/api/v1/agents/push"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response with role 2
    mock_response = mock_connect_response
    mock_response.json.return_value = {"role": 2}
    mock_check_auth = mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request):
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    
    # Verify that Connect was called
    mock_check_auth.assert_called_once()
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "role does not have permission" in response.body.decode()

async def test_role_validation_allows_high_role_push(
    auth_middleware,
    mock_connect_response,
    mocker: MockerFixture
):
    """Test that users with role >= 3 can push"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/api/v1/agents/push"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response with role 4
    mock_response = mock_connect_response
    mock_response.json.return_value = {"role": 4}
    mock_check_auth = mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request):
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    
    # Verify that Connect was called
    mock_check_auth.assert_called_once()
    assert response.status_code == status.HTTP_200_OK

async def test_role_validation_allows_high_role_non_push(
    auth_middleware,
    mock_connect_response,
    mocker: MockerFixture
):
    """Test that users with role > 3 can do non-push operations"""
    # Mock the request
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/api/v1/agents"
    mock_request.headers = {
        "Authorization": "Bearer token",
        "X-Project-Uuid": "123"
    }
    
    # Mock the connect client response
    mock_response = mock_connect_response
    mock_response.json.return_value = {"role": 4}
    mocker.patch("app.clients.connect_client.ConnectClient.check_authorization", return_value=mock_response)
    
    # Mock call_next
    async def mock_call_next(request):
        return Response(status_code=status.HTTP_200_OK)
    
    response = await auth_middleware(mock_request, mock_call_next)
    assert response.status_code == status.HTTP_200_OK 