"""Tests for permissions endpoints."""

import json
import uuid
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from requests import Response

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Return an API path for permissions verify endpoint."""
    return f"{settings.API_PREFIX}/v1/permissions/verify"


@pytest.fixture
def valid_request_data() -> dict[str, Any]:
    """Return valid request data for permission verification."""
    return {"project_uuid": str(uuid.uuid4())}


@pytest.fixture
def mock_connect_client(mocker: MockerFixture) -> Any:
    """Mock the ConnectClient."""
    mock = mocker.MagicMock()
    mocker.patch("app.api.v1.routers.permissions.ConnectClient", return_value=mock)
    return mock


def test_verify_permission_success(
    client: TestClient, api_path: str, valid_request_data: dict[str, Any], mock_connect_client: Any
) -> None:
    """Test successful permission verification."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_200_OK
    mock_connect_client.check_authorization.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={"Authorization": "Bearer test_token"},
    )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}
    mock_connect_client.check_authorization.assert_called_once()


def test_verify_permission_unauthorized(
    client: TestClient, api_path: str, valid_request_data: dict[str, Any], mock_connect_client: Any
) -> None:
    """Test unauthorized permission verification."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_401_UNAUTHORIZED
    mock_response._content = json.dumps({"detail": "Unauthorized"}).encode()
    mock_connect_client.check_authorization.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={"Authorization": "Bearer invalid_token"},
    )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["status"] == "error"
    assert "Unauthorized" in str(result["message"])
    mock_connect_client.check_authorization.assert_called_once()


def test_verify_permission_exception(
    client: TestClient, api_path: str, valid_request_data: dict[str, Any], mock_connect_client: Any
) -> None:
    """Test exception during permission verification."""
    # Setup
    mock_connect_client.check_authorization.side_effect = Exception("Connection error")

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={"Authorization": "Bearer test_token"},
    )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result["status"] == "error"
    assert result["message"] == "Connection error"
    mock_connect_client.check_authorization.assert_called_once()


def test_verify_permission_missing_authorization(
    client: TestClient, api_path: str, valid_request_data: dict[str, Any]
) -> None:
    """Test missing authorization header."""
    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        # No Authorization header
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "authorization" in response.text.lower()


def test_verify_permission_invalid_request_data(client: TestClient, api_path: str) -> None:
    """Test invalid request data."""
    # Execute
    response = client.post(
        api_path,
        json={"invalid_field": "value"},  # Missing required project_uuid
        headers={"Authorization": "Bearer test_token"},
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "project_uuid" in response.text.lower()
