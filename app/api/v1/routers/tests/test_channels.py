"""Tests for channels endpoints."""

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
    """Return an API path for channels endpoint."""
    return f"{settings.API_PREFIX}/v1/channels"


@pytest.fixture
def project_uuid() -> str:
    """Return a test project UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def valid_request_data(project_uuid: str) -> dict[str, Any]:
    """Return valid request data for channel creation."""
    channel_definition = {
        "channel_type": "WAC",
        "name": "Test Channel",
        "address": "+5511999999999",
        "config": {"wa_pin": "123456", "wa_verified_name": "Test Business"},
    }
    return {"project_uuid": project_uuid, "channel_definition": channel_definition}


@pytest.fixture
def mock_flows_client(mocker: MockerFixture) -> Any:
    """Mock the FlowsClient."""
    mock = mocker.MagicMock()
    mocker.patch("app.api.v1.routers.channels.FlowsClient", return_value=mock)
    return mock


def test_create_channel_success(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test successful channel creation."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_201_CREATED
    mock_response._content = json.dumps(
        {"uuid": "channel-uuid-123", "name": "Test Channel", "address": "+5511999999999", "channel_type": "WAC"}
    ).encode()
    mock_flows_client.create_channel.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    result = response.json()
    assert result["uuid"] == "channel-uuid-123"
    assert result["name"] == "Test Channel"
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_bad_request(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with bad request."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_400_BAD_REQUEST
    mock_response._content = json.dumps({"detail": "Invalid channel data"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_unauthorized(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with unauthorized request."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_401_UNAUTHORIZED
    mock_response._content = json.dumps({"detail": "Invalid authentication credentials"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer invalid_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_forbidden(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with forbidden access."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_403_FORBIDDEN
    mock_response._content = json.dumps({"detail": "Permission denied"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_internal_server_error(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with internal server error."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_response._content = json.dumps({"detail": "Internal server error"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_exception(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test exception during channel creation."""
    # Setup
    mock_flows_client.create_channel.side_effect = Exception("Unexpected error")

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Internal server error" in response.json()["detail"]
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_missing_project_uuid(client: TestClient, api_path: str, mock_auth_middleware: None) -> None:
    """Test channel creation with missing project_uuid."""
    # Setup
    invalid_data = {"channel_definition": {"channel_type": "WAC", "name": "Test Channel"}}

    # Execute
    response = client.post(
        api_path,
        json=invalid_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": str(uuid.uuid4()),
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_channel_missing_channel_definition(
    client: TestClient, api_path: str, project_uuid: str, mock_auth_middleware: None
) -> None:
    """Test channel creation with missing channel_definition."""
    # Setup
    invalid_data = {"project_uuid": project_uuid}

    # Execute
    response = client.post(
        api_path,
        json=invalid_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_channel_invalid_project_uuid_format(
    client: TestClient, api_path: str, mock_auth_middleware: None
) -> None:
    """Test channel creation with invalid project_uuid format."""
    # Setup
    invalid_data = {
        "project_uuid": "not-a-valid-uuid",
        "channel_definition": {"channel_type": "WAC", "name": "Test Channel"},
    }

    # Execute
    response = client.post(
        api_path,
        json=invalid_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": str(uuid.uuid4()),
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
