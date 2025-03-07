"""Test for health endpoint."""

import json
from datetime import UTC, datetime

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Return an API path for health check."""
    return f"{settings.API_PREFIX}/v1/health"


def test_health_endpoint_returns_200(client: TestClient, api_path: str, mock_auth_middleware: None) -> None:
    """Test health endpoint returns 200 OK."""
    response = client.get(api_path)
    assert response.status_code == status.HTTP_200_OK


def test_health_endpoint_returns_json(client: TestClient, api_path: str, mock_auth_middleware: None) -> None:
    """Test health endpoint returns JSON."""
    response = client.get(api_path)
    assert response.headers["Content-Type"] == "application/json"
    assert json.loads(response.content)  # Ensure the response is valid JSON


def test_health_endpoint_returns_expected_fields(
    client: TestClient, api_path: str, mock_auth_middleware: None
) -> None:
    """Test health endpoint returns expected fields."""
    response = client.get(api_path)
    data = json.loads(response.content)
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data


def test_health_endpoint_status_is_ok(client: TestClient, api_path: str, mock_auth_middleware: None) -> None:
    """Test health endpoint status is OK."""
    response = client.get(api_path)
    data = json.loads(response.content)
    assert data["status"] == "ok"


def test_health_endpoint_version_matches_settings(
    client: TestClient, api_path: str, mock_auth_middleware: None
) -> None:
    """Test health endpoint version matches settings."""
    response = client.get(api_path)
    data = json.loads(response.content)
    assert data["version"] == settings.VERSION


def test_health_endpoint_timestamp_is_recent_utc(
    client: TestClient, api_path: str, mock_auth_middleware: None, mocker: MockerFixture
) -> None:
    """Test health endpoint timestamp is recent and in UTC."""
    # Create a fixed datetime for testing
    mock_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Create a class that mocks datetime
    mock_datetime = mocker.MagicMock()
    mock_datetime.now.return_value = mock_dt
    mock_datetime.UTC = UTC  # Keep the original UTC

    # Patch the datetime class in the health router
    mocker.patch("app.api.v1.routers.health.datetime", mock_datetime)

    # Make the request
    response = client.get(api_path)
    data = json.loads(response.content)

    # Parse timestamp from response
    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    # Assert timestamp is our mocked time
    assert timestamp.tzinfo is not None  # Ensure timestamp has timezone info
    assert timestamp == mock_dt  # Timestamp should match our mocked time exactly
