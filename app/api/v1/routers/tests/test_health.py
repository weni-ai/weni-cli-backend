"""
Tests for health check endpoint.
"""
from datetime import UTC, datetime, timezone
from unittest import mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture
def api_path() -> str:
    """Get the correct API path for health endpoint."""
    return f"{settings.API_PREFIX}/v1/health"


def test_health_endpoint_returns_200(client: TestClient, api_path: str) -> None:
    """Test that the health endpoint returns a 200 status code."""
    response = client.get(api_path)
    assert response.status_code == status.HTTP_200_OK


def test_health_endpoint_returns_json(client: TestClient, api_path: str) -> None:
    """Test that the health endpoint returns JSON data."""
    response = client.get(api_path)
    assert response.headers["Content-Type"] == "application/json"


def test_health_endpoint_returns_expected_fields(client: TestClient, api_path: str) -> None:
    """Test that the health endpoint returns the expected fields."""
    response = client.get(api_path)
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data


def test_health_endpoint_status_is_ok(client: TestClient, api_path: str) -> None:
    """Test that the health endpoint status is 'ok'."""
    response = client.get(api_path)
    data = response.json()
    assert data["status"] == "ok"


def test_health_endpoint_version_matches_settings(client: TestClient, api_path: str) -> None:
    """Test that the health endpoint version matches the settings version."""
    response = client.get(api_path)
    data = response.json()
    assert data["version"] == settings.VERSION


def test_health_endpoint_timestamp_is_recent_utc(client: TestClient, api_path: str) -> None:
    """Test that the health endpoint timestamp is a recent UTC datetime."""
    # Use a mock to fix the current time for testing
    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

    with mock.patch("app.api.v1.routers.health.datetime") as mock_datetime:
        # Set up the mock
        mock_datetime.now.return_value = fixed_datetime
        mock_datetime.timezone = timezone

        # Make the request
        response = client.get(api_path)
        data = response.json()

        # Parse the returned timestamp
        returned_timestamp = datetime.fromisoformat(data["timestamp"])

        # Assert that the returned timestamp is the mocked time
        assert returned_timestamp == fixed_datetime
