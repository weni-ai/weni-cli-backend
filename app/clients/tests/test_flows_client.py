"""Tests for FlowsClient class."""

import base64
import json
from http import HTTPStatus
from urllib.parse import urlparse

import pytest
import requests
import requests_mock
from pytest_mock import MockerFixture

from app.clients.flows_client import FlowsClient
from app.core.config import settings


def create_test_jwt_token(email: str) -> str:
    """Create a test JWT token with the given email."""
    payload = {"email": email}
    encoded_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"Bearer header.{encoded_payload}.signature"


class TestFlowsClient:
    """Tests for the FlowsClient class."""

    TEST_PROJECT_UUID = "123e4567-e89b-12d3-a456-426614174000"
    TEST_USER_EMAIL = "test@example.com"
    TEST_AUTH_TOKEN = create_test_jwt_token(TEST_USER_EMAIL)
    TEST_CHANNEL_DEFINITION = {
        "channel_type": "WAC",
        "name": "Test Channel",
        "address": "+5511999999999",
        "schemes": ["tel"],
        "config": {"wa_pin": "123456", "wa_verified_name": "Test Business"},
    }
    TEST_RESPONSE_DATA = {
        "uuid": "channel-uuid-123",
        "name": "Test Channel",
        "address": "+5511999999999",
        "channel_type": "WAC",
    }

    def test_init(self) -> None:
        """Test FlowsClient initialization."""
        # Act
        client = FlowsClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)

        # Assert
        assert client.headers == {"Authorization": self.TEST_AUTH_TOKEN}
        assert client.base_url == settings.FLOWS_BASE_URL
        assert client.project_uuid == self.TEST_PROJECT_UUID
        assert client.user_email == self.TEST_USER_EMAIL

    @pytest.mark.parametrize(
        "status_code,response_data",
        [
            (HTTPStatus.OK, TEST_RESPONSE_DATA),
            (HTTPStatus.CREATED, TEST_RESPONSE_DATA),
            (HTTPStatus.BAD_REQUEST, {"detail": "Invalid channel data"}),
            (HTTPStatus.UNAUTHORIZED, {"detail": "Invalid authentication credentials"}),
            (HTTPStatus.FORBIDDEN, {"detail": "Permission denied"}),
            (HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "Internal server error"}),
        ],
    )
    def test_create_channel_status_codes(
        self, requests_mock: requests_mock.Mocker, status_code: int, response_data: dict
    ) -> None:
        """Test create_channel method with different status codes."""
        # Arrange
        client = FlowsClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)
        expected_url = f"{settings.FLOWS_BASE_URL}/api/v2/internals/channel/"

        # Mock the API response
        requests_mock.post(expected_url, status_code=status_code, json=response_data)

        # Act
        response = client.create_channel(self.TEST_CHANNEL_DEFINITION)

        # Assert
        assert response.status_code == status_code
        assert response.json() == response_data

        # Verify the request had the correct headers
        assert requests_mock.last_request is not None
        assert requests_mock.last_request.headers["Authorization"] == self.TEST_AUTH_TOKEN

        # Check URL construction
        parsed_url = urlparse(requests_mock.last_request.url)
        assert f"{parsed_url.scheme}://{parsed_url.netloc}" == settings.FLOWS_BASE_URL.rstrip("/")
        assert parsed_url.path == "/api/v2/internals/channel/"

    def test_create_channel_request_body(self, requests_mock: requests_mock.Mocker) -> None:
        """Test that create_channel sends the correct request body as JSON."""
        # Arrange
        client = FlowsClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)
        expected_url = f"{settings.FLOWS_BASE_URL}/api/v2/internals/channel/"

        # Mock the API response
        requests_mock.post(expected_url, status_code=HTTPStatus.CREATED, json=self.TEST_RESPONSE_DATA)

        # Act
        response = client.create_channel(self.TEST_CHANNEL_DEFINITION)

        # Assert
        assert response.status_code == HTTPStatus.CREATED

        # Verify request body follows the correct format
        assert requests_mock.last_request is not None
        request_json = requests_mock.last_request.json()
        
        # ClaimView expects these fields at top level
        assert request_json["org"] == self.TEST_PROJECT_UUID
        assert request_json["user"] == self.TEST_USER_EMAIL
        assert request_json["channeltype_code"] == "WAC"
        assert request_json["name"] == "Test Channel"
        assert request_json["address"] == "+5511999999999"
        assert request_json["schemes"] == ["tel"]
        
        # Config data should be nested under 'data' key
        assert "data" in request_json
        assert request_json["data"] == {"wa_pin": "123456", "wa_verified_name": "Test Business"}

    @pytest.mark.parametrize(
        "exception_class,exception_message",
        [
            (requests.ConnectionError, "Connection refused"),
            (requests.Timeout, "Request timed out"),
        ],
    )
    def test_create_channel_exceptions(
        self, mocker: MockerFixture, exception_class: type[Exception], exception_message: str
    ) -> None:
        """Test create_channel method when exceptions occur."""
        # Arrange
        client = FlowsClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)

        # Mock the requests.post method to raise the specified exception
        mocker.patch("requests.post", side_effect=exception_class(exception_message))

        # Act & Assert
        with pytest.raises(exception_class, match=exception_message):
            client.create_channel(self.TEST_CHANNEL_DEFINITION)

    def test_empty_auth_token(self) -> None:
        """Test FlowsClient with empty auth token."""
        # Arrange & Act
        client = FlowsClient("", self.TEST_PROJECT_UUID)

        # Assert
        assert client.headers == {"Authorization": ""}
        assert client.user_email == ""  # Should be empty when token is invalid

    def test_empty_project_uuid(self, requests_mock: requests_mock.Mocker) -> None:
        """Test FlowsClient with empty project UUID."""
        # Arrange
        client = FlowsClient(self.TEST_AUTH_TOKEN, "")
        expected_url = f"{settings.FLOWS_BASE_URL}/api/v2/internals/channel/"

        # Mock the API response
        requests_mock.post(expected_url, status_code=HTTPStatus.BAD_REQUEST, json={"detail": "Invalid project"})

        # Act
        response = client.create_channel(self.TEST_CHANNEL_DEFINITION)

        # Assert
        assert client.project_uuid == ""
        assert response.status_code == HTTPStatus.BAD_REQUEST

    def test_empty_channel_definition(self, requests_mock: requests_mock.Mocker) -> None:
        """Test create_channel with empty channel definition."""
        # Arrange
        client = FlowsClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)
        expected_url = f"{settings.FLOWS_BASE_URL}/api/v2/internals/channel/"

        # Mock the API response
        requests_mock.post(
            expected_url, status_code=HTTPStatus.BAD_REQUEST, json={"detail": "Channel data is required"}
        )

        # Act
        response = client.create_channel({})

        # Assert
        assert response.status_code == HTTPStatus.BAD_REQUEST
