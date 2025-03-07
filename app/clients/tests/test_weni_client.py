"""Tests for WeniClient class."""

from http import HTTPStatus
from urllib.parse import urlparse

import pytest
import requests
import requests_mock
from pytest_mock import MockerFixture

from app.clients.weni_client import WeniClient
from app.core.config import settings


class TestWeniClient:
    """Tests for the WeniClient class."""

    TEST_AUTH_TOKEN = "test-token"
    TEST_PROJECT_UUID = "123e4567-e89b-12d3-a456-426614174000"
    TEST_RESPONSE_DATA = {
        "valid": True,
        "user": {"id": "user-id", "email": "user@example.com", "first_name": "Test", "last_name": "User"},
        "permissions": ["read", "write"],
    }

    def test_init(self) -> None:
        """Test WeniClient initialization."""
        # Act
        client = WeniClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)

        # Assert
        assert client.headers == {"Authorization": self.TEST_AUTH_TOKEN}
        assert client.base_url == settings.WENI_API_URL
        assert client.project_uuid == self.TEST_PROJECT_UUID

    @pytest.mark.parametrize(
        "status_code,response_data,expected_text",
        [
            (HTTPStatus.OK, TEST_RESPONSE_DATA, None),
            (
                HTTPStatus.UNAUTHORIZED,
                {"detail": "Invalid authentication credentials"},
                "Invalid authentication credentials",
            ),
            (HTTPStatus.FORBIDDEN, {"detail": "Permission denied"}, "Permission denied"),
            (HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "Internal server error"}, "Internal server error"),
            (HTTPStatus.NOT_FOUND, {"detail": "Project not found"}, "Project not found"),
        ],
    )
    def test_check_authorization_status_codes(
        self, requests_mock: requests_mock.Mocker, status_code: int, response_data: dict, expected_text: str | None
    ) -> None:
        """Test check_authorization method with different status codes."""
        # Arrange
        client = WeniClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)
        expected_url = f"{settings.WENI_API_URL}/v2/projects/{self.TEST_PROJECT_UUID}/authorization"

        # Mock the API response
        requests_mock.get(expected_url, status_code=status_code, json=response_data)

        # Act
        response = client.check_authorization()

        # Assert
        assert response.status_code == status_code
        if status_code == HTTPStatus.OK:
            assert response.json() == response_data
            # Verify the request had the correct headers
            assert requests_mock.last_request is not None
            assert requests_mock.last_request.headers["Authorization"] == self.TEST_AUTH_TOKEN

            # Check URL construction
            parsed_url = urlparse(requests_mock.last_request.url)
            assert f"{parsed_url.scheme}://{parsed_url.netloc}" == settings.WENI_API_URL.rstrip("/")
            assert parsed_url.path == f"/v2/projects/{self.TEST_PROJECT_UUID}/authorization"
        elif expected_text:
            assert expected_text in response.text

    @pytest.mark.parametrize(
        "exception_class,exception_message",
        [
            (requests.ConnectionError, "Connection refused"),
            (requests.Timeout, "Request timed out"),
        ],
    )
    def test_check_authorization_exceptions(
        self, mocker: MockerFixture, exception_class: type[Exception], exception_message: str
    ) -> None:
        """Test check_authorization method when exceptions occur."""
        # Arrange
        client = WeniClient(self.TEST_AUTH_TOKEN, self.TEST_PROJECT_UUID)

        # Mock the requests.get method to raise the specified exception
        mocker.patch("requests.get", side_effect=exception_class(exception_message))

        # Act & Assert
        with pytest.raises(exception_class, match=exception_message):
            client.check_authorization()

    def test_empty_auth_token(self) -> None:
        """Test WeniClient with empty auth token."""
        # Arrange & Act
        client = WeniClient("", self.TEST_PROJECT_UUID)

        # Assert
        assert client.headers == {"Authorization": ""}

    def test_empty_project_uuid(self, requests_mock: requests_mock.Mocker) -> None:
        """Test WeniClient with empty project UUID."""
        # Arrange
        client = WeniClient(self.TEST_AUTH_TOKEN, "")
        expected_url = f"{settings.WENI_API_URL}/v2/projects//authorization"

        # Mock the API response
        requests_mock.get(expected_url, status_code=HTTPStatus.NOT_FOUND, json={"detail": "Project not found"})

        # Act
        response = client.check_authorization()

        # Assert
        assert client.project_uuid == ""
        assert response.status_code == HTTPStatus.NOT_FOUND
