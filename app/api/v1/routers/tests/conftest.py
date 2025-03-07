"""
Shared fixtures for API router tests.
"""

import pytest
import requests
from fastapi import status
from pytest_mock import MockerFixture


@pytest.fixture(scope="function")
def mock_auth_middleware(mocker: MockerFixture) -> None:
    """
    Mock the authorization middleware to allow test requests to pass through.

    This fixture mocks the WeniClient class and its check_authorization method to return
    a successful response, bypassing the actual authorization check.
    """
    # Create a successful mock response
    mock_response = mocker.MagicMock(spec=requests.Response)
    mock_response.status_code = status.HTTP_200_OK
    mock_response.json.return_value = {
        "valid": True,
        "user": {"id": "test-user-id", "email": "test@example.com", "first_name": "Test", "last_name": "User"},
        "permissions": ["test.permission"],
    }

    # Create a mock for check_authorization method
    check_auth_mock = mocker.MagicMock(return_value=mock_response)

    # Create WeniClient mock class
    MockWeniClient = mocker.MagicMock()  # noqa: N806
    # Set the check_authorization as an instance method on the MockWeniClient instance
    MockWeniClient.return_value.check_authorization = check_auth_mock

    # Patch the WeniClient import
    mocker.patch("app.api.v1.middlewares.WeniClient", MockWeniClient)
