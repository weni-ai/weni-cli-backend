"""Tests for the tool logs endpoint."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.core.config import settings
from app.main import app

# Constants
TEST_AGENT_KEY = "test-agent"
TEST_TOOL_KEY = "test-tool"
TEST_PROJECT_UUID = str(uuid4())
TEST_AUTH_TOKEN = "Bearer test-token"
TEST_LOG_GROUP_ARN = "arn:aws:logs:us-east-1:123456789012:log-group:/aws/lambda/test-function:*"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Get the API path for the tool logs endpoint."""
    return f"{settings.API_PREFIX}/v1/tool-logs/"


@pytest.fixture
def common_headers() -> dict[str, str]:
    """Return common headers needed for requests."""
    return {
        "Authorization": TEST_AUTH_TOKEN,
        "X-Project-Uuid": TEST_PROJECT_UUID,
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


@pytest.fixture
def default_query_params() -> dict[str, Any]:
    """Return default query parameters for the GET request, ensuring values are strings."""
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(hours=1)
    return {
        "agent_key": TEST_AGENT_KEY,
        "tool_key": TEST_TOOL_KEY,
        "start_time": start_time.isoformat(), # Ensure ISO string format
        "end_time": end_time.isoformat(),     # Ensure ISO string format
        # pattern and next_token are optional and will be None if not set
    }


class TestGetLogsEndpoint:
    """Tests for the GET / endpoint in tool_logs router."""

    def test_get_logs_success(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        common_headers: dict[str, str],
        default_query_params: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test successful retrieval of logs."""
        # Mock NexusClient response
        mock_nexus_response = mocker.Mock()
        mock_nexus_response.status_code = status.HTTP_200_OK
        mock_nexus_response.json.return_value = {"log_group": {"log_group_arn": TEST_LOG_GROUP_ARN}}
        mock_nexus_client = mocker.patch("app.api.v1.routers.tool_logs.NexusClient").return_value
        mock_nexus_client.get_log_group.return_value = mock_nexus_response

        # Mock AWSLogsClient response
        mock_logs = [{"timestamp": datetime.now(UTC).timestamp() * 1000, "message": "Log message 1"}]
        mock_next_token = "next-page-token"
        mock_aws_client = mocker.patch("app.api.v1.routers.tool_logs.AWSLogsClient").return_value
        mock_aws_client.get_function_logs = mocker.AsyncMock(return_value=(mock_logs, mock_next_token))

        # Make request
        response = client.get(api_path, params=default_query_params, headers=common_headers)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "ok"
        assert len(response_data["logs"]) == 1
        # Need to compare timestamps carefully due to potential float precision
        assert abs(response_data["logs"][0]["timestamp"] - mock_logs[0]["timestamp"]) < 1
        assert response_data["logs"][0]["message"] == mock_logs[0]["message"]
        assert response_data["next_token"] == mock_next_token

        # Verify mocks were called
        mock_nexus_client.get_log_group.assert_called_once_with(TEST_AGENT_KEY, TEST_TOOL_KEY)
        mock_aws_client.get_function_logs.assert_called_once_with(
            TEST_LOG_GROUP_ARN,
            mocker.ANY,  # Check specific times if needed, requires parsing params
            mocker.ANY,
            None,  # default_query_params doesn't include pattern
            None,  # default_query_params doesn't include next_token
        )

    def test_get_logs_success_no_next_token(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        common_headers: dict[str, str],
        default_query_params: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test successful retrieval when AWS returns no next_token."""
        # Mock NexusClient response (same as success case)
        mock_nexus_response = mocker.Mock()
        mock_nexus_response.status_code = status.HTTP_200_OK
        mock_nexus_response.json.return_value = {"log_group": {"log_group_arn": TEST_LOG_GROUP_ARN}}
        mock_nexus_client = mocker.patch("app.api.v1.routers.tool_logs.NexusClient").return_value
        mock_nexus_client.get_log_group.return_value = mock_nexus_response

        # Mock AWSLogsClient response with None as next_token
        mock_logs = [{"timestamp": datetime.now(UTC).timestamp() * 1000, "message": "Log message 1"}]
        mock_aws_client = mocker.patch("app.api.v1.routers.tool_logs.AWSLogsClient").return_value
        mock_aws_client.get_function_logs = mocker.AsyncMock(return_value=(mock_logs, None)) # No next token

        # Make request
        response = client.get(api_path, params=default_query_params, headers=common_headers)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "ok"
        assert response_data["next_token"] is None # Check for None
        assert len(response_data["logs"]) == 1

    def test_get_logs_nexus_get_log_group_fails(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        common_headers: dict[str, str],
        default_query_params: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test failure when NexusClient fails to get the log group."""
        # Mock NexusClient response for failure
        error_message = "Nexus internal error"
        mock_nexus_response = mocker.Mock()
        mock_nexus_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_nexus_response.text = error_message
        mock_nexus_client = mocker.patch("app.api.v1.routers.tool_logs.NexusClient").return_value
        mock_nexus_client.get_log_group.return_value = mock_nexus_response

        mock_aws_client = mocker.patch("app.api.v1.routers.tool_logs.AWSLogsClient").return_value

        # Make request
        response = client.get(api_path, params=default_query_params, headers=common_headers)

        # Assertions
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert response_data["status"] == "error"
        assert error_message in response_data["message"]

        # Verify AWS client was NOT called
        mock_aws_client.get_function_logs.assert_not_called()

    def test_get_logs_nexus_response_missing_arn(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        common_headers: dict[str, str],
        default_query_params: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test failure when Nexus response is successful but missing the ARN."""
        # Mock NexusClient response (successful status, bad data)
        mock_nexus_response = mocker.Mock()
        mock_nexus_response.status_code = status.HTTP_200_OK
        mock_nexus_response.json.return_value = {"log_group": {"wrong_key": "some_value"}} # Missing log_group_arn
        mock_nexus_client = mocker.patch("app.api.v1.routers.tool_logs.NexusClient").return_value
        mock_nexus_client.get_log_group.return_value = mock_nexus_response

        mock_aws_client = mocker.patch("app.api.v1.routers.tool_logs.AWSLogsClient").return_value

        # Make request
        response = client.get(api_path, params=default_query_params, headers=common_headers)

        # Assertions
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "Log group not found in Nexus response" in response_data["message"]

        # Verify AWS client was NOT called
        mock_aws_client.get_function_logs.assert_not_called()


    def test_get_logs_aws_client_raises_exception(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        common_headers: dict[str, str],
        default_query_params: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test failure when AWSLogsClient raises an unexpected exception."""
        # Mock NexusClient response (success)
        mock_nexus_response = mocker.Mock()
        mock_nexus_response.status_code = status.HTTP_200_OK
        mock_nexus_response.json.return_value = {"log_group": {"log_group_arn": TEST_LOG_GROUP_ARN}}
        mock_nexus_client = mocker.patch("app.api.v1.routers.tool_logs.NexusClient").return_value
        mock_nexus_client.get_log_group.return_value = mock_nexus_response

        # Mock AWSLogsClient to raise an error
        error_message = "AWS Client Error"
        mock_aws_client = mocker.patch("app.api.v1.routers.tool_logs.AWSLogsClient").return_value
        mock_aws_client.get_function_logs = mocker.AsyncMock(side_effect=RuntimeError(error_message))

        # Make request - Expecting the exception to propagate out in the test environment
        with pytest.raises(RuntimeError) as exc_info:
            client.get(api_path, params=default_query_params, headers=common_headers)

        # Assert that the raised exception contains the expected message
        assert error_message in str(exc_info.value)

    def test_get_logs_with_pattern_and_token(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        common_headers: dict[str, str],
        default_query_params: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test successful retrieval using optional pattern and next_token."""
        # Mock NexusClient response
        mock_nexus_response = mocker.Mock()
        mock_nexus_response.status_code = status.HTTP_200_OK
        mock_nexus_response.json.return_value = {"log_group": {"log_group_arn": TEST_LOG_GROUP_ARN}}
        mock_nexus_client = mocker.patch("app.api.v1.routers.tool_logs.NexusClient").return_value
        mock_nexus_client.get_log_group.return_value = mock_nexus_response

        # Mock AWSLogsClient response
        mock_logs = [{"timestamp": datetime.now(UTC).timestamp() * 1000, "message": "Filtered Log"}]
        mock_next_token_response = "next-page-token-2"
        mock_aws_client = mocker.patch("app.api.v1.routers.tool_logs.AWSLogsClient").return_value
        mock_aws_client.get_function_logs = mocker.AsyncMock(return_value=(mock_logs, mock_next_token_response))

        # Add pattern and next_token to query params
        query_params = default_query_params.copy()
        test_pattern = "ERROR"
        test_token = "prev-page-token"
        query_params["pattern"] = test_pattern
        query_params["next_token"] = test_token


        # Make request
        response = client.get(api_path, params=query_params, headers=common_headers)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "ok"
        assert len(response_data["logs"]) == 1
        assert response_data["logs"][0]["message"] == "Filtered Log"
        assert response_data["next_token"] == mock_next_token_response

        # Verify mocks were called with pattern and token
        mock_aws_client.get_function_logs.assert_called_once_with(
            TEST_LOG_GROUP_ARN,
            mocker.ANY,
            mocker.ANY,
            test_pattern, # Check pattern was passed
            test_token,   # Check token was passed
        )
