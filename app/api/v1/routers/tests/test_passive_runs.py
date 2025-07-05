"""Tests for passive tool runs endpoint."""

import io
import json
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.core.config import settings
from app.main import app
from app.tests.utils import AsyncMock

# Common test constants
TEST_CONTENT = b"test content"
TEST_AGENT_NAME = "test-agent"
TEST_TOOL_NAME = "test-tool"
TEST_TOOL_KEY = "test_tool"
TEST_AGENT_KEY = "test_agent"
TEST_TOKEN = "Bearer test-token"
TEST_FUNCTION_NAME = "test-function-name"
TEST_FUNCTION_ARN = "arn:aws:lambda:us-east-1:123456789012:function:test-function-name"
TEST_START_TIME = 1000.0
TEST_END_TIME = 1030.0  # 30 seconds after start


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Return the API path for passive runs."""
    return f"{settings.API_PREFIX}/v1/passive_runs"


@pytest.fixture(scope="module")
def project_uuid() -> UUID:
    """Generate a random project UUID."""
    return UUID(str(uuid4()))


@pytest.fixture(scope="module")
def auth_header(project_uuid: UUID) -> dict[str, str]:
    """Return an authorization header."""
    return {
        "Authorization": TEST_TOKEN,
        "X-Project-Uuid": str(project_uuid),
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


@pytest.fixture
def run_tool_request_data() -> dict[str, Any]:
    """Return test data for run_passive_tool_test endpoint."""
    agent_definition = {
        "name": TEST_AGENT_NAME,
        "slug": "test-agent-slug",
        "tools": [
            {
                "key": TEST_TOOL_KEY,
                "name": TEST_TOOL_NAME,
                "slug": "test-tool-slug",
            }
        ],
    }

    test_definition = {
        "tests": {
            "test_case_1": {"parameters": {"input": "Hello world"}},
            "test_case_2": {"parameters": {"input": "Another test"}},
        }
    }

    return {
        "project_uuid": str(uuid4()),
        "definition": json.dumps({"agents": {TEST_AGENT_KEY: agent_definition}}),
        "test_definition": json.dumps(test_definition),
        "tool_key": TEST_TOOL_KEY,
        "agent_key": TEST_AGENT_KEY,
        "tool_credentials": json.dumps(
            {
                "credential-test-key": "test-value",
            }
        ),
        "tool_globals": json.dumps(
            {
                "global-test-key": "test-value",
            }
        ),
        "toolkit_version": "1.0.0",
    }


@pytest.fixture
def post_run_request_factory(
    client: TestClient, api_path: str, auth_header: dict[str, str], run_tool_request_data: dict[str, Any]
) -> Callable[[], Any]:
    """Return a factory function for making POST requests to the passive runs endpoint."""

    def make_post_request() -> Any:
        # Create files dictionary with the tool file
        files = {
            "tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        # Merge the data and files for the multipart request
        data = {**run_tool_request_data}

        headers = {**auth_header, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}

        return client.post(api_path, data=data, files=files, headers=headers)

    return make_post_request


@pytest.fixture
def custom_post_run_request_factory(
    client: TestClient,
    api_path: str,
    auth_header: dict[str, str],
) -> Callable[[dict[str, Any], dict[str, Any], dict[str, str] | None], Any]:
    """Return a factory function for making custom POST requests to the passive runs endpoint."""

    def make_custom_post_request(
        data_fields: dict[str, Any],
        files_fields: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        # Use default auth_header if no headers provided
        request_headers = headers if headers is not None else auth_header

        return client.post(api_path, data=data_fields, files=files_fields, headers=request_headers)

    return make_custom_post_request


def parse_streaming_response(response: Any) -> list[dict[str, Any]]:
    """Parse a streaming response into a list of JSON objects."""
    result = []
    for line in response.iter_lines():
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                pass
    return result


class TestPassiveRunToolEndpoint:
    """Tests for the run_passive_tool_test endpoint."""

    @pytest.fixture
    def mock_success_dependencies(self, mocker: MockerFixture) -> None:
        """Mock dependencies for successful run_passive_tool_test."""
        # Mock process_tool to return a successful result
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {
                "tool_key": TEST_TOOL_KEY,
            },
            "success": True,
            "code": "TOOL_PROCESSED",
        }

        mocker.patch(
            "app.api.v1.routers.passive_runs.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock AWS Lambda client
        mock_lambda_client = mocker.MagicMock()
        wait_for_function_active_mock = AsyncMock(return_value=True)
        mock_lambda_client.wait_for_function_active = wait_for_function_active_mock
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value={
                "FunctionName": TEST_FUNCTION_NAME,
                "FunctionArn": TEST_FUNCTION_ARN,
            }
        )
        mock_lambda_client.invoke_function = mocker.MagicMock(
            return_value=(
                {
                    "response": {"result": "success", "output": "Test output"},
                    "status_code": 200,
                    "logs": ["Log line 1", "Log line 2"],
                },
                TEST_START_TIME,
                TEST_END_TIME,
            )
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.passive_runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep to avoid delays in tests
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

    def test_run_passive_tool_success(
        self, post_run_request_factory: Callable[[], Any], mock_success_dependencies: None, mock_auth_middleware: None
    ) -> None:
        """Test successful run_passive_tool_test endpoint."""
        # Minimum expected response items
        min_expected_responses = 4  # initial, tool processed, lambda creating, completion/error

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for expected response objects
        assert (
            len(response_data) >= min_expected_responses
        )  # At least initial response, tool processed, lambda creating, completion/error

        # Check initial response
        assert response_data[0]["code"] == "PROCESSING_STARTED", "First message should have code=PROCESSING_STARTED"
        assert response_data[0]["success"] is True, "First message should have success=True"

        # Check for tool processed message
        tool_processed_msgs = [r for r in response_data if r.get("code") == "TOOL_PROCESSED"]
        assert len(tool_processed_msgs) > 0, "Should include a TOOL_PROCESSED message"

        # Check for lambda creating message
        lambda_creating_msgs = [r for r in response_data if r.get("code") == "LAMBDA_FUNCTION_CREATING"]
        assert len(lambda_creating_msgs) > 0, "Should include a LAMBDA_FUNCTION_CREATING message"

    @pytest.mark.parametrize(
        "test_id, data_fields, files_fields, headers, expected_status, expected_error_code",
        [
            (
                "missing_tool_file",
                {
                    "project_uuid": str(uuid4()),
                    "definition": json.dumps({"agents": {}}),
                    "test_definition": json.dumps({"tests": {}}),
                    "tool_key": TEST_TOOL_KEY,
                    "agent_key": TEST_AGENT_KEY,
                    "tool_credentials": json.dumps({}),
                    "tool_globals": json.dumps({}),
                    "toolkit_version": "1.0.0",
                },
                {},  # No tool file
                None,  # Use default auth header
                status.HTTP_400_BAD_REQUEST,
                None,  # No streaming response for 400
            ),
            (
                "missing_authorization",
                {
                    "project_uuid": str(uuid4()),
                    "definition": json.dumps({"agents": {}}),
                    "test_definition": json.dumps({"tests": {}}),
                    "tool_key": TEST_TOOL_KEY,
                    "agent_key": TEST_AGENT_KEY,
                    "tool_credentials": json.dumps({}),
                    "tool_globals": json.dumps({}),
                    "toolkit_version": "1.0.0",
                },
                {
                    "tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
                },
                {
                    "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
                },  # Empty headers - no auth
                status.HTTP_400_BAD_REQUEST,
                "Missing Authorization or X-Project-Uuid header",
            ),
        ],
    )
    def test_validation_errors(  # noqa: PLR0913
        self,
        custom_post_run_request_factory: Callable[[dict[str, Any], dict[str, Any], dict[str, str] | None], Any],
        test_id: str,
        data_fields: dict[str, Any],
        files_fields: dict[str, Any],
        headers: dict[str, str] | None,
        expected_status: int,
        expected_error_code: str | None,
        mock_auth_middleware: None,
    ) -> None:
        """Test validation errors for run_passive_tool_test endpoint."""
        # Execute
        response = custom_post_run_request_factory(data_fields, files_fields, headers)

        # Assert
        assert response.status_code == expected_status, f"Expected status {expected_status} for {test_id}"

    def test_process_tool_error(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test error handling when process_tool raises an exception."""
        # Setup
        error_message = "Error processing tool"
        mocker.patch("app.api.v1.routers.passive_runs.process_tool", new=AsyncMock(side_effect=ValueError(error_message)))

        # Mock AWS clients
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.passive_runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for initial response
        assert response_data[0]["code"] == "PROCESSING_STARTED"
        assert response_data[0]["success"] is True

        # Check for error response
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert error_message in str(error_responses[-1]) 