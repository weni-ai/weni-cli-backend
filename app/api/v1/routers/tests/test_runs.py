"""Tests for tool runs endpoint."""

import io
import json
import logging
import re
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.api.v1.project_binding import PROJECT_MISMATCH_CODE, PROJECT_MISMATCH_MESSAGE
from app.api.v1.run_attribution import RUN_NOT_ATTRIBUTABLE_CODE, RUN_NOT_ATTRIBUTABLE_MESSAGE
from app.clients.aws.lambda_client import LambdaFunction
from app.core.config import settings
from app.main import app
from app.services.jwt_generator import DEFAULT_EXPIRATION_MINUTES
from app.tests.utils import AsyncMock, generate_rsa_key_pair, make_cli_bearer_token

# Common test constants
TEST_CONTENT = b"test content"
TEST_AGENT_NAME = "test-agent"
TEST_TOOL_NAME = "test-tool"
TEST_TOOL_KEY = "test_tool"
TEST_AGENT_KEY = "test_agent"
TEST_PROJECT_UUID = "c67bc61e-c2b2-43f1-a409-88dec4bd4b9e"
TEST_USER_EMAIL = "dev@example.com"
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
    """Return the API path for runs."""
    return f"{settings.API_PREFIX}/v1/runs"


@pytest.fixture(scope="module")
def project_uuid() -> UUID:
    """Return a test project UUID."""
    return UUID(TEST_PROJECT_UUID)


@pytest.fixture(scope="module")
def auth_header(project_uuid: UUID) -> dict[str, str]:
    """Return an authorization header for tests."""
    return {
        "Authorization": make_cli_bearer_token(TEST_USER_EMAIL),
        "X-Project-Uuid": str(project_uuid),
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


@pytest.fixture
def run_tool_request_data(project_uuid: UUID) -> dict[str, Any]:
    """Return test data for run_tool_test endpoint."""
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
        "project_uuid": str(project_uuid),
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
def test_log_events() -> list[dict[str, Any]]:
    """Return test log events."""
    return [
        {"timestamp": 1001, "message": "START RequestId: test-request-id", "logStreamName": "stream1"},
        {"timestamp": 1002, "message": "Processing request", "logStreamName": "stream1"},
        {"timestamp": 1003, "message": "END RequestId: test-request-id", "logStreamName": "stream1"},
    ]


@pytest.fixture
def post_run_request_factory(
    client: TestClient, api_path: str, auth_header: dict[str, str], run_tool_request_data: dict[str, Any]
) -> Callable[[], Any]:
    """Return a factory function for making POST requests to the runs endpoint."""

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
    """Return a factory function for making custom POST requests to the runs endpoint."""

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


class TestRunToolEndpoint:
    """Tests for the run_tool_test endpoint."""

    @pytest.fixture
    def mock_success_dependencies(self, mocker: MockerFixture) -> None:
        """Mock dependencies for successful run_tool_test."""
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
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock AWS Lambda client
        mock_lambda_client = mocker.MagicMock()
        wait_for_function_active_mock = AsyncMock(return_value=True)
        mock_lambda_client.wait_for_function_active = wait_for_function_active_mock
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value=LambdaFunction(
                arn=TEST_FUNCTION_ARN,
                name=TEST_FUNCTION_NAME,
                log_group="test-log-group",
            )
        )
        mock_lambda_client.invoke_function = mocker.MagicMock(
            return_value=(
                {"response": {"result": "success"}, "status_code": 200, "logs": ""},
                TEST_START_TIME,
                TEST_END_TIME,
            )
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep to avoid delays in tests
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

    def test_run_tool_success(
        self, post_run_request_factory: Callable[[], Any], mock_success_dependencies: None, mock_auth_middleware: None
    ) -> None:
        """Test successful run_tool_test endpoint."""
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
                    "project_uuid": TEST_PROJECT_UUID,
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
        """Test validation errors for run_tool_test endpoint."""
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
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(side_effect=ValueError(error_message)),
        )

        # Mock AWS clients
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

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

    def test_lambda_function_creation_failure(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test error handling when lambda function creation fails."""
        # Setup - mock process_tool to succeed
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {
                "tool_key": TEST_TOOL_KEY,
            },
            "success": True,
            "code": "TOOL_PROCESSED",
        }
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock Lambda client to fail function creation
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.create_function = mocker.MagicMock(side_effect=ValueError("Function creation failed"))
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

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

    def test_lambda_function_activation_failure(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test error handling when lambda function activation fails."""
        # Setup - mock process_tool to succeed
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {
                "tool_key": TEST_TOOL_KEY,
            },
            "success": True,
            "code": "TOOL_PROCESSED",
        }
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock Lambda client
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value={
                "FunctionName": TEST_FUNCTION_NAME,
                "FunctionArn": TEST_FUNCTION_ARN,
            }
        )
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=False)
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

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

    def test_lambda_function_invocation_error(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test error handling when lambda function invocation fails."""
        # Setup - mock process_tool to succeed
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {
                "tool_key": TEST_TOOL_KEY,
            },
            "success": True,
            "code": "TOOL_PROCESSED",
        }
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock Lambda client
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value={
                "FunctionName": TEST_FUNCTION_NAME,
                "FunctionArn": TEST_FUNCTION_ARN,
            }
        )
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=True)
        mock_lambda_client.invoke_function = mocker.MagicMock(
            side_effect=ValueError("Lambda function invocation failed")
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

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

    def test_clean_up_on_error(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test that resources are cleaned up when an error occurs."""
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool", new=AsyncMock(side_effect=ValueError("Process tool error"))
        )

        # When the Lambda client is created early in the code, make lambda_function_name available
        # for the delete_function call to find
        function_name = f"cli-{uuid4()}"

        # Mock Lambda client with a synchronous (non-async) delete_function method
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value={
                "FunctionName": function_name,
                "FunctionArn": f"arn:aws:lambda:us-east-1:123456789012:function:{function_name}",
            }
        )
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Execute
        response = post_run_request_factory()

        # Assert status code
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0

        # Assert that delete_function was called to clean up resources
        mock_lambda_client.delete_function.assert_called_once()

    def test_agent_not_found_in_definition(
        self,
        post_run_request_factory: Callable[[], Any],
        mocker: MockerFixture,
        run_tool_request_data: dict[str, Any],
        auth_header: dict[str, str],
        mock_auth_middleware: None,
    ) -> None:
        """Test error handling when agent is not found in definition."""
        # Setup - Create a definition with no agents
        empty_definition: dict[str, dict] = {"agents": {}}

        # Setup mocks
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Modify the request data
        modified_data = run_tool_request_data.copy()
        modified_data["definition"] = json.dumps(empty_definition)

        # Make the request
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/runs"

        files = {
            "tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        response = client.post(
            api_path,
            data=modified_data,
            files=files,
            headers=auth_header,
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error message
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert f"Could not find agent {TEST_AGENT_KEY}" in str(error_responses[-1])

    def test_tool_not_found_for_agent(
        self,
        post_run_request_factory: Callable[[], Any],
        mocker: MockerFixture,
        run_tool_request_data: dict[str, Any],
        auth_header: dict[str, str],
        mock_auth_middleware: None,
    ) -> None:
        """Test error handling when tool is not found for agent."""
        # Setup - Create a definition with an agent but no tools
        agent_without_tools = {
            "name": TEST_AGENT_NAME,
            "slug": "test-agent-slug",
            "tools": [],  # Empty tools list
        }
        definition = {"agents": {TEST_AGENT_KEY: agent_without_tools}}

        # Setup mocks
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Modify the request data
        modified_data = run_tool_request_data.copy()
        modified_data["definition"] = json.dumps(definition)

        # Make the request
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/runs"

        files = {
            "tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        response = client.post(
            api_path,
            data=modified_data,
            files=files,
            headers=auth_header,
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error message
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert f"Could not find tool {TEST_TOOL_KEY}" in str(error_responses[-1])

    def test_jwt_always_injected_in_credentials(
        self,
        post_run_request_factory: Callable[[], Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """Test that JWT token is always injected into credentials."""
        # Setup - mock process_tool to succeed
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {"tool_key": TEST_TOOL_KEY},
            "success": True,
            "code": "TOOL_PROCESSED",
        }
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock Lambda client
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value=LambdaFunction(
                arn=TEST_FUNCTION_ARN,
                name=TEST_FUNCTION_NAME,
                log_group="test-log-group",
            )
        )
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=True)
        mock_lambda_client.invoke_function = mocker.MagicMock(
            return_value=(
                {"response": {"result": "success"}, "status_code": 200, "logs": ""},
                TEST_START_TIME,
                TEST_END_TIME,
            )
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Mock generate_jwt_token to return a predictable token
        mocker.patch(
            "app.services.runs.token_issuer.generate_jwt_token",
            return_value="mocked-jwt-token",
        )

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Verify invoke_function was called with project containing the auth_token
        invoke_calls = mock_lambda_client.invoke_function.call_args_list
        assert len(invoke_calls) > 0

        for call in invoke_calls:
            test_event = call[0][1]  # second positional arg is the event
            project = json.loads(test_event["sessionAttributes"]["project"])
            assert "auth_token" in project, "JWT Token should always be injected into project"
            assert project["auth_token"] == "mocked-jwt-token"

    def test_empty_tool_zip_bytes(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test error handling when tool_zip_bytes is empty after processing."""
        # Setup - mock process_tool to return None for tool_zip_bytes
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {
                "tool_key": TEST_TOOL_KEY,
            },
            "success": True,
            "code": "TOOL_PROCESSED",
        }
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(
                return_value=(mock_process_result, None)  # Return None for tool_zip_bytes
            ),
        )

        # Mock Lambda client
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error message
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert "Failed to process tool" in str(error_responses[-1])


# Common constants for active agent tests
TEST_ACTIVE_AGENT_KEY = "payment_agent_pix_recovery"
TEST_RULE_KEY = "PaymentRecovery"


@pytest.fixture
def active_agent_definition() -> dict[str, Any]:
    """Return a minimal valid active agent definition."""
    return {
        "agents": {
            TEST_ACTIVE_AGENT_KEY: {
                "name": "Whatsapp Payment Recovery",
                "description": "Recovers incomplete PIX orders",
                "language": "pt_BR",
                "rules": {
                    TEST_RULE_KEY: {
                        "display_name": "Payment Recovery",
                        "template": "payment_recovery",
                        "start_condition": "incomplete order without PIX",
                        "source": {
                            "entrypoint": "main.PaymentRecovery",
                            "path": "rules/payment_recovery",
                        },
                        "example": {"input": {}, "output": {}},
                    },
                },
                "pre_processing": {
                    "source": {
                        "entrypoint": "processing.PreProcessor",
                        "path": "pre_processors/processor",
                    },
                    "result_examples_file": "result_example.json",
                },
            },
        },
    }


@pytest.fixture
def run_active_request_data(active_agent_definition: dict[str, Any], project_uuid: UUID) -> dict[str, Any]:
    """Return form data for an active agent run request."""
    test_definition = {
        "tests": {
            "pix_pending": {
                "payload": {"OrderId": "1621590779140-01", "State": "payment-pending"},
                "params": {},
                "credentials": {},
                "project": {
                    "uuid": "6f8d2b1e-4a3c-4f5e-9b8d-1234567890ab",
                    "vtex_account": "minhaloja",
                    "country_phone_code": "55",
                },
            },
        }
    }

    return {
        "project_uuid": str(project_uuid),
        "definition": json.dumps(active_agent_definition),
        "test_definition": json.dumps(test_definition),
        "agent_key": TEST_ACTIVE_AGENT_KEY,
        "type": "active",
        "toolkit_version": "1.0.0",
    }


@pytest.fixture
def post_active_run_request_factory(
    client: TestClient,
    api_path: str,
    auth_header: dict[str, str],
    run_active_request_data: dict[str, Any],
) -> Callable[..., Any]:
    """Return a factory function for POSTing active agent run requests."""

    def make_post_request(extra_files: dict[str, Any] | None = None) -> Any:
        files = {
            f"{TEST_ACTIVE_AGENT_KEY}:preprocessor_folder": (
                "preprocessor.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
            f"{TEST_ACTIVE_AGENT_KEY}:{TEST_RULE_KEY}": (
                "rule.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
        }
        if extra_files:
            files.update(extra_files)

        headers = {**auth_header, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}
        return client.post(api_path, data=run_active_request_data, files=files, headers=headers)

    return make_post_request


class TestRunActiveAgentEndpoint:
    """Tests for the active agent run flow."""

    @pytest.fixture
    def mock_active_success_dependencies(self, mocker: MockerFixture) -> None:
        """Mock dependencies for a successful active agent run."""
        mock_processor = mocker.MagicMock()
        mock_processor.process = mocker.MagicMock(return_value=io.BytesIO(b"fake-active-zip-bytes"))
        mocker.patch(
            "app.services.runs.active_strategy.ActiveAgentProcessor",
            return_value=mock_processor,
        )

        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=True)
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value=LambdaFunction(
                arn=TEST_FUNCTION_ARN,
                name=TEST_FUNCTION_NAME,
                log_group="test-log-group",
            )
        )
        mock_lambda_client.invoke_function = mocker.MagicMock(
            return_value=(
                {
                    "response": {
                        "status": 0,
                        "template": "payment_recovery",
                        "template_variables": {"1": "Maria"},
                        "contact_urn": "whatsapp:5511999999999",
                        "error": {},
                        "traces": {"preprocessor": {}, "project_rule": {}, "official_rule": {}},
                    },
                    "status_code": 200,
                    "logs": "",
                },
                TEST_START_TIME,
                TEST_END_TIME,
            )
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        mocker.patch(
            "app.services.runs.token_issuer.generate_jwt_token",
            return_value="mocked-jwt-token",
        )

    def test_active_run_success(
        self,
        post_active_run_request_factory: Callable[..., Any],
        mock_active_success_dependencies: None,
        mock_auth_middleware: None,
    ) -> None:
        """Active agent run produces full progression of status codes and a TEST_CASE_COMPLETED."""
        response = post_active_run_request_factory()

        assert response.status_code == status.HTTP_200_OK

        response_data = parse_streaming_response(response)

        codes = [r.get("code") for r in response_data]
        assert "PROCESSING_STARTED" in codes
        assert "ACTIVE_AGENT_PROCESSED" in codes
        assert "LAMBDA_FUNCTION_CREATING" in codes
        assert "STARTING_TEST_CASES" in codes
        assert "TEST_CASE_RUNNING" in codes
        assert "TEST_CASE_COMPLETED" in codes

        completed = next(r for r in response_data if r.get("code") == "TEST_CASE_COMPLETED")
        assert completed["data"]["test_case"] == "pix_pending"
        assert completed["data"]["test_status_code"] == status.HTTP_200_OK
        assert completed["data"]["test_response"]["template"] == "payment_recovery"

    def test_active_run_jwt_injected_in_project(
        self,
        post_active_run_request_factory: Callable[..., Any],
        mock_active_success_dependencies: None,
        mock_auth_middleware: None,
        mocker: MockerFixture,
    ) -> None:
        """JWT token must be injected into project.auth_token for the active flow."""
        response = post_active_run_request_factory()
        assert response.status_code == status.HTTP_200_OK

        from app.api.v1.routers import runs as runs_module

        invoke_calls = runs_module.AWSLambdaClient.return_value.invoke_function.call_args_list  # type: ignore
        assert len(invoke_calls) > 0

        for call in invoke_calls:
            test_event = call[0][1]
            assert "auth_token" in test_event["project"]
            assert test_event["project"]["auth_token"] == "mocked-jwt-token"
            assert "payload" in test_event
            assert "params" in test_event
            assert "credentials" in test_event
            assert "project_rules" in test_event
            assert "ignored_official_rules" in test_event
            assert "global_rule" in test_event

    def test_active_run_missing_resources(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_active_request_data: dict[str, Any],
        mock_auth_middleware: None,
    ) -> None:
        """Active run with no multipart resources should return 400."""
        headers = {**auth_header, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}
        response = client.post(api_path, data=run_active_request_data, headers=headers)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_active_run_missing_preprocessor(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_active_request_data: dict[str, Any],
        mock_auth_middleware: None,
        mocker: MockerFixture,
    ) -> None:
        """Sending only a rule (no preprocessor) should surface an error in the stream."""
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        files = {
            f"{TEST_ACTIVE_AGENT_KEY}:{TEST_RULE_KEY}": (
                "rule.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
        }
        headers = {**auth_header, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}
        response = client.post(api_path, data=run_active_request_data, files=files, headers=headers)

        assert response.status_code == status.HTTP_200_OK

        response_data = parse_streaming_response(response)
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert "Preprocessor" in str(error_responses[-1])

    def test_active_run_processor_error(
        self,
        post_active_run_request_factory: Callable[..., Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
    ) -> None:
        """If ActiveAgentProcessor raises, the lambda must be cleaned up."""
        mocker.patch(
            "app.services.runs.active_strategy.ActiveAgentProcessor",
            side_effect=ValueError("Processor failed"),
        )

        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        response = post_active_run_request_factory()

        assert response.status_code == status.HTTP_200_OK

        response_data = parse_streaming_response(response)
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert "Processor failed" in str(error_responses[-1])

        mock_lambda_client.delete_function.assert_called_once()


_MISMATCH_PROJECT_UUID = "6f1c2c1e-8b7a-4d3e-9c2b-0a1b2c3d4e5f"
_QUOTED_FIELD = re.compile(r'([a-z_]+)=("(?:\\.|[^"\\])*")')


def _quoted_fields(message: str) -> list[tuple[str, str]]:
    return [(key, json.loads(value)) for key, value in _QUOTED_FIELD.findall(message)]


def _mismatch_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.message.startswith("event=project_mismatch_rejected")
    ]


def _assert_bearer_absent(caplog: pytest.LogCaptureFixture, authorization: str) -> None:
    assert authorization not in caplog.text
    assert authorization.removeprefix("Bearer ") not in caplog.text


def _assert_no_run_token_minted(caplog: pytest.LogCaptureFixture) -> None:
    assert not any(record.message.startswith("event=run_token_minted") for record in caplog.records)


class TestRunProjectBinding:
    @pytest.fixture(autouse=True)
    def downstream(self, mocker: MockerFixture, mock_auth_middleware: None) -> dict[str, Any]:
        process_tool = AsyncMock()
        mint = mocker.patch("app.services.runs.token_issuer.generate_jwt_token")
        return {
            "process_tool": process_tool,
            "lambda_client": mocker.patch("app.api.v1.routers.runs.AWSLambdaClient"),
            "processor": mocker.patch("app.services.runs.active_strategy.ActiveAgentProcessor"),
            "tool_mint": mint,
            "active_mint": mint,
            "process_tool_patch": mocker.patch("app.services.runs.tool_strategy.process_tool", new=process_tool),
        }

    def test_tool_run_rejects_mismatched_project(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        downstream: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {**run_tool_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=auth_header)

        self._assert_mismatch_response(response, TEST_PROJECT_UUID, _MISMATCH_PROJECT_UUID)
        downstream["lambda_client"].assert_not_called()
        assert downstream["process_tool"].call_count == 0
        downstream["tool_mint"].assert_not_called()
        downstream["active_mint"].assert_not_called()
        assert len(_mismatch_warnings(caplog)) == 1
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def test_active_run_rejects_mismatched_project(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_active_request_data: dict[str, Any],
        downstream: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {**run_active_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}
        files = {
            f"{TEST_ACTIVE_AGENT_KEY}:preprocessor_folder": (
                "preprocessor.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
            f"{TEST_ACTIVE_AGENT_KEY}:{TEST_RULE_KEY}": (
                "rule.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
        }

        response = client.post(api_path, data=body, files=files, headers=auth_header)

        self._assert_mismatch_response(response, TEST_PROJECT_UUID, _MISMATCH_PROJECT_UUID)
        downstream["lambda_client"].assert_not_called()
        assert downstream["process_tool"].call_count == 0
        downstream["processor"].assert_not_called()
        downstream["tool_mint"].assert_not_called()
        downstream["active_mint"].assert_not_called()
        assert len(_mismatch_warnings(caplog)) == 1
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def test_upper_case_body_is_a_mismatch(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {**run_tool_request_data, "project_uuid": TEST_PROJECT_UUID.upper()}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=auth_header)

        self._assert_mismatch_response(response, TEST_PROJECT_UUID, TEST_PROJECT_UUID.upper())
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def test_mismatch_event_fields(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {**run_tool_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=auth_header)

        payload = response.json()
        fields = dict(_quoted_fields(_mismatch_warnings(caplog)[0].message))
        assert list(fields) == [
            "header_project_uuid",
            "body_project_uuid",
            "endpoint",
            "request_id",
            "user_email",
        ]
        assert fields["header_project_uuid"] == TEST_PROJECT_UUID
        assert fields["body_project_uuid"] == _MISMATCH_PROJECT_UUID
        assert fields["endpoint"] == api_path
        assert fields["request_id"] == payload["request_id"]
        assert fields["user_email"] == TEST_USER_EMAIL
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def test_mismatch_event_omits_email_when_token_has_none(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        authorization = make_cli_bearer_token(None)
        headers = {**auth_header, "Authorization": authorization}
        body = {**run_tool_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=headers)

        self._assert_mismatch_response(response, TEST_PROJECT_UUID, _MISMATCH_PROJECT_UUID)
        fields = dict(_quoted_fields(_mismatch_warnings(caplog)[0].message))
        assert "user_email" not in fields
        _assert_bearer_absent(caplog, authorization)
        _assert_no_run_token_minted(caplog)

    @pytest.mark.parametrize(
        "body_project_uuid",
        [
            pytest.param("not-a-uuid", id="not-a-uuid"),
            pytest.param(None, id="missing"),
        ],
    )
    def test_invalid_or_missing_body_project_returns_422(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        body_project_uuid: str | None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {key: value for key, value in run_tool_request_data.items() if key != "project_uuid"}
        if body_project_uuid is not None:
            body["project_uuid"] = body_project_uuid
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=auth_header)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert _mismatch_warnings(caplog) == []
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def test_tool_mismatch_without_tool_file_is_403(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        downstream: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {**run_tool_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}

        response = client.post(api_path, data=body, headers=auth_header)

        self._assert_mismatch_response(response, TEST_PROJECT_UUID, _MISMATCH_PROJECT_UUID)
        downstream["lambda_client"].assert_not_called()
        assert downstream["process_tool"].call_count == 0
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def test_active_mismatch_without_resources_is_403(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_active_request_data: dict[str, Any],
        downstream: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {**run_active_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}

        response = client.post(api_path, data=body, headers=auth_header)

        self._assert_mismatch_response(response, TEST_PROJECT_UUID, _MISMATCH_PROJECT_UUID)
        downstream["processor"].assert_not_called()
        downstream["lambda_client"].assert_not_called()
        _assert_bearer_absent(caplog, auth_header["Authorization"])
        _assert_no_run_token_minted(caplog)

    def _assert_mismatch_response(self, response: Any, header_project: str, body_project: str) -> None:
        assert response.status_code == status.HTTP_403_FORBIDDEN
        payload = response.json()
        assert payload["message"] == PROJECT_MISMATCH_MESSAGE
        assert payload["data"] is None
        assert payload["success"] is False
        assert payload["code"] == PROJECT_MISMATCH_CODE
        assert str(UUID(payload["request_id"])) == payload["request_id"]
        assert set(payload) == {"message", "data", "success", "code", "request_id"}
        assert header_project not in response.text
        assert body_project not in response.text


def _not_attributable_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.message.startswith("event=run_not_attributable")
    ]


class TestRunAttribution:
    @pytest.fixture(autouse=True)
    def downstream(self, mocker: MockerFixture, mock_auth_middleware: None) -> dict[str, Any]:
        process_tool = AsyncMock()
        mint = mocker.patch("app.services.runs.token_issuer.generate_jwt_token")
        return {
            "process_tool": process_tool,
            "lambda_client": mocker.patch("app.api.v1.routers.runs.AWSLambdaClient"),
            "processor": mocker.patch("app.services.runs.active_strategy.ActiveAgentProcessor"),
            "tool_mint": mint,
            "active_mint": mint,
            "process_tool_patch": mocker.patch("app.services.runs.tool_strategy.process_tool", new=process_tool),
        }

    @pytest.mark.parametrize(
        "authorization",
        [
            pytest.param("Bearer test-token", id="not-a-jwt"),
            pytest.param(make_cli_bearer_token(None), id="token-without-email"),
            pytest.param(make_cli_bearer_token(""), id="empty-email"),
        ],
    )
    def test_tool_run_without_identity_is_403(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        downstream: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
        authorization: str,
    ) -> None:
        caplog.set_level(logging.INFO)
        headers = {**auth_header, "Authorization": authorization}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=run_tool_request_data, files=files, headers=headers)

        self._assert_not_attributable_response(response)
        downstream["lambda_client"].assert_not_called()
        assert downstream["process_tool"].call_count == 0
        downstream["tool_mint"].assert_not_called()
        downstream["active_mint"].assert_not_called()
        warnings = _not_attributable_warnings(caplog)
        assert len(warnings) == 1
        assert dict(_quoted_fields(warnings[0].message))["request_id"] == response.json()["request_id"]
        _assert_bearer_absent(caplog, authorization)
        _assert_no_run_token_minted(caplog)

    def test_active_run_with_auth_tokens_and_no_email_is_403(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_active_request_data: dict[str, Any],
        downstream: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        authorization = make_cli_bearer_token(None)
        test_definition = json.loads(run_active_request_data["test_definition"])
        for test_case in test_definition["tests"].values():
            test_case["project"]["auth_token"] = "user-supplied-token"
        body = {**run_active_request_data, "test_definition": json.dumps(test_definition)}
        headers = {**auth_header, "Authorization": authorization}
        files = {
            f"{TEST_ACTIVE_AGENT_KEY}:preprocessor_folder": (
                "preprocessor.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
            f"{TEST_ACTIVE_AGENT_KEY}:{TEST_RULE_KEY}": (
                "rule.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
        }

        response = client.post(api_path, data=body, files=files, headers=headers)

        self._assert_not_attributable_response(response)
        downstream["processor"].assert_not_called()
        assert len(_not_attributable_warnings(caplog)) == 1
        _assert_bearer_absent(caplog, authorization)
        _assert_no_run_token_minted(caplog)

    def test_tool_run_with_zero_test_cases_and_no_email_is_403(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        authorization = make_cli_bearer_token(None)
        body = {**run_tool_request_data, "test_definition": json.dumps({"tests": {}})}
        headers = {**auth_header, "Authorization": authorization}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=headers)

        self._assert_not_attributable_response(response)
        assert len(_not_attributable_warnings(caplog)) == 1
        _assert_bearer_absent(caplog, authorization)
        _assert_no_run_token_minted(caplog)

    def test_mismatch_wins_over_missing_identity(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        authorization = "Bearer test-token"
        headers = {**auth_header, "Authorization": authorization}
        body = {**run_tool_request_data, "project_uuid": _MISMATCH_PROJECT_UUID}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=headers)

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json()["code"] == PROJECT_MISMATCH_CODE
        assert _not_attributable_warnings(caplog) == []
        fields = dict(_quoted_fields(_mismatch_warnings(caplog)[0].message))
        assert "user_email" not in fields
        _assert_bearer_absent(caplog, authorization)
        _assert_no_run_token_minted(caplog)

    def test_invalid_body_project_wins_over_missing_identity(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        authorization = "Bearer test-token"
        headers = {**auth_header, "Authorization": authorization}
        body = {**run_tool_request_data, "project_uuid": "not-a-uuid"}
        files = {"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")}

        response = client.post(api_path, data=body, files=files, headers=headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert _mismatch_warnings(caplog) == []
        assert _not_attributable_warnings(caplog) == []
        _assert_bearer_absent(caplog, authorization)
        _assert_no_run_token_minted(caplog)

    def _assert_not_attributable_response(self, response: Any) -> None:
        assert response.status_code == status.HTTP_403_FORBIDDEN
        payload = response.json()
        assert payload["message"] == RUN_NOT_ATTRIBUTABLE_MESSAGE
        assert payload["data"] is None
        assert payload["success"] is False
        assert payload["code"] == RUN_NOT_ATTRIBUTABLE_CODE
        assert str(UUID(payload["request_id"])) == payload["request_id"]
        assert set(payload) == {"message", "data", "success", "code", "request_id"}


_DEFINITION_PROJECT_UUID = "6f8d2b1e-4a3c-4f5e-9b8d-1234567890ab"
_USER_SUPPLIED_TOKEN = "user-supplied-token"


def _assert_minted_tokens_absent(caplog: pytest.LogCaptureFixture, authorization: str, tokens: list[str]) -> None:
    _assert_bearer_absent(caplog, authorization)
    for token in tokens:
        assert token not in caplog.text
        for part in token.split("."):
            assert part not in caplog.text


def _decode_run_token(token: str, public_pem: str) -> dict[str, Any]:
    decoded = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert set(decoded) == {"project_uuid", "exp", "iat"}
    assert decoded["exp"] - decoded["iat"] == DEFAULT_EXPIRATION_MINUTES * 60
    return decoded


def _minted_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.message.startswith("event=run_token_minted")
    ]


class TestRunTokenMinting:
    @pytest.fixture(autouse=True)
    def public_pem(self, mocker: MockerFixture, mock_auth_middleware: None) -> str:
        private_pem, public_pem = generate_rsa_key_pair()
        mocker.patch.object(settings, "JWT_SECRET_KEY", private_pem)
        return public_pem

    def _mock_tool_lambda(self, mocker: MockerFixture) -> Any:
        mock_process_result = {
            "message": "Tool processed successfully",
            "data": {"tool_key": TEST_TOOL_KEY},
            "success": True,
            "code": "TOOL_PROCESSED",
        }
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.create_function = mocker.MagicMock(
            return_value=LambdaFunction(
                arn=TEST_FUNCTION_ARN,
                name=TEST_FUNCTION_NAME,
                log_group="test-log-group",
            )
        )
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=True)
        mock_lambda_client.invoke_function = mocker.MagicMock(
            return_value=(
                {"response": {"result": "success"}, "status_code": 200, "logs": ""},
                TEST_START_TIME,
                TEST_END_TIME,
            )
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))
        return mock_lambda_client

    def _mock_active_lambda(self, mocker: MockerFixture) -> Any:
        mock_processor = mocker.MagicMock()
        mock_processor.process = mocker.MagicMock(return_value=io.BytesIO(b"fake-active-zip-bytes"))
        mocker.patch("app.services.runs.active_strategy.ActiveAgentProcessor", return_value=mock_processor)
        return self._mock_tool_lambda(mocker)

    def test_tool_run_mints_one_token_per_test_case(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        public_pem: str,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        lambda_client = self._mock_tool_lambda(mocker)

        response = client.post(
            api_path,
            data=run_tool_request_data,
            files={"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
            headers=auth_header,
        )

        assert response.status_code == status.HTTP_200_OK
        messages = parse_streaming_response(response)
        test_cases = json.loads(run_tool_request_data["test_definition"])["tests"]
        completed = [message for message in messages if message.get("code") == "TEST_CASE_COMPLETED"]
        assert len(completed) == len(test_cases)
        request_id = messages[0]["request_id"]
        assert all(message["request_id"] == request_id for message in messages)

        tokens = []
        for call in lambda_client.invoke_function.call_args_list:
            project = json.loads(call[0][1]["sessionAttributes"]["project"])
            tokens.append(project["auth_token"])
        assert len(tokens) == len(test_cases)
        decoded_projects = []
        for token in tokens:
            decoded = _decode_run_token(token, public_pem)
            assert decoded["project_uuid"] == TEST_PROJECT_UUID
            decoded_projects.append(decoded["project_uuid"])

        records = _minted_records(caplog)
        assert len(records) == len(test_cases)
        for record, project_uuid in zip(records, decoded_projects, strict=True):
            fields = dict(_quoted_fields(record.message))
            assert list(fields) == [
                "user_email",
                "project_uuid",
                "agent_key",
                "tool_key",
                "run_type",
                "request_id",
            ]
            assert fields["user_email"] == TEST_USER_EMAIL
            assert fields["project_uuid"] == project_uuid
            assert fields["agent_key"] == TEST_AGENT_KEY
            assert fields["tool_key"] == TEST_TOOL_KEY
            assert fields["run_type"] == "passive"
            assert fields["request_id"] == request_id
        _assert_minted_tokens_absent(caplog, auth_header["Authorization"], tokens)

    def test_upper_case_header_mints_the_canonical_project(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        public_pem: str,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        lambda_client = self._mock_tool_lambda(mocker)
        headers = {**auth_header, "X-Project-Uuid": TEST_PROJECT_UUID.upper()}
        body = {**run_tool_request_data, "project_uuid": TEST_PROJECT_UUID.upper()}

        response = client.post(
            api_path,
            data=body,
            files={"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
            headers=headers,
        )

        assert response.status_code == status.HTTP_200_OK
        tokens = [
            json.loads(call[0][1]["sessionAttributes"]["project"])["auth_token"]
            for call in lambda_client.invoke_function.call_args_list
        ]
        assert tokens
        for token in tokens:
            assert _decode_run_token(token, public_pem)["project_uuid"] == TEST_PROJECT_UUID
        for record in _minted_records(caplog):
            assert dict(_quoted_fields(record.message))["project_uuid"] == TEST_PROJECT_UUID
        _assert_minted_tokens_absent(caplog, headers["Authorization"], tokens)

    def test_active_run_passes_through_supplied_token_and_mints_the_other(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_active_request_data: dict[str, Any],
        public_pem: str,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        lambda_client = self._mock_active_lambda(mocker)
        test_definition = {
            "tests": {
                "with_token": {
                    "payload": {},
                    "project": {"uuid": _DEFINITION_PROJECT_UUID, "auth_token": _USER_SUPPLIED_TOKEN},
                },
                "without_token": {
                    "payload": {},
                    "project": {"uuid": _DEFINITION_PROJECT_UUID},
                },
            }
        }
        body = {**run_active_request_data, "test_definition": json.dumps(test_definition)}
        files = {
            f"{TEST_ACTIVE_AGENT_KEY}:preprocessor_folder": (
                "preprocessor.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
            f"{TEST_ACTIVE_AGENT_KEY}:{TEST_RULE_KEY}": (
                "rule.zip",
                io.BytesIO(TEST_CONTENT),
                "application/zip",
            ),
        }

        response = client.post(api_path, data=body, files=files, headers=auth_header)

        assert response.status_code == status.HTTP_200_OK
        projects = [call[0][1]["project"] for call in lambda_client.invoke_function.call_args_list]
        assert projects[0]["auth_token"] == _USER_SUPPLIED_TOKEN
        minted = projects[1]["auth_token"]
        decoded = _decode_run_token(minted, public_pem)
        assert decoded["project_uuid"] == TEST_PROJECT_UUID
        assert decoded["project_uuid"] != _DEFINITION_PROJECT_UUID

        records = _minted_records(caplog)
        assert len(records) == 1
        fields = dict(_quoted_fields(records[0].message))
        assert "tool_key" not in fields
        assert fields["run_type"] == "active"
        assert fields["project_uuid"] == decoded["project_uuid"]
        _assert_minted_tokens_absent(caplog, auth_header["Authorization"], [minted])

    def test_zero_test_cases_mint_nothing(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        self._mock_tool_lambda(mocker)
        body = {**run_tool_request_data, "test_definition": json.dumps({"tests": {}})}

        response = client.post(
            api_path,
            data=body,
            files={"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
            headers=auth_header,
        )

        assert response.status_code == status.HTTP_200_OK
        assert _minted_records(caplog) == []
        _assert_minted_tokens_absent(caplog, auth_header["Authorization"], [])

    def test_invoke_failure_keeps_the_token_minted_before_it(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        lambda_client = self._mock_tool_lambda(mocker)
        lambda_client.invoke_function.side_effect = RuntimeError("invoke failed")

        response = client.post(
            api_path,
            data=run_tool_request_data,
            files={"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
            headers=auth_header,
        )

        assert response.status_code == status.HTTP_200_OK
        assert len(_minted_records(caplog)) == 1
        token = json.loads(lambda_client.invoke_function.call_args_list[0][0][1]["sessionAttributes"]["project"])[
            "auth_token"
        ]
        _assert_minted_tokens_absent(caplog, auth_header["Authorization"], [token])

    def test_process_tool_failure_mints_nothing(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        run_tool_request_data: dict[str, Any],
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        self._mock_tool_lambda(mocker)
        mocker.patch(
            "app.services.runs.tool_strategy.process_tool",
            new=AsyncMock(side_effect=RuntimeError("process failed")),
        )

        response = client.post(
            api_path,
            data=run_tool_request_data,
            files={"tool": ("test_tool.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
            headers=auth_header,
        )

        assert response.status_code == status.HTTP_200_OK
        assert _minted_records(caplog) == []
        _assert_minted_tokens_absent(caplog, auth_header["Authorization"], [])
