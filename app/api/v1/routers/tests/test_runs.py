"""Tests for skill runs endpoint."""

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
TEST_SKILL_NAME = "test-skill"
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
    """Return the API path for runs."""
    return f"{settings.API_PREFIX}/v1/runs"


@pytest.fixture
def project_uuid() -> UUID:
    """Return a test project UUID."""
    return uuid4()


@pytest.fixture(scope="module")
def auth_header() -> dict[str, str]:
    """Return an authorization header for tests."""
    return {"Authorization": TEST_TOKEN}


@pytest.fixture
def run_skill_request_data() -> dict[str, Any]:
    """Return test data for run_skill_test endpoint."""
    agent_definition = {
        "name": TEST_AGENT_NAME,
        "slug": "test-agent-slug",
        "skills": [
            {
                "name": TEST_SKILL_NAME,
                "slug": "test-skill-slug",
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
        "definition": json.dumps({"agents": {"test-agent-id": agent_definition}}),
        "test_definition": json.dumps(test_definition),
        "skill_name": TEST_SKILL_NAME,
        "agent_name": TEST_AGENT_NAME,
        "skill_credentials": json.dumps(
            {
                "credential-test-key": "test-value",
            }
        ),
        "skill_globals": json.dumps(
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
    client: TestClient, api_path: str, auth_header: dict[str, str], run_skill_request_data: dict[str, Any]
) -> Callable[[], Any]:
    """Return a factory function for making POST requests to the runs endpoint."""

    def make_post_request() -> Any:
        # Create files dictionary with the skill file
        files = {
            "skill": ("test_skill.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        # Merge the data and files for the multipart request
        data = {**run_skill_request_data}

        return client.post(api_path, data=data, files=files, headers=auth_header)

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


class TestRunSkillEndpoint:
    """Tests for the run_skill_test endpoint."""

    @pytest.fixture
    def mock_success_dependencies(self, mocker: MockerFixture) -> None:
        """Mock dependencies for successful run_skill_test."""
        # Mock process_skill to return a successful result
        mock_process_result = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL_NAME,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }

        mocker.patch(
            "app.api.v1.routers.runs.process_skill",
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
            return_value={
                "StatusCode": 200,
                "Payload": '{"result": "success", "output": "Test output"}',
            }
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock logs client
        mock_logs_client = mocker.MagicMock()
        mock_logs = [
            {"timestamp": TEST_START_TIME, "message": "START RequestId: test-request-id"},
            {"timestamp": TEST_END_TIME, "message": "END RequestId: test-request-id"},
        ]
        mock_logs_client.get_function_logs = AsyncMock(return_value=mock_logs)
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Mock asyncio.sleep to avoid delays in tests
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

    def test_run_skill_success(
        self, post_run_request_factory: Callable[[], Any], mock_success_dependencies: None
    ) -> None:
        """Test successful run_skill_test endpoint."""
        # Minimum expected response items
        min_expected_responses = 4  # initial, skill processed, lambda creating, completion/error

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for expected response objects
        assert (
            len(response_data) >= min_expected_responses
        )  # At least initial response, skill processed, lambda creating, completion/error

        # Check initial response
        assert response_data[0]["code"] == "PROCESSING_STARTED", "First message should have code=PROCESSING_STARTED"
        assert response_data[0]["success"] is True, "First message should have success=True"

        # Check for skill processed message
        skill_processed_msgs = [r for r in response_data if r.get("code") == "SKILL_PROCESSED"]
        assert len(skill_processed_msgs) > 0, "Should include a SKILL_PROCESSED message"

        # Check for lambda creating message
        lambda_creating_msgs = [r for r in response_data if r.get("code") == "LAMBDA_FUNCTION_CREATING"]
        assert len(lambda_creating_msgs) > 0, "Should include a LAMBDA_FUNCTION_CREATING message"

    @pytest.mark.parametrize(
        "test_id, data_fields, files_fields, headers, expected_status, expected_error_code",
        [
            (
                "missing_skill_file",
                {
                    "project_uuid": str(uuid4()),
                    "definition": json.dumps({"agents": {}}),
                    "test_definition": json.dumps({"tests": {}}),
                    "skill_name": TEST_SKILL_NAME,
                    "agent_name": TEST_AGENT_NAME,
                    "skill_credentials": json.dumps({}),
                    "skill_globals": json.dumps({}),
                    "toolkit_version": "1.0.0",
                },
                {},  # No skill file
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
                    "skill_name": TEST_SKILL_NAME,
                    "agent_name": TEST_AGENT_NAME,
                    "skill_credentials": json.dumps({}),
                    "skill_globals": json.dumps({}),
                    "toolkit_version": "1.0.0",
                },
                {
                    "skill": ("test_skill.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
                },
                {},  # Empty headers - no auth
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                None,  # No streaming response for 422
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
    ) -> None:
        """Test validation errors for run_skill_test endpoint."""
        # Execute
        response = custom_post_run_request_factory(data_fields, files_fields, headers)

        # Assert
        assert response.status_code == expected_status, f"Expected status {expected_status} for {test_id}"

    def test_process_skill_error(self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture) -> None:
        """Test error handling when process_skill raises an exception."""
        # Setup
        error_message = "Error processing skill"
        mocker.patch("app.api.v1.routers.runs.process_skill", new=AsyncMock(side_effect=ValueError(error_message)))

        # Mock AWS clients
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

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
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture
    ) -> None:
        """Test error handling when lambda function creation fails."""
        # Setup - mock process_skill to succeed
        mock_process_result = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL_NAME,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }
        mocker.patch(
            "app.api.v1.routers.runs.process_skill",
            new=AsyncMock(return_value=(mock_process_result, io.BytesIO(TEST_CONTENT))),
        )

        # Mock Lambda client to fail function creation
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.create_function = mocker.MagicMock(side_effect=ValueError("Function creation failed"))
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock Logs client
        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

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
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture
    ) -> None:
        """Test error handling when lambda function activation fails."""
        # Setup - mock process_skill to succeed
        mock_process_result = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL_NAME,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }
        mocker.patch(
            "app.api.v1.routers.runs.process_skill",
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

        # Mock Logs client
        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

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
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture
    ) -> None:
        """Test error handling when lambda function invocation fails."""
        # Setup - mock process_skill to succeed
        mock_process_result = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL_NAME,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }
        mocker.patch(
            "app.api.v1.routers.runs.process_skill",
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

        # Mock Logs client
        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

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

    def test_clean_up_on_error(self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture) -> None:
        """Test that resources are cleaned up when an error occurs."""
        mocker.patch(
            "app.api.v1.routers.runs.process_skill", new=AsyncMock(side_effect=ValueError("Process skill error"))
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

        # Mock Logs client
        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

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
        run_skill_request_data: dict[str, Any],
    ) -> None:
        """Test error handling when agent is not found in definition."""
        # Setup - Create a definition with no agents
        empty_definition: dict[str, dict] = {"agents": {}}

        # Setup mocks
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Modify the request data
        modified_data = run_skill_request_data.copy()
        modified_data["definition"] = json.dumps(empty_definition)

        # Make the request
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/runs"

        files = {
            "skill": ("test_skill.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        response = client.post(
            api_path,
            data=modified_data,
            files=files,
            headers={"Authorization": TEST_TOKEN},
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error message
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert f"Could not find agent {TEST_AGENT_NAME}" in str(error_responses[-1])

    def test_skill_not_found_for_agent(
        self,
        post_run_request_factory: Callable[[], Any],
        mocker: MockerFixture,
        run_skill_request_data: dict[str, Any],
    ) -> None:
        """Test error handling when skill is not found for agent."""
        # Setup - Create a definition with an agent but no skills
        agent_without_skills = {
            "name": TEST_AGENT_NAME,
            "slug": "test-agent-slug",
            "skills": [],  # Empty skills list
        }
        definition = {"agents": {"test-agent-id": agent_without_skills}}

        # Setup mocks
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Modify the request data
        modified_data = run_skill_request_data.copy()
        modified_data["definition"] = json.dumps(definition)

        # Make the request
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/runs"

        files = {
            "skill": ("test_skill.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        response = client.post(
            api_path,
            data=modified_data,
            files=files,
            headers={"Authorization": TEST_TOKEN},
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error message
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert f"Could not find skill {TEST_SKILL_NAME}" in str(error_responses[-1])

    def test_empty_skill_zip_bytes(self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture) -> None:
        """Test error handling when skill_zip_bytes is empty after processing."""
        # Setup - mock process_skill to return None for skill_zip_bytes
        mock_process_result = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL_NAME,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }
        mocker.patch(
            "app.api.v1.routers.runs.process_skill",
            new=AsyncMock(
                return_value=(mock_process_result, None)  # Return None for skill_zip_bytes
            ),
        )

        # Mock Lambda client
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock Logs client
        mock_logs_client = mocker.MagicMock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

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
        assert "Failed to process skill" in str(error_responses[-1])
