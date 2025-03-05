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
        """Mock dependencies for successful test run."""
        # Mock process_skill
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

        # Mock AWSLambdaClient
        # Create a proper async mock for wait_for_function_active
        wait_for_function_active_mock = AsyncMock(return_value=True)

        mock_create_function_result = mocker.Mock()
        mock_create_function_result.function_arn = TEST_FUNCTION_ARN
        mock_create_function_result.function_name = TEST_FUNCTION_NAME

        mock_lambda_client = mocker.Mock()
        mock_lambda_client.create_function.return_value = mock_create_function_result
        mock_lambda_client.invoke_function.return_value = (
            {"status_code": 200, "response": {"result": "Test executed successfully"}},
            TEST_START_TIME,
            TEST_END_TIME,
        )
        mock_lambda_client.wait_for_function_active = wait_for_function_active_mock

        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock AWSLogsClient
        # Create a properly configured AWSLogsClient mock with async method
        mock_logs = [
            {"timestamp": 1001, "message": "START RequestId: test-request-id"},
            {"timestamp": 1002, "message": "Processing request"},
            {"timestamp": 1003, "message": "END RequestId: test-request-id"},
        ]

        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=mock_logs)

        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

    def test_run_skill_success(
        self, post_run_request_factory: Callable[[], Any], mock_success_dependencies: None
    ) -> None:
        """Test successful run_skill_test endpoint."""
        # Minimum expected response items
        min_expected_responses = 5  # initial, skill processed, lambda creating, test case, completion

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for expected response objects
        assert (
            len(response_data) >= min_expected_responses
        )  # At least initial response, skill processed, lambda creating, test case, completion

        # Check initial response
        assert response_data[0]["code"] == "PROCESSING_STARTED"
        assert response_data[0]["success"] is True

        # Check that test cases were run
        test_case_completed = False
        for resp in response_data:
            if resp.get("code") == "TEST_CASE_COMPLETED":
                test_case_completed = True
                assert "test_response" in resp["data"]
                assert "logs" in resp["data"]
                assert "duration" in resp["data"]

        assert test_case_completed, "No test case completion message found"

        # Check completion message
        final_message = response_data[-1]
        assert final_message["code"] == "TEST_RUN_COMPLETED"
        assert final_message["success"] is True

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
        assert response.status_code == expected_status

        # For error responses with streaming, check the error code
        if expected_error_code:
            response_data = parse_streaming_response(response)
            assert len(response_data) > 0
            assert response_data[-1]["code"] == expected_error_code
            assert response_data[-1]["success"] is False

    def test_process_skill_error(self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture) -> None:
        """Test error during skill processing."""
        # Setup
        error_message = "Error processing skill"
        mocker.patch("app.api.v1.routers.runs.process_skill", new=AsyncMock(side_effect=ValueError(error_message)))

        # Mock lambda client and its methods to avoid any real AWS calls
        mock_lambda_client = mocker.Mock()
        # Make sure delete_function is properly mocked
        mock_lambda_client.delete_function.return_value = None

        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock the logs client as well
        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])

        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert error_message in final_message["data"]["error"]

    def test_lambda_function_creation_failure(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture
    ) -> None:
        """Test failure in lambda function creation."""
        # Mock process_skill to succeed
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

        # Mock lambda client to fail on create_function
        mock_lambda_client = mocker.Mock()
        mock_lambda_client.create_function.return_value = mocker.Mock(function_arn=None, function_name=None)
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert "Failed to create lambda function" in final_message["data"]["error"]

    def test_lambda_function_activation_failure(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture
    ) -> None:
        """Test failure in lambda function activation."""
        # Mock process_skill to succeed
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

        # Mock AWSLambdaClient to succeed on create but fail on activation
        mock_create_function_result = mocker.Mock()
        mock_create_function_result.function_arn = TEST_FUNCTION_ARN
        mock_create_function_result.function_name = TEST_FUNCTION_NAME

        mock_lambda_client = mocker.Mock()
        mock_lambda_client.create_function.return_value = mock_create_function_result
        # Use AsyncMock for wait_for_function_active
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=False)

        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert "Lambda function did not became active" in final_message["data"]["error"]

    def test_lambda_function_invocation_error(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture
    ) -> None:
        """Test error during lambda function invocation."""
        # Mock process_skill to succeed
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

        # Mock AWSLambdaClient to succeed on create and activation
        mock_create_function_result = mocker.Mock()
        mock_create_function_result.function_arn = TEST_FUNCTION_ARN
        mock_create_function_result.function_name = TEST_FUNCTION_NAME

        mock_lambda_client = mocker.Mock()
        mock_lambda_client.create_function.return_value = mock_create_function_result
        # Mock invoke_function to raise an exception
        mock_lambda_client.invoke_function.side_effect = Exception("Invoke function error")
        # Need to properly mock the wait_for_function_active method as async
        mock_lambda_client.wait_for_function_active = AsyncMock(return_value=True)

        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Create a properly configured AWSLogsClient mock with async method
        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])

        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert "Invoke function error" in final_message["data"]["error"]

    def test_clean_up_on_error(self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture) -> None:
        """Test that lambda function is deleted even when errors occur."""
        # Mock process_skill to fail
        mocker.patch(
            "app.api.v1.routers.runs.process_skill", new=AsyncMock(side_effect=ValueError("Process skill error"))
        )

        # Mock lambda client
        mock_lambda_client = mocker.Mock()
        mock_lambda_client.delete_function.return_value = None

        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock logs client
        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])

        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Execute
        post_run_request_factory()

        # Assert
        # Check that delete_function was called even though the process failed
        assert mock_lambda_client.delete_function.call_count == 1
        assert mock_lambda_client.delete_function.call_args[1]["function_name"] is not None

    def test_agent_not_found_in_definition(
        self,
        post_run_request_factory: Callable[[], Any],
        mocker: MockerFixture,
        run_skill_request_data: dict[str, Any],
    ) -> None:
        """Test error when agent is not found in the definition."""
        # Modify the definition to have an empty agents dict
        modified_data = run_skill_request_data.copy()
        modified_data["definition"] = json.dumps({"agents": {}})

        # Create mock for all necessary components
        mock_lambda_client = mocker.Mock()
        mock_lambda_client.delete_function.return_value = None
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Create custom request with the modified data
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/runs"
        files = {
            "skill": ("test_skill.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        # Execute
        response = client.post(api_path, data=modified_data, files=files, headers={"Authorization": TEST_TOKEN})

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert f"Could not find agent {TEST_AGENT_NAME} in definition" in final_message["data"]["error"]

    def test_skill_not_found_for_agent(
        self,
        post_run_request_factory: Callable[[], Any],
        mocker: MockerFixture,
        run_skill_request_data: dict[str, Any],
    ) -> None:
        """Test error when skill is not found for the agent."""
        # Modify the definition to have an agent with no skills
        agent_definition = {
            "name": TEST_AGENT_NAME,
            "slug": "test-agent-slug",
            "skills": [],  # Empty skills list
        }

        modified_data = run_skill_request_data.copy()
        modified_data["definition"] = json.dumps({"agents": {"test-agent-id": agent_definition}})

        # Create mock for all necessary components
        mock_lambda_client = mocker.Mock()
        mock_lambda_client.delete_function.return_value = None
        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])
        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Create custom request with the modified data
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/runs"
        files = {
            "skill": ("test_skill.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        # Execute
        response = client.post(api_path, data=modified_data, files=files, headers={"Authorization": TEST_TOKEN})

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert f"Could not find skill {TEST_SKILL_NAME} for agent {TEST_AGENT_NAME}" in final_message["data"]["error"]

    def test_empty_skill_zip_bytes(self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture) -> None:
        """Test error when skill_zip_bytes is empty after processing."""
        # Mock process_skill to return empty skill_zip_bytes
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
                return_value=(
                    mock_process_result,
                    None,  # Empty skill_zip_bytes
                )
            ),
        )

        # Mock lambda client and its methods to avoid any real AWS calls
        mock_lambda_client = mocker.Mock()
        mock_lambda_client.delete_function.return_value = None

        mocker.patch("app.api.v1.routers.runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock the logs client as well
        mock_logs_client = mocker.Mock()
        mock_logs_client.get_function_logs = AsyncMock(return_value=[])

        mocker.patch("app.api.v1.routers.runs.AWSLogsClient", return_value=mock_logs_client)

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        final_message = response_data[-1]
        assert final_message["code"] == "PROCESSING_ERROR"
        assert final_message["success"] is False
        assert "Failed to process skill, aborting test run" in final_message["data"]["error"]
