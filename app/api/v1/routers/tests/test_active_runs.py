"""Tests for active agent runs endpoint."""

import io
import json
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4
from pathlib import Path

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
TEST_AGENT_KEY = "test_agent"
TEST_TOKEN = "Bearer test-token"
TEST_FUNCTION_NAME = "test-function-name"
TEST_FUNCTION_ARN = "arn:aws:lambda:us-east-1:123456789012:function:test-function-name"
TEST_START_TIME = 1000.0
TEST_END_TIME = 1030.0  # 30 seconds after start
TEST_PROJECT_UUID = UUID(str(uuid4()))
TEST_AGENT_DEFINITION = {
    "name": "Active Order Status",
    "description": "Agent responsible for triggering order status templates, based on the status received from VTEX - Order Status",
    "language": "pt_BR",
    "path_test": "tests/test.yaml",
    "rules": {
        "PaymentApproved": {
            "display_name": "Payment Approved",
            "template": "payment_confirmation_2",
            "start_condition": "When the status is 'payment-approved'",
            "source": {
                "entrypoint": "main.PaymentApproved",
                "path": "rules/status_payment_approved"
            }
        }
    },
    "pre-processing": {
        "source": {
            "entrypoint": "processing.PreProcessor",
            "path": "pre_processors/processor"
        },
        "result_examples_file": "result_example.json"
    }
}
TEST_TOOLKIT_VERSION = "1.0.0"
TEST_DEFINITION = {
    "tests": {
        "test_1": {
            "params": {
                "name": "Leo",
                "age": 21
            },
            "payload": "webhooks/webhook1.json",
            "project": {
                "uuid": "",
                "base_url": "",
            },
            "credentials": {
                "BASE_URL": "https://hinodeteste.myvtex.com",
                "ACCOUNT_NAME": "hinodeb2b"
            }
        },
        "test_2": {
            "params": {
                "name": "Maria",
                "age": 25
            },
            "payload": "webhooks/webhook2.json",
            "project": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "base_url": "https://api.example.com",
            },
            "credentials": {
                "BASE_URL": "https://anotherteste.myvtex.com",
                "ACCOUNT_NAME": "anotherb2b"
            }
        },
    }
}
TEST_RULE_KEY = "PaymentApproved"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Return the API path for active runs."""
    return f"{settings.API_PREFIX}/v1/active_runs"


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
def run_agent_request_data() -> dict[str, Any]:
    """Return test data for run_active_agent_test endpoint."""
    agent_definition = {
        "name": "Active Order Status",
        "description": "Agent responsible for triggering order status templates, based on the status received from VTEX - Order Status",
        "language": "pt_BR",
        "path_test": "tests/test.yaml",
        "rules": {
            "PaymentApproved": {
                "display_name": "Payment Approved",
                "template": "payment_confirmation_2",
                "start_condition": "When the status is 'payment-approved'",
                "source": {
                    "entrypoint": "main.PaymentApproved",
                    "path": "rules/status_payment_approved"
                }
            }
        },
        "pre-processing": {
            "source": {
                "entrypoint": "processing.PreProcessor",
                "path": "pre_processors/processor"
            },
            "result_examples_file": "result_example.json"
        }
    }

    test_definition = {
        "tests": {
            "test_1": {
                "params": {
                    "name": "Leo",
                    "age": 21
                },
                "payload": "webhooks/webhook1.json",
                "project": {
                    "uuid": "",
                    "base_url": "",
                },
                "credentials": {
                    "BASE_URL": "https://hinodeteste.myvtex.com",
                    "ACCOUNT_NAME": "hinodeb2b"
                }
            },
            "test_2": {
                "params": {
                    "name": "Maria",
                    "age": 25
                },
                "payload": "webhooks/webhook2.json",
                "project": {
                    "uuid": "123e4567-e89b-12d3-a456-426614174000",
                    "base_url": "https://api.example.com",
                },
                "credentials": {
                    "BASE_URL": "https://anotherteste.myvtex.com",
                    "ACCOUNT_NAME": "anotherb2b"
                }
            },
        }
    }

    return {
        "project_uuid": str(uuid4()),
        "definition": json.dumps({"agents": {TEST_AGENT_KEY: agent_definition}}),
        "test_definition": json.dumps(test_definition),
        "agent_key": TEST_AGENT_KEY,
        "rule_key": "PaymentApproved",
        "rule_credentials": json.dumps(
            {
                "BASE_URL": "https://defaulttest.myvtex.com",
                "ACCOUNT_NAME": "defaultb2b",
            }
        ),
        "rule_globals": json.dumps(
            {
                "global-test-key": "test-value",
            }
        ),
        "type": "active",
        "toolkit_version": "1.0.0",
    }


@pytest.fixture
def post_run_request_factory(
    client: TestClient, api_path: str, auth_header: dict[str, str], run_agent_request_data: dict[str, Any]
) -> Callable[[], Any]:
    """Return a factory function for making POST requests to the active runs endpoint."""

    def make_post_request() -> Any:
        # Create files dictionary with the rule file
        files = {
            "rule": ("test_rule.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        # Merge the data and files for the multipart request
        data = {**run_agent_request_data}

        headers = {**auth_header, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}

        return client.post(api_path, data=data, files=files, headers=headers)

    return make_post_request


@pytest.fixture
def custom_post_run_request_factory(
    client: TestClient,
    api_path: str,
    auth_header: dict[str, str],
) -> Callable[[dict[str, Any], dict[str, Any], dict[str, str] | None], Any]:
    """Return a factory function for making custom POST requests to the active runs endpoint."""

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


class TestActiveRunAgentEndpoint:
    """Tests for the run_active_agent_test endpoint."""

    @pytest.fixture
    def mock_success_dependencies(self, mocker: MockerFixture) -> None:
        """Mock dependencies for successful run_active_agent_test."""
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
                    "response": {"status": 0, "template": "Test template", "template_variables": {}},
                    "status_code": 200,
                    "logs": ["Log line 1", "Log line 2"],
                },
                TEST_START_TIME,
                TEST_END_TIME,
            )
        )
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.active_runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep to avoid delays in tests
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

    @pytest.fixture
    def payload_path_test_data(self, tmp_path: Path) -> dict[str, Any]:
        """Create test data with webhook payload file."""
        # Create a temporary webhook payload JSON file
        webhook_data = {
            "status": "payment-approved",
            "amount": 100.00,
            "customer_id": "12345"
        }
        
        payload_file = tmp_path / "webhook_payload.json"
        with open(payload_file, 'w') as f:
            json.dump(webhook_data, f)
        
        return {
            "project_uuid": TEST_PROJECT_UUID,
            "definition": json.dumps(TEST_AGENT_DEFINITION),
            "toolkit_version": TEST_TOOLKIT_VERSION,
            "agent_key": TEST_AGENT_KEY,
            "rule_key": TEST_RULE_KEY,
            "rule_credentials": {},
            "rule_globals": {},
            "type": "active",
            "test_definition": json.dumps(TEST_DEFINITION),
            "payload_path": str(payload_file),
        }

    def test_run_active_agent_success(
        self, post_run_request_factory: Callable[[], Any], mock_success_dependencies: None, mock_auth_middleware: None
    ) -> None:
        """Test successful run_active_agent_test endpoint."""
        # Minimum expected response items
        min_expected_responses = 4  # initial, agent processing, lambda creating, completion/error

        # Execute
        response = post_run_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for expected response objects
        assert (
            len(response_data) >= min_expected_responses
        )  # At least initial response, agent processed, lambda creating, completion/error

        # Check initial response
        assert response_data[0]["code"] == "PROCESSING_STARTED", "First message should have code=PROCESSING_STARTED"
        assert response_data[0]["success"] is True, "First message should have success=True"

        # Check for agent processed message
        agent_processed_msgs = [r for r in response_data if r.get("code") == "AGENT_PROCESSING"]
        assert len(agent_processed_msgs) > 0, "Should include a AGENT_PROCESSING message"

        # Check for lambda creating message
        lambda_creating_msgs = [r for r in response_data if r.get("code") == "LAMBDA_FUNCTION_CREATING"]
        assert len(lambda_creating_msgs) > 0, "Should include a LAMBDA_FUNCTION_CREATING message"

    @pytest.mark.parametrize(
        "test_id, data_fields, files_fields, headers, expected_status, expected_error_code",
        [
            (
                "missing_agent_file",
                {
                    "project_uuid": str(uuid4()),
                    "definition": json.dumps({"agents": {}}),
                    "test_definition": json.dumps({"tests": {}}),
                    "agent_key": TEST_AGENT_KEY,
                    "rule_key": "test_rule",
                    "rule_credentials": json.dumps({}),
                    "rule_globals": json.dumps({}),
                    "type": "active",
                    "toolkit_version": "1.0.0",
                },
                {},  # No rule file
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
                    "agent_key": TEST_AGENT_KEY,
                    "rule_key": "test_rule",
                    "rule_credentials": json.dumps({}),
                    "rule_globals": json.dumps({}),
                    "type": "active",
                    "toolkit_version": "1.0.0",
                },
                {
                    "rule": ("test_rule.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
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
        """Test validation errors for run_active_agent_test endpoint."""
        # Execute
        response = custom_post_run_request_factory(data_fields, files_fields, headers)

        # Assert
        assert response.status_code == expected_status, f"Expected status {expected_status} for {test_id}"

    def test_agent_without_rules_error(
        self, post_run_request_factory: Callable[[], Any], mocker: MockerFixture, mock_auth_middleware: None
    ) -> None:
        """Test error handling when agent doesn't have rules (not an active agent)."""
        # Mock AWS clients
        mock_lambda_client = mocker.MagicMock()
        mock_lambda_client.delete_function = mocker.MagicMock(return_value=None)
        mocker.patch("app.api.v1.routers.active_runs.AWSLambdaClient", return_value=mock_lambda_client)

        # Mock asyncio.sleep
        mocker.patch("asyncio.sleep", new=AsyncMock(return_value=None))

        # Create a passive agent definition (with tools instead of rules)
        passive_agent_definition = {
            "name": "Passive Test Agent",
            "description": "A passive agent for testing",
            "language": "pt_BR",
            "tools": [  # This should cause an error for active agents
                {
                    "key": "test_tool",
                    "name": "Test Tool",
                    "slug": "test-tool-slug",
                    "description": "A test tool",
                    "source": {
                        "entrypoint": "main.TestTool",
                        "path": "tools/test_tool"
                    }
                }
            ],
        }

        # Make the request with passive agent definition
        client = TestClient(app)
        api_path = f"{settings.API_PREFIX}/v1/active_runs"

        files = {
            "rule": ("test_rule.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
        }

        data = {
            "project_uuid": str(uuid4()),
            "definition": json.dumps({"agents": {TEST_AGENT_KEY: passive_agent_definition}}),
            "test_definition": json.dumps({"tests": {}}),
            "agent_key": TEST_AGENT_KEY,
            "rule_key": "test_rule",
            "rule_credentials": json.dumps({}),
            "rule_globals": json.dumps({}),
            "type": "active",
            "toolkit_version": "1.0.0",
        }

        response = client.post(
            api_path,
            data=data,
            files=files,
            headers={
                "Authorization": TEST_TOKEN,
                "X-Project-Uuid": str(uuid4()),
                "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
            },
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)

        # Check for error message
        error_responses = [r for r in response_data if r.get("success") is False]
        assert len(error_responses) > 0
        assert "does not have rules" in str(error_responses[-1])

    def test_webhook_data_loading_from_payload_path(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        payload_path_test_data: dict[str, Any],
        tmp_path: Path,
        mock_success_dependencies: None,
        mock_auth_middleware: None,
    ) -> None:
        """Test that webhook data is loaded from payload_path."""
        # Make request with payload_path
        response = client.post(
            api_path,
            data=payload_path_test_data,
            files={"rule": ("test.zip", io.BytesIO(b"test content"), "application/zip")},
            headers=auth_header,
        )

        # Assert response is successful
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)
        
        # Verify test case was run
        test_case_messages = [
            msg for msg in response_data 
            if msg.get("code") == "TEST_CASE_COMPLETED"
        ]
        assert len(test_case_messages) > 0, "Should have completed test cases"

    def test_webhook_data_fallback_when_no_payload_path(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        post_request_data: dict[str, Any],
        mock_success_dependencies: None,
        mock_auth_middleware: None,
    ) -> None:
        """Test that webhook data is empty when no payload_path is provided."""
        # Make request without payload_path
        response = client.post(
            api_path,
            data=post_request_data,
            files={"rule": ("test.zip", io.BytesIO(b"test content"), "application/zip")},
            headers=auth_header,
        )

        # Assert response is successful
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)
        
        # Verify test case was run (even without webhook data)
        test_case_messages = [
            msg for msg in response_data 
            if msg.get("code") == "TEST_CASE_COMPLETED"
        ]
        assert len(test_case_messages) > 0, "Should have completed test cases"

    def test_webhook_data_error_handling_invalid_path(
        self,
        client: TestClient,
        api_path: str,
        auth_header: dict[str, str],
        post_request_data: dict[str, Any],
        mock_success_dependencies: None,
        mock_auth_middleware: None,
    ) -> None:
        """Test error handling when payload_path points to non-existent file."""
        # Add invalid payload_path
        post_request_data["payload_path"] = "/nonexistent/path/webhook.json"
        
        # Make request with invalid payload_path
        response = client.post(
            api_path,
            data=post_request_data,
            files={"rule": ("test.zip", io.BytesIO(b"test content"), "application/zip")},
            headers=auth_header,
        )

        # Assert response is successful (should handle error gracefully)
        assert response.status_code == status.HTTP_200_OK

        # Parse streaming response
        response_data = parse_streaming_response(response)
        
        # Verify test case was still run (with empty webhook data)
        test_case_messages = [
            msg for msg in response_data 
            if msg.get("code") == "TEST_CASE_COMPLETED"
        ]
        assert len(test_case_messages) > 0, "Should have completed test cases even with invalid payload_path" 