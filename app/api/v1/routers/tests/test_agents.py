"""Tests for agents configuration endpoint."""

import io
import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from starlette.datastructures import UploadFile
from starlette.responses import StreamingResponse

from app.core.config import settings
from app.main import app
from app.tests.utils import AsyncMock

# Common test constants
TEST_CONTENT = b"test content"
TEST_AGENT = "test-agent"
TEST_TOOL = "test-tool"
TEST_TOOL_KEY = "test_tool"
TEST_AGENT_KEY = "test_agent"
TEST_FULL_TOOL_KEY = f"{TEST_AGENT_KEY}:{TEST_TOOL_KEY}"
TEST_TOKEN = "Bearer test-token"
TEST_PROJECT_UUID = "c67bc61e-c2b2-43f1-a409-88dec4bd4b9e"
TEST_REQUEST_ID = "b1e887cc-ac6f-4e81-9d82-2bc65343a756"
TEST_TOOLKIT_VERSION = "1.0.0"


# Common fixtures
@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Get the API path for agents endpoint."""
    return f"{settings.API_PREFIX}/v1/agents"


@pytest.fixture(scope="module")
def project_uuid() -> UUID:
    """Generate a random project UUID."""
    # Use a fixed UUID for consistency in tests if needed, otherwise keep random
    # return uuid4()
    return UUID(TEST_PROJECT_UUID)


@pytest.fixture(scope="module")
def auth_header(project_uuid: UUID) -> dict[str, str]:
    """Create an authorization header."""
    return {
        "Authorization": TEST_TOKEN,
        "X-Project-Uuid": str(project_uuid), # Remove: Sent via form data model
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


@pytest.fixture
def agent_definition() -> dict[str, Any]:
    """Create a standard agent definition for testing."""
    return {
        "agents": {
            TEST_AGENT_KEY: {
                "name": "Test Agent",
                "slug": TEST_AGENT,
                "description": "A test agent",
                "tools": [
                    {
                        "key": TEST_TOOL_KEY,
                        "slug": TEST_TOOL,
                        "name": "Test Tool",
                        "description": "A test tool",
                        "source": {"entrypoint": "main.TestTool"},
                    }
                ],
            }
        }
    }


# Helper to create mock StreamingResponse content
def create_mock_stream_content(events: list[dict[str, Any]]) -> bytes:
    """Create NDJSON byte stream from a list of event dictionaries."""
    return "\n".join(json.dumps(event) for event in events).encode("utf-8")


# Helper to create a mock StreamingResponse
def create_mock_streaming_response(events: list[dict[str, Any]]) -> StreamingResponse:
    """Create a mock StreamingResponse."""
    content = create_mock_stream_content(events)
    return StreamingResponse(io.BytesIO(content), media_type="application/x-ndjson")


@pytest.fixture
def post_request_factory(
    client: TestClient,
    api_path: str,
    project_uuid: UUID,
    auth_header: dict[str, str],
    agent_definition: dict[str, Any],
) -> Callable[[], Any]:
    """Create a factory function for making standardized POST requests in tests."""

    def make_post_request() -> Any:
        return client.post(
            api_path,
            data={
                "type": "passive",
                "project_uuid": str(project_uuid),
                "definition": json.dumps(agent_definition),
                "toolkit_version": TEST_TOOLKIT_VERSION,
            },
            files={
                TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
            },
            headers=auth_header,
        )

    return make_post_request


@pytest.fixture
def custom_post_request_factory(
    client: TestClient,
    api_path: str,
    auth_header: dict[str, str],
) -> Callable[[dict[str, Any], dict[str, Any], dict[str, str] | None], Any]:
    """Create a factory function for making customized POST requests in tests."""

    def make_custom_post_request(
        data_fields: dict[str, Any],
        files_fields: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_headers = headers if headers is not None else auth_header
        return client.post(
            api_path,
            data=data_fields,
            files=files_fields,
            headers=request_headers,
        )

    return make_custom_post_request


def parse_streaming_response(response: Any) -> list[dict[str, Any]]:
    """Parse streaming response into JSON events."""
    if not hasattr(response, "content") or not response.content:
        return []

    lines = response.content.decode("utf-8").splitlines()
    result = []
    for line in lines:
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Failed to decode JSON line: {line}")  # Debugging
                pass  # Ignore lines that are not valid JSON
    return result


class TestAgentConfigEndpoint:
    """Tests for agent config endpoint."""

    @pytest.fixture
    def mock_passive_configurator(self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> Any:
        """Mock the PassiveAgentConfigurator using mocker."""
        # Mock the class itself using mocker
        mock_configurator_class = mocker.MagicMock()

        # Mock the instance and its configure_agents method
        mock_configurator_instance = mocker.MagicMock()
        mock_configurator_class.return_value = mock_configurator_instance

        # Patch the class in the router module using monkeypatch
        monkeypatch.setattr("app.api.v1.routers.agents.PassiveAgentConfigurator", mock_configurator_class)

        # Return the mock instance for further configuration in tests
        return mock_configurator_instance

    @pytest.fixture
    def mock_helper_functions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock helper functions called directly by the endpoint."""
        mock_upload_file = UploadFile(
            filename="test_tool.zip",
            file=io.BytesIO(TEST_CONTENT),
        )
        monkeypatch.setattr(
            "app.api.v1.routers.agents.extract_agent_resources_files",
            AsyncMock(return_value={TEST_FULL_TOOL_KEY: mock_upload_file}),
        )
        monkeypatch.setattr(
            "app.api.v1.routers.agents.read_agent_resources_content",
            AsyncMock(return_value=[(TEST_FULL_TOOL_KEY, TEST_CONTENT)]),
        )
        # Mock uuid4 used for request_id
        fixed_uuid = UUID(TEST_REQUEST_ID)
        monkeypatch.setattr("app.api.v1.routers.agents.uuid4", lambda: fixed_uuid)

    def test_success(  # noqa: PLR0913
        self,
        post_request_factory: Callable[[], Any],
        mock_helper_functions: None,
        mock_passive_configurator: Any,
        mock_auth_middleware: None,
    ) -> None:
        """Test successful agent config endpoint."""
        # Setup mock response for configure_agents
        success_stream = create_mock_streaming_response(
            [
                {"message": "Processing agents...", "success": True, "code": "PROCESSING_STARTED", "progress": 0.01},
                {"message": "Tool processed successfully", "success": True, "code": "TOOL_PROCESSED"},
                {"message": "Updating agents...", "success": True, "code": "NEXUS_UPLOAD_STARTED", "progress": 0.99},
                {"message": "Agents processed successfully", "success": True, "code": "PROCESSING_COMPLETED", "progress": 1.0},
            ]
        )
        mock_passive_configurator.configure_agents.return_value = success_stream

        # Execute
        response = post_request_factory()

        print("RESPONSE 🔥🔥🔥", response.content)

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"
        mock_passive_configurator.configure_agents.assert_called_once() # Verify configurator was called

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check that the streaming response includes expected messages
        assert len(response_data) >= 2, "Should contain at least start and end messages"
        assert response_data[0]["code"] == "PROCESSING_STARTED"
        assert response_data[-1]["code"] == "PROCESSING_COMPLETED"
        assert response_data[-1]["success"] is True

    @pytest.mark.parametrize(
        "test_id, data_fields, files_fields, headers, expected_status, error_msg, mock_stream_events",
        [
            # --- Validation Errors (Handled before configurator is called) ---
            (
                "missing_project_uuid",
                {
                    "type": "passive",
                    "definition": json.dumps({"agents": {}}),
                    "toolkit_version": TEST_TOOLKIT_VERSION,
                },
                {TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(b"test"), "application/zip")},
                {"Authorization": TEST_TOKEN, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}, # Missing X-Project-Uuid
                status.HTTP_400_BAD_REQUEST, # Expect 400 for missing header based on auth logic
                "Should require project_uuid in data or header",
                None, # Configurator not called
            ),
            (
                "missing_toolkit_version",
                {
                    "type": "passive",
                    "project_uuid": TEST_PROJECT_UUID,
                    "definition": json.dumps({"agents": {}}),
                    # Missing toolkit_version
                },
                {TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(b"test"), "application/zip")},
                None, # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Should require toolkit_version",
                None, # Configurator not called
            ),
            (
                "invalid_definition",
                {
                    "type": "passive",
                    "project_uuid": TEST_PROJECT_UUID,
                    "definition": "invalid json",
                    "toolkit_version": TEST_TOOLKIT_VERSION,
                },
                {TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(b"test"), "application/zip")},
                None, # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Should validate definition JSON",
                None, # Configurator not called
            ),
             (
                "missing_authorization",
                {
                    "type": "passive",
                    "project_uuid": TEST_PROJECT_UUID,
                    "definition": json.dumps({"agents": {}}),
                    "toolkit_version": TEST_TOOLKIT_VERSION,
                },
                {TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
                {"X-Project-Uuid": TEST_PROJECT_UUID, "X-CLI-Version": settings.CLI_MINIMUM_VERSION}, # Missing Auth
                status.HTTP_400_BAD_REQUEST, # Expect 400 for missing header based on auth logic
                "Missing Authorization header",
                None, # Configurator not called
            ),
             (
                "missing_type",
                {
                    # "type": "passive", # Missing type
                    "project_uuid": TEST_PROJECT_UUID,
                    "definition": json.dumps({"agents": {}}),
                    "toolkit_version": TEST_TOOLKIT_VERSION,
                },
                {TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
                None, # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Missing type field",
                None, # Configurator not called
            ),
            (
                "invalid_type",
                {
                    "type": "invalid", # Invalid type
                    "project_uuid": TEST_PROJECT_UUID,
                    "definition": json.dumps({"agents": {}}),
                    "toolkit_version": TEST_TOOLKIT_VERSION,
                },
                {TEST_FULL_TOOL_KEY: ("test.zip", io.BytesIO(TEST_CONTENT), "application/zip")},
                None, # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY, # Validation error
                "Invalid type field",
                None, # Configurator not called
            ),
            # --- Configurator Errors (Handled within the stream) ---
            (
                "configurator_tool_process_error",
                None, # Use default post_request_factory
                None, # Use default post_request_factory
                None, # Use default post_request_factory
                status.HTTP_200_OK, # Stream starts successfully
                "Should handle tool processing error from configurator",
                [
                    {"message": "Processing agents...", "success": True, "code": "PROCESSING_STARTED"},
                    {"message": "Error processing tool", "data": {"error": "Simulated tool error"}, "success": False, "code": "TOOL_PROCESSING_ERROR"},
                    {"message": "Error processing agents", "data": {"error": "Failed to process tool..."}, "success": False, "code": "PROCESSING_ERROR"}
                ],
            ),
             (
                "configurator_nexus_push_error",
                None, # Use default post_request_factory
                None, # Use default post_request_factory
                None, # Use default post_request_factory
                status.HTTP_200_OK, # Stream starts successfully
                "Should handle nexus push error from configurator",
                [
                    {"message": "Processing agents...", "success": True, "code": "PROCESSING_STARTED"},
                    {"message": "Tool processed successfully", "success": True, "code": "TOOL_PROCESSED"},
                    {"message": "Updating agents...", "success": True, "code": "NEXUS_UPLOAD_STARTED"},
                    {"message": "Failed to push agents", "data": {"error": "Simulated Nexus error"}, "success": False, "code": "NEXUS_UPLOAD_ERROR"},
                    {"message": "Error processing agents", "data": {"error": "Simulated Nexus error"}, "success": False, "code": "PROCESSING_ERROR"}
                ],
            ),
        ],
    )
    def test_errors_and_validations( # noqa: PLR0913
        self,
        custom_post_request_factory: Callable[[dict[str, Any], dict[str, Any], dict[str, str] | None], Any],
        post_request_factory: Callable[[], Any],
        project_uuid: UUID, # Included to ensure fixture is available if needed by auth_header
        test_id: str,
        data_fields: dict[str, Any] | None,
        files_fields: dict[str, Any] | None,
        headers: dict[str, str] | None,
        expected_status: int,
        error_msg: str,
        mock_stream_events: list[dict[str, Any]] | None,
        mock_helper_functions: None, # Applied to all tests in this parametrize
        mock_passive_configurator: Any, # Applied to all tests
        mock_auth_middleware: None, # Applied to all tests
    ) -> None:
        """Test validation errors and error handling within the stream."""
        if mock_stream_events:
            # This is a test case where the configurator is expected to be called and return a stream (potentially with errors)
            mock_response = create_mock_streaming_response(mock_stream_events)
            mock_passive_configurator.configure_agents.return_value = mock_response
            response = post_request_factory() # Use the standard factory

            # Assert status and stream content
            assert response.status_code == expected_status, f"{test_id}: {error_msg} - Status Code Check"
            response_data = parse_streaming_response(response)

            # Check if the last message indicates failure if expected
            should_fail = any(not event.get("success", True) for event in mock_stream_events)
            if should_fail:
                 assert any(not event.get("success", True) for event in response_data), f"{test_id}: {error_msg} - Stream Error Check"
                 # Check the last message is a failure/processing error
                 assert response_data[-1].get("success") is False, f"{test_id}: {error_msg} - Last Message Failure Check"
                 assert response_data[-1].get("code") in ["PROCESSING_ERROR", "NEXUS_UPLOAD_ERROR", "TOOL_PROCESSING_ERROR"], f"{test_id}: {error_msg} - Last Message Code Check"
            else:
                 assert all(event.get("success", True) for event in response_data), f"{test_id}: {error_msg} - Stream Success Check"

        else:
            # This is a test case for input validation errors (configurator should not be called)
            data = {} if data_fields is None else data_fields
            files = {} if files_fields is None else files_fields
            response = custom_post_request_factory(data, files, headers)

            # Assert status only, configurator not called
            assert response.status_code == expected_status, f"{test_id}: {error_msg} - Status Code Check"
            mock_passive_configurator.configure_agents.assert_not_called() # Verify configurator was NOT called

    def test_push_to_nexus_error_response(
        self, post_request_factory: Callable[[], Any], mock_helper_functions: None, mock_passive_configurator: Any, mock_auth_middleware: None
    ) -> None:
        """Test handling of error response from push_to_nexus via configurator."""
        # Setup mock response for configure_agents to simulate Nexus error
        nexus_error_stream = create_mock_streaming_response(
             [
                {"message": "Processing agents...", "success": True, "code": "PROCESSING_STARTED"},
                {"message": "Tool processed successfully", "success": True, "code": "TOOL_PROCESSED"},
                {"message": "Updating agents...", "success": True, "code": "NEXUS_UPLOAD_STARTED"},
                {"message": "Failed to push agents", "data": {"error": "Simulated Nexus Error"}, "success": False, "code": "NEXUS_UPLOAD_ERROR"},
                {"message": "Error processing agents", "data": {"error": "Simulated Nexus Error"}, "success": False, "code": "PROCESSING_ERROR"}
            ]
        )
        mock_passive_configurator.configure_agents.return_value = nexus_error_stream

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK as stream starts"
        mock_passive_configurator.configure_agents.assert_called_once()

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        error_messages = [r for r in response_data if r.get("success") is False]
        assert len(error_messages) > 0, "Should include an error message"
        assert error_messages[-1]["code"] == "PROCESSING_ERROR", "Final message should be PROCESSING_ERROR"
        assert "Simulated Nexus Error" in str(error_messages[-1]["data"]), "Error message should contain Nexus error details"

    def test_final_success_message(
        self, post_request_factory: Callable[[], Any], mock_helper_functions: None, mock_passive_configurator: Any, mock_auth_middleware: None
    ) -> None:
        """Test final success message in streaming response."""
         # Setup mock response for configure_agents
        success_stream = create_mock_streaming_response(
            [
                {"message": "Processing agents...", "success": True, "code": "PROCESSING_STARTED"},
                {"message": "Tool processed successfully", "success": True, "code": "TOOL_PROCESSED"},
                {"message": "Updating agents...", "success": True, "code": "NEXUS_UPLOAD_STARTED"},
                {"message": "Agents processed successfully", "data": {"status": "completed"}, "success": True, "code": "PROCESSING_COMPLETED", "progress": 1.0},
            ]
        )
        mock_passive_configurator.configure_agents.return_value = success_stream

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"
        mock_passive_configurator.configure_agents.assert_called_once()

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for success response
        assert len(response_data) > 0, "Response should not be empty"
        final_message = response_data[-1]
        assert final_message["success"] is True, "Final message should have success=True"
        assert final_message["code"] == "PROCESSING_COMPLETED", "Final message should have code=PROCESSING_COMPLETED"
        assert final_message["data"]["status"] == "completed", "Final message data should indicate completion"


    def test_push_to_nexus_with_no_tools(
         self, post_request_factory: Callable[[], Any], mock_helper_functions: None, mock_passive_configurator: Any, mock_auth_middleware: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test pushing to Nexus when there are no tools."""
        # Override helper mocks to return no tools
        monkeypatch.setattr("app.api.v1.routers.agents.extract_agent_resources_files", AsyncMock(return_value={}))
        monkeypatch.setattr("app.api.v1.routers.agents.read_agent_resources_content", AsyncMock(return_value=[]))

        # Setup mock response for configure_agents (simulating success with 0 tools)
        no_tools_success_stream = create_mock_streaming_response(
            [
                {"message": "Processing agents...", "data": {"total_files": 0}, "success": True, "code": "PROCESSING_STARTED"},
                # No TOOL_PROCESSED message
                {"message": "Updating agents...", "data": {"tool_count": 0}, "success": True, "code": "NEXUS_UPLOAD_STARTED"},
                {"message": "Agents processed successfully", "data": {"total_files": 0, "processed_files": 0, "status": "completed"}, "success": True, "code": "PROCESSING_COMPLETED", "progress": 1.0},
            ]
        )
        mock_passive_configurator.configure_agents.return_value = no_tools_success_stream

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"

        # Verify configurator was called correctly (even with no tools)
        mock_passive_configurator.configure_agents.assert_called_once_with([]) # Ensure called with empty list

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for complete response
        assert len(response_data) > 0, "Response should not be empty"
        final_message = response_data[-1]
        assert final_message["success"] is True, "Final message should have success=True"
        assert final_message["code"] == "PROCESSING_COMPLETED", "Final message should have code=PROCESSING_COMPLETED"
        assert final_message["data"]["processed_files"] == 0, "Final message data should show 0 processed files"

    def test_process_tool_failure_stops_processing(
        self, post_request_factory: Callable[[], Any], mock_helper_functions: None, mock_passive_configurator: Any, mock_auth_middleware: None
    ) -> None:
        """Test that processing stops on tool processing failure within the configurator stream."""
        # Setup mock response for configure_agents to simulate tool processing error
        tool_error_stream = create_mock_streaming_response(
             [
                {"message": "Processing agents...", "success": True, "code": "PROCESSING_STARTED"},
                {"message": "Error processing tool", "data": {"error": "Simulated tool processing failure"}, "success": False, "code": "TOOL_PROCESSING_ERROR"},
                {"message": "Error processing agents", "data": {"error": "Failed to process tool..."}, "success": False, "code": "PROCESSING_ERROR"}
            ]
        )
        mock_passive_configurator.configure_agents.return_value = tool_error_stream

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK for streaming response"
        mock_passive_configurator.configure_agents.assert_called_once()

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        error_messages = [r for r in response_data if not r.get("success", True)]
        assert len(error_messages) > 0, "Should include an error message"
        assert error_messages[0]["code"] == "TOOL_PROCESSING_ERROR", "First error should be TOOL_PROCESSING_ERROR"
        assert response_data[-1]["code"] == "PROCESSING_ERROR", "Last message should be PROCESSING_ERROR"
        assert response_data[-1]["success"] is False, "Last message should indicate failure"
        # Verify Nexus update/final success message isn't present
        assert not any(r.get("code") == "NEXUS_UPLOAD_STARTED" for r in response_data), "Nexus upload should not start"
        assert not any(r.get("code") == "PROCESSING_COMPLETED" for r in response_data), "Processing should not complete successfully"


# Remove TestHelperFunctions related to push_to_nexus if it's gone
# Keep TestHelperFunctions for functions still used directly by the router
class TestHelperFunctions:
    """Tests for helper functions in agents.py."""

    def test_extract_agent_resources_files(self) -> None:
        """Test extracting tool files from form data."""
        import asyncio
        from collections.abc import ItemsView

        from app.api.v1.routers.agents import extract_agent_resources_files

        mock_file = UploadFile(filename="test.zip", file=io.BytesIO(TEST_CONTENT))
        # Simulate Starlette FormData structure more accurately
        class MockFormData:
            def __init__(self, items: dict[str, Any]):
                self._items = items
            def items(self) -> ItemsView[str, Any]:
                return self._items.items()

        mock_form_items = {
            "project_uuid": "test-uuid", 
            "definition": "{}", 
            "type": "passive",
            "toolkit_version": "1.0.0",
            TEST_FULL_TOOL_KEY: mock_file, 
            "another_agent:another_tool": mock_file,
            "invalid-key-no-colon": mock_file,
            "some_other_field": "some_value",
        }
        mock_form = MockFormData(mock_form_items)

        result = asyncio.run(extract_agent_resources_files(mock_form))
        assert len(result) == 2, "Should extract two valid tool files"
        assert TEST_FULL_TOOL_KEY in result, "Should extract first valid file"
        assert "another_agent:another_tool" in result, "Should extract second valid file"
        assert "invalid-key-no-colon" not in result, "Should ignore keys without colons"
        assert "project_uuid" not in result, "Should ignore non-file fields"

    def test_read_agent_resources_content(self) -> None:
        """Test reading content from tool files."""
        import asyncio

        from app.api.v1.routers.agents import read_agent_resources_content

        file1 = UploadFile(filename="test1.zip", file=io.BytesIO(b"content1"))
        file2 = UploadFile(filename="test2.zip", file=io.BytesIO(b"content2"))
        tools_folders_zips = {"agent1:tool1": file1, "agent2:tool2": file2}

        # Define constant for expected number of files
        expected_file_count = 2

        result = asyncio.run(read_agent_resources_content(tools_folders_zips))
        assert len(result) == expected_file_count, "Should read two files"
        # Sort results by key for consistent order checking
        result.sort(key=lambda x: x[0])
        assert result[0][0] == "agent1:tool1" and result[0][1] == b"content1", "First content should match"
        assert result[1][0] == "agent2:tool2" and result[1][1] == b"content2", "Second content should match"

