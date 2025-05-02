"""Tests for agents configuration endpoint."""

import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

import pytest
import requests
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from starlette.datastructures import UploadFile

from app.core.config import settings
from app.main import app
from app.tests.utils import AsyncMock

# Common test constants
TEST_CONTENT = b"test content"
TEST_AGENT = "test-agent"
TEST_SKILL = "test-skill"
TEST_SKILL_KEY = f"{TEST_AGENT}:{TEST_SKILL}"
TEST_TOKEN = "Bearer test-token"


# Common fixtures
@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Get the API path for agents endpoint."""
    return f"{settings.API_PREFIX}/v1/agents"


@pytest.fixture
def project_uuid() -> UUID:
    """Generate a random project UUID."""
    return uuid4()


@pytest.fixture(scope="module")
def auth_header() -> dict[str, str]:
    """Create an authorization header."""
    return {
        "Authorization": TEST_TOKEN,
        "X-Project-Uuid": str(project_uuid),
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


@pytest.fixture
def agent_definition() -> dict[str, Any]:
    """Create a standard agent definition for testing."""
    return {
        "agents": {
            TEST_AGENT: {
                "name": "Test Agent",
                "slug": TEST_AGENT,
                "description": "A test agent",
                "skills": [
                    {
                        "slug": TEST_SKILL,
                        "name": "Test Skill",
                        "description": "A test skill",
                        "source": {"entrypoint": "main.TestSkill"},
                    }
                ],
            }
        }
    }


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
                "project_uuid": str(project_uuid),
                "definition": json.dumps(agent_definition),
                "toolkit_version": "1.0.0",  # Add default toolkit version for tests
            },
            files={
                TEST_SKILL_KEY: ("test.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
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
        # Ensure toolkit_version is included if not explicitly provided
        if "toolkit_version" not in data_fields:
            data_fields["toolkit_version"] = "1.0.0"

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
                pass
    return result


class TestAgentConfigEndpoint:
    """Tests for agent config endpoint."""

    @pytest.fixture
    def mock_success_dependencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock dependencies for successful test."""
        mock_upload_file = UploadFile(
            filename="test_skill.zip",
            file=io.BytesIO(TEST_CONTENT),
        )
        monkeypatch.setattr(
            "app.api.v1.routers.agents.extract_skill_files",
            AsyncMock(return_value={TEST_SKILL_KEY: mock_upload_file}),
        )
        monkeypatch.setattr(
            "app.api.v1.routers.agents.read_skills_content",
            AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
        )
        process_response = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL,
                "agent_name": TEST_AGENT,
                "size_kb": 1.0,
                "progress": 100,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }

        # Use a custom mock for process_skill that accepts the toolkit_version parameter
        async def mock_process_skill(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], io.BytesIO]:
            return (process_response, io.BytesIO(b"processed content"))

        monkeypatch.setattr("app.services.skill.packager.process_skill", mock_process_skill)
        monkeypatch.setattr(
            "app.services.skill.packager.create_skill_zip", lambda *args, **kwargs: io.BytesIO(b"packed content")
        )
        monkeypatch.setattr(
            "app.api.v1.routers.agents.push_to_nexus",
            lambda *args, **kwargs: (True, None),
        )
        mock_nexus_response = type("MockResponse", (), {"status_code": 200, "json": lambda: {"success": True}})()
        mock_nexus = type("MockNexus", (), {"push_agents": lambda *a, **k: mock_nexus_response})()
        monkeypatch.setattr("app.api.v1.routers.agents.NexusClient", lambda *a, **k: mock_nexus)

        # Mock datetime and UUID
        mock_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_datetime = type("MockDatetime", (), {"now": lambda tz=None: mock_dt, "UTC": UTC})
        monkeypatch.setattr("app.core.response.datetime", mock_datetime)
        fixed_uuid = UUID("12345678-1234-5678-1234-567812345678")
        monkeypatch.setattr("app.api.v1.routers.agents.uuid4", lambda: fixed_uuid)

    def test_success(  # noqa: PLR0913
        self,
        post_request_factory: Callable[[], Any],
        mock_success_dependencies: None,
        mock_auth_middleware: None,
    ) -> None:
        """Test successful agent config endpoint."""
        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check that the streaming response includes expected messages
        minimum_message_count = 2  # At least processing and completed messages
        assert len(response_data) >= minimum_message_count

    @pytest.mark.parametrize(
        "test_id, data_fields, files_fields, headers, expected_status, error_msg, custom_setup",
        [
            (
                "missing_project_uuid",
                {
                    "definition": json.dumps({"agents": {}}),
                    "toolkit_version": "1.0.0",  # Add toolkit_version to avoid failing on that parameter
                },
                {
                    TEST_SKILL_KEY: ("test.zip", io.BytesIO(b"test"), "application/zip"),
                },
                None,  # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Should require project_uuid",
                None,  # No custom setup
            ),
            (
                "missing_toolkit_version",
                {
                    "project_uuid": "test-uuid",
                    "definition": json.dumps({"agents": {}}),
                },
                {
                    TEST_SKILL_KEY: ("test.zip", io.BytesIO(b"test"), "application/zip"),
                },
                None,  # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Should require toolkit_version",
                None,  # No custom setup
            ),
            (
                "invalid_definition",
                {
                    "project_uuid": "test-uuid",
                    "definition": "invalid json",
                    "toolkit_version": "1.0.0",  # Add toolkit_version to avoid failing on that parameter
                },
                {
                    TEST_SKILL_KEY: ("test.zip", io.BytesIO(b"test"), "application/zip"),
                },
                None,  # Use default auth header
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Should validate definition JSON",
                None,  # No custom setup
            ),
            (
                "missing_authorization",
                {
                    "project_uuid": "test-uuid",
                    "definition": json.dumps({"agents": {}}),
                    "toolkit_version": "1.0.0",  # Add toolkit_version to avoid failing on that parameter
                },
                {
                    TEST_SKILL_KEY: ("test.zip", io.BytesIO(TEST_CONTENT), "application/zip"),
                },
                {
                    "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
                },  # Empty headers - no auth
                status.HTTP_400_BAD_REQUEST,
                "Missing Authorization or X-Project-Uuid header",
                None,  # No custom setup
            ),
            (
                "process_skill_error",
                None,  # Will be set in the test
                None,  # Will be set in the test
                None,  # Use default auth header
                status.HTTP_200_OK,
                "Should handle process_skill error",
                {
                    "mock_process_skill": AsyncMock(side_effect=RuntimeError("Simulated error in processing")),
                },
            ),
        ],
    )
    def test_validation_errors(  # noqa: PLR0913
        self,
        custom_post_request_factory: Callable[[dict[str, Any], dict[str, Any], dict[str, str] | None], Any],
        post_request_factory: Callable[[], Any],
        project_uuid: UUID,
        test_id: str,
        data_fields: dict[str, Any] | None,
        files_fields: dict[str, Any] | None,
        headers: dict[str, str] | None,
        expected_status: int,
        error_msg: str,
        custom_setup: dict[str, Any] | None,
        monkeypatch: pytest.MonkeyPatch,
        mock_auth_middleware: None,
    ) -> None:
        """Test validation errors for agent config endpoint."""
        # For the process_skill_error case, we need a special setup
        if test_id == "process_skill_error":
            # Use the default post_request_factory
            # But first set up the mocks as defined in custom_setup
            if custom_setup:
                for key, value in custom_setup.items():
                    if key == "mock_process_skill":
                        monkeypatch.setattr("app.services.skill.packager.process_skill", value)

            response = post_request_factory()
        else:
            # Use the custom_post_request_factory for other cases
            data = {} if data_fields is None else data_fields
            files = {} if files_fields is None else files_fields
            response = custom_post_request_factory(data, files, headers)

        # Assert
        assert response.status_code == expected_status, error_msg

    def test_push_to_nexus_error_response(
        self, post_request_factory: Callable[[], Any], mock_auth_middleware: None, mocker: MockerFixture
    ) -> None:
        """Test handling of error response from push_to_nexus."""
        # Mock successful skill processing
        process_response = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL,
                "agent_name": TEST_AGENT,
                "size_kb": 1.0,
                "progress": 100,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }

        # Mock create_skill_zip to avoid zip file error
        mocker.patch(
            "app.services.skill.packager.create_skill_zip",
            return_value=io.BytesIO(b"valid zip content"),
        )

        # Mock process_skill to succeed
        mocker.patch(
            "app.services.skill.packager.process_skill",
            new=AsyncMock(return_value=(process_response, io.BytesIO(b"processed content"))),
        )

        # Mock extract_skill_files and read_skills_content to return test data
        mocker.patch(
            "app.api.v1.routers.agents.extract_skill_files",
            new=AsyncMock(return_value={TEST_SKILL_KEY: mocker.Mock()}),
        )
        mocker.patch(
            "app.api.v1.routers.agents.read_skills_content",
            new=AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
        )

        # Create a nexus error response
        nexus_error = {
            "message": "Failed to push agents",
            "data": {
                "error": "Error pushing to Nexus",
                "skills_processed": 1,
            },
            "success": False,
            "code": "NEXUS_UPLOAD_ERROR",
            "progress": 0.9,
        }

        # Mock push_to_nexus to return an error
        mocker.patch(
            "app.api.v1.routers.agents.push_to_nexus",
            return_value=(False, nexus_error),
        )

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        error_messages = [r for r in response_data if r.get("success") is False]
        assert len(error_messages) > 0, "Should include an error message"
        assert "Error pushing to Nexus" in str(error_messages[-1])

    def test_final_success_message(
        self, post_request_factory: Callable[[], Any], mock_auth_middleware: None, mocker: MockerFixture
    ) -> None:
        """Test final success message in streaming response."""
        # Mock successful skill processing
        process_response = {
            "message": "Skill processed successfully",
            "data": {
                "skill_name": TEST_SKILL,
                "agent_name": TEST_AGENT,
                "size_kb": 1.0,
                "progress": 100,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
        }

        # Mock create_skill_zip to avoid zip file error
        mocker.patch(
            "app.services.skill.packager.create_skill_zip",
            return_value=io.BytesIO(b"valid zip content"),
        )

        # Mock process_skill to succeed
        mocker.patch(
            "app.services.skill.packager.process_skill",
            new=AsyncMock(return_value=(process_response, io.BytesIO(b"processed content"))),
        )

        # Mock extract_skill_files and read_skills_content to return test data
        mocker.patch(
            "app.api.v1.routers.agents.extract_skill_files",
            new=AsyncMock(return_value={TEST_SKILL_KEY: mocker.Mock()}),
        )
        mocker.patch(
            "app.api.v1.routers.agents.read_skills_content",
            new=AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
        )

        # Mock push_to_nexus to succeed
        mocker.patch(
            "app.api.v1.routers.agents.push_to_nexus",
            return_value=(True, None),
        )

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for success response
        final_message = response_data[-1]
        assert final_message["success"] is True, "Final message should have success=True"
        assert final_message["code"] == "PROCESSING_COMPLETED", "Final message should have code=PROCESSING_COMPLETED"

    def test_push_to_nexus_with_no_skills(
        self, post_request_factory: Callable[[], Any], mock_auth_middleware: None, mocker: MockerFixture
    ) -> None:
        """Test pushing to Nexus when there are no skills."""
        # Mock extract_skill_files and read_skills_content to return empty results
        mocker.patch("app.api.v1.routers.agents.extract_skill_files", new=AsyncMock(return_value={}))
        mocker.patch("app.api.v1.routers.agents.read_skills_content", new=AsyncMock(return_value=[]))

        # Create a mock response object
        mock_response = mocker.MagicMock(spec=requests.Response)
        mock_response.status_code = status.HTTP_200_OK
        mock_response.json.return_value = {"success": True}

        # Create a mock NexusClient
        mock_nexus_client = mocker.MagicMock()
        mock_nexus_client.push_agents.return_value = mock_response

        # Patch the NexusClient
        mocker.patch("app.api.v1.routers.agents.NexusClient", return_value=mock_nexus_client)

        # Mock push_to_nexus to succeed
        mocker.patch(
            "app.api.v1.routers.agents.push_to_nexus",
            return_value=(True, None),
        )

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for complete response
        assert response_data[-1]["success"] is True, "Final message should have success=True"
        assert response_data[-1]["code"] == "PROCESSING_COMPLETED", (
            "Final message should have code=PROCESSING_COMPLETED"
        )

    def test_process_skill_failure_stops_processing(
        self, post_request_factory: Callable[[], Any], mock_auth_middleware: None, mocker: MockerFixture
    ) -> None:
        """Test that processing stops on skill processing failure."""
        error_message = "Error processing skill"

        # Create an error response
        error_response = {
            "message": error_message,
            "data": {},
            "success": False,
            "code": "SKILL_PROCESSING_ERROR",
        }

        # Setup mocks to simulate skill processing failure using mocker
        mocker.patch(
            "app.api.v1.routers.agents.extract_skill_files",
            new=AsyncMock(return_value={TEST_SKILL_KEY: mocker.Mock()}),
        )
        mocker.patch(
            "app.api.v1.routers.agents.read_skills_content",
            new=AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
        )
        mocker.patch("app.services.skill.packager.process_skill", new=AsyncMock(return_value=(error_response, None)))

        # Mock push_to_nexus to ensure it's not called
        def mock_push_to_nexus(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("push_to_nexus should not be called after skill processing failure")

        mocker.patch("app.api.v1.routers.agents.push_to_nexus", mock_push_to_nexus)

        # Execute
        response = post_request_factory()

        # Assert
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK for streaming response"

        # Parse the streaming response
        response_data = parse_streaming_response(response)

        # Check for error response
        error_messages = [r for r in response_data if not r.get("success", True)]
        assert len(error_messages) > 0, "Should include an error message"
        # We don't check the exact error message as it may be transformed by the error handling logic


class TestHelperFunctions:
    """Tests for helper functions in agents.py."""

    def test_extract_skill_files(self) -> None:
        """Test extracting skill files from form data."""
        import asyncio

        from app.api.v1.routers.agents import extract_skill_files

        mock_file = UploadFile(filename="test.zip", file=io.BytesIO(TEST_CONTENT))
        mock_form = {
            "project_uuid": "test-uuid",  # Not a file
            "definition": "{}",  # Not a file
            TEST_SKILL_KEY: mock_file,  # Valid
            "invalid-key": mock_file,  # Invalid (no colon)
        }

        result = asyncio.run(extract_skill_files(mock_form))
        assert len(result) == 1, "Should extract one file"
        assert TEST_SKILL_KEY in result, "Should extract valid file"
        assert "invalid-key" not in result, "Should ignore invalid keys"

    def test_read_skills_content(self) -> None:
        """Test reading content from skill files."""
        import asyncio

        from app.api.v1.routers.agents import read_skills_content

        file1 = UploadFile(filename="test1.zip", file=io.BytesIO(b"content1"))
        file2 = UploadFile(filename="test2.zip", file=io.BytesIO(b"content2"))
        skills_folders_zips = {"agent1:skill1": file1, "agent2:skill2": file2}

        # Define constant for expected number of files
        expected_file_count = 2

        result = asyncio.run(read_skills_content(skills_folders_zips))
        assert len(result) == expected_file_count, "Should read two files"
        assert result[0][0] == "agent1:skill1" and result[0][1] == b"content1", "First content should match"
        assert result[1][0] == "agent2:skill2" and result[1][1] == b"content2", "Second content should match"


class TestPushToNexus:
    """Tests for push_to_nexus function."""

    # Common test data
    project_uuid: ClassVar[str] = str(uuid4())
    definition: ClassVar[dict[str, Any]] = {"agents": {TEST_AGENT: {"skills": []}}}
    skill_mapping: ClassVar[dict[str, Any]] = {TEST_SKILL_KEY: io.BytesIO(b"skill content")}
    request_id: ClassVar[str] = str(uuid4())
    authorization: ClassVar[str] = TEST_TOKEN

    def test_success(self, mocker: MockerFixture) -> None:
        """Test successful push to Nexus."""
        from app.api.v1.routers.agents import push_to_nexus

        # Create a mock response with status_code
        mock_response = mocker.Mock(spec=requests.Response)
        mock_response.status_code = status.HTTP_200_OK
        mock_response.text = "Success"

        # Create a mock NexusClient instance
        mock_nexus_client = mocker.MagicMock()
        mock_nexus_client.push_agents.return_value = mock_response

        # Patch the NexusClient class to return our mock instance
        mocker.patch("app.api.v1.routers.agents.NexusClient", return_value=mock_nexus_client)

        success, response = push_to_nexus(
            self.project_uuid, self.definition, self.skill_mapping, self.request_id, self.authorization
        )

        assert success is True, "Should report success"
        assert response is None, "Should not return a response"

    def test_exception(self, mocker: MockerFixture) -> None:
        """Test exception handling in Nexus push."""
        from app.api.v1.routers.agents import push_to_nexus

        # Create a mock NexusClient instance that raises an error
        mock_nexus_client = mocker.MagicMock()
        mock_nexus_client.push_agents.side_effect = RuntimeError("API error")

        # Patch the NexusClient class to return our mock instance
        mocker.patch("app.api.v1.routers.agents.NexusClient", return_value=mock_nexus_client)

        success, response = push_to_nexus(
            self.project_uuid, self.definition, self.skill_mapping, self.request_id, self.authorization
        )

        assert success is False, "Should report failure"
        assert response is not None, "Should return an error response"
        assert response["success"] is False, "Response should indicate failure"
        assert response["data"] is not None, "Response should include data"
        assert "API error" in response["data"]["error"], "Should include the exception message"

    @pytest.mark.parametrize(
        "status_code, status_text, expected_error_fragment",
        [
            (status.HTTP_400_BAD_REQUEST, "Bad request error", "Failed to push agents: 400"),
            (status.HTTP_401_UNAUTHORIZED, "Unauthorized", "Failed to push agents: 401"),
            (status.HTTP_500_INTERNAL_SERVER_ERROR, "Server error", "Failed to push agents: 500"),
        ],
    )
    def test_non_200_status_codes(
        self, mocker: MockerFixture, status_code: int, status_text: str, expected_error_fragment: str
    ) -> None:
        """Test handling of various non-200 status codes from Nexus API."""
        from app.api.v1.routers.agents import push_to_nexus

        # Create a mock response with the specified non-200 status code
        mock_response = mocker.Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_response.text = status_text

        # Create a mock NexusClient instance
        mock_nexus_client = mocker.MagicMock()
        mock_nexus_client.push_agents.return_value = mock_response

        # Patch the NexusClient class to return our mock instance
        mocker.patch("app.api.v1.routers.agents.NexusClient", return_value=mock_nexus_client)

        success, response = push_to_nexus(
            self.project_uuid, self.definition, self.skill_mapping, self.request_id, self.authorization
        )

        assert success is False, f"Should report failure when status code is {status_code}"
        assert response is not None, "Should return an error response"
        assert response["success"] is False, "Response should indicate failure"
        assert response["data"] is not None, "Response should include data"
        assert expected_error_fragment in response["data"]["error"], "Should include status code in error"
        assert status_text in response["data"]["error"], "Should include response text in error message"
