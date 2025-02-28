"""Tests for agents configuration endpoint."""

import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from app.core.config import settings
from app.main import app

# Common test constants
TEST_CONTENT = b"test content"
TEST_AGENT = "test-agent"
TEST_SKILL = "test-skill"
TEST_SKILL_KEY = f"{TEST_AGENT}:{TEST_SKILL}"
TEST_TOKEN = "Bearer test-token"


class AsyncMock:
    """Simple async mock for tests."""

    def __init__(self, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.side_effect:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            if callable(self.side_effect):
                return self.side_effect(*args, **kwargs)
            return self.side_effect
        return self.return_value


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
    return {"Authorization": TEST_TOKEN}


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
    """Tests for agent configuration endpoint."""

    @pytest.fixture
    def mock_success_dependencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up mocks for successful processing flow."""
        # Basic successful mock setup
        mock_upload_file = UploadFile(filename="test-skill.zip", file=io.BytesIO(TEST_CONTENT))
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

        monkeypatch.setattr("app.api.v1.routers.agents.process_skill", mock_process_skill)
        monkeypatch.setattr(
            "app.api.v1.routers.agents.push_to_nexus",
            lambda *args, **kwargs: (True, None),
        )
        monkeypatch.setattr(
            "app.api.v1.routers.agents.create_skill_zip", lambda *args, **kwargs: io.BytesIO(b"packed content")
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
    ) -> None:
        """Test successful agent configuration."""
        response = post_request_factory()
        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"

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
                {},  # Empty headers - no auth
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Should require authorization",
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
        data_fields: dict | None,
        files_fields: dict | None,
        headers: dict[str, str] | None,
        expected_status: int,
        error_msg: str,
        custom_setup: dict | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test validation errors and error handling."""
        # Handle the process_skill_error case separately
        if test_id == "process_skill_error":
            with monkeypatch.context() as mp:
                # Setup basic mocks
                mock_upload_file = UploadFile(filename="test-skill.zip", file=io.BytesIO(TEST_CONTENT))
                mp.setattr(
                    "app.api.v1.routers.agents.extract_skill_files",
                    AsyncMock(return_value={TEST_SKILL_KEY: mock_upload_file}),
                )
                mp.setattr(
                    "app.api.v1.routers.agents.read_skills_content",
                    AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
                )

                # Apply custom mocks from the setup
                if custom_setup:
                    for attr, mock_value in custom_setup.items():
                        if attr == "mock_process_skill":
                            mp.setattr("app.api.v1.routers.agents.process_skill", mock_value)

                response = post_request_factory()

            assert response.status_code == expected_status, error_msg
            events = parse_streaming_response(response)
            error_events = [e for e in events if e.get("success") is False]
            assert error_events, "Should contain error event"
            assert error_events[-1]["message"] == "Error processing agents", "Should report error"
            return

        # For tests that need project_uuid, replace the placeholder with the real UUID
        if test_id == "missing_authorization" and data_fields and "project_uuid" in data_fields:
            data_fields["project_uuid"] = str(project_uuid)

        # For regular validation tests
        response = custom_post_request_factory(data_fields or {}, files_fields or {}, headers)
        assert response.status_code == expected_status, error_msg

    def test_push_to_nexus_error_response(self, post_request_factory: Callable[[], Any]) -> None:
        """Test handling of error response from push_to_nexus."""
        with pytest.MonkeyPatch().context() as mp:
            # Success setup for early steps
            mock_upload_file = UploadFile(filename="test-skill.zip", file=io.BytesIO(TEST_CONTENT))
            mp.setattr(
                "app.api.v1.routers.agents.extract_skill_files",
                AsyncMock(return_value={TEST_SKILL_KEY: mock_upload_file}),
            )
            mp.setattr(
                "app.api.v1.routers.agents.read_skills_content",
                AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
            )
            process_response = {
                "message": "Skill processed successfully",
                "data": {"skill_name": TEST_SKILL, "agent_name": TEST_AGENT, "size_kb": 1.0},
                "success": True,
                "code": "SKILL_PROCESSED",
            }
            mp.setattr(
                "app.api.v1.routers.agents.process_skill",
                AsyncMock(return_value=(process_response, io.BytesIO(b"processed content"))),
            )

            # Mock nexus error response
            error_response = {
                "message": "Failed to push agents to Nexus",
                "data": {"error": "Nexus API returned an error"},
                "success": False,
                "code": "NEXUS_ERROR",
            }
            mp.setattr(
                "app.api.v1.routers.agents.push_to_nexus",
                lambda *args, **kwargs: (False, error_response),
            )

            response = post_request_factory()

        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"
        events = parse_streaming_response(response)
        error_events = [e for e in events if e.get("success") is False]
        assert error_events, "Should contain error event"
        assert "Error processing agents" in error_events[-1]["message"], "Should report error"

    def test_final_success_message(self, post_request_factory: Callable[[], Any]) -> None:
        """Test the final success message in streaming response."""
        with pytest.MonkeyPatch().context() as mp:
            # Setup success mocks
            mock_upload_file = UploadFile(filename="test-skill.zip", file=io.BytesIO(TEST_CONTENT))
            mp.setattr(
                "app.api.v1.routers.agents.extract_skill_files",
                AsyncMock(return_value={TEST_SKILL_KEY: mock_upload_file}),
            )
            mp.setattr(
                "app.api.v1.routers.agents.read_skills_content",
                AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
            )
            process_response = {
                "message": "Skill processed successfully",
                "data": {"skill_name": TEST_SKILL, "agent_name": TEST_AGENT, "size_kb": 1.0},
                "success": True,
                "code": "SKILL_PROCESSED",
            }
            mp.setattr(
                "app.api.v1.routers.agents.process_skill",
                AsyncMock(return_value=(process_response, io.BytesIO(b"processed content"))),
            )
            mp.setattr(
                "app.api.v1.routers.agents.push_to_nexus",
                lambda *args, **kwargs: (True, None),
            )

            response = post_request_factory()

        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"
        events = parse_streaming_response(response)
        final_event = events[-1]
        assert final_event["success"] is True, "Final event should be success"
        assert final_event["progress"] == 1.0, "Progress should be 100%"
        assert final_event["code"] == "PROCESSING_COMPLETED", "Should report completion"

    def test_push_to_nexus_with_no_skills(self, post_request_factory: Callable[[], Any]) -> None:
        """Test that Nexus push happens even when there are no skills."""
        nexus_called = False

        def mock_push_agents(*args: Any, **kwargs: Any) -> Any:
            nonlocal nexus_called
            nexus_called = True
            # Create a mock response object with status_code
            return type("MockResponse", (), {"status_code": status.HTTP_200_OK, "text": "Success"})()

        with pytest.MonkeyPatch().context() as mp:
            # Mock empty skills
            mp.setattr(
                "app.api.v1.routers.agents.extract_skill_files",
                AsyncMock(return_value={}),
            )
            mp.setattr(
                "app.api.v1.routers.agents.read_skills_content",
                AsyncMock(return_value=[]),
            )

            # Mock Nexus client to verify it's called
            mock_nexus = type("MockNexus", (), {"push_agents": mock_push_agents})()
            mp.setattr("app.api.v1.routers.agents.NexusClient", lambda *a, **k: mock_nexus)

            response = post_request_factory()

        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK"
        events = parse_streaming_response(response)

        # Check that Nexus upload started event exists
        nexus_events = [e for e in events if e.get("code") == "NEXUS_UPLOAD_STARTED"]
        assert nexus_events, "Should include Nexus upload event even with no skills"

        # Check that we have a final success message
        final_events = [e for e in events if e.get("code") == "PROCESSING_COMPLETED"]
        assert final_events, "Should include final completion event"
        assert final_events[-1]["progress"] == 1.0, "Final progress should be 100%"

        # Verify Nexus client was called
        assert nexus_called, "NexusClient.push_agents should be called even with no skills"

    def test_process_skill_failure_stops_processing(self, post_request_factory: Callable[[], Any]) -> None:
        """Test that processing stops immediately if a skill fails to process."""
        with pytest.MonkeyPatch().context() as mp:
            # Setup basic mocks
            mock_upload_file = UploadFile(filename="test-skill.zip", file=io.BytesIO(TEST_CONTENT))
            mp.setattr(
                "app.api.v1.routers.agents.extract_skill_files",
                AsyncMock(return_value={TEST_SKILL_KEY: mock_upload_file}),
            )
            mp.setattr(
                "app.api.v1.routers.agents.read_skills_content",
                AsyncMock(return_value=[(TEST_SKILL_KEY, TEST_CONTENT)]),
            )

            # Mock process_skill to return error response and None for skill_zip
            error_response = {
                "message": "Failed to process skill",
                "data": {"error": "Test error message", "skill_name": TEST_SKILL, "agent_name": TEST_AGENT},
                "success": False,
                "code": "SKILL_PROCESSING_ERROR",
            }
            mp.setattr(
                "app.api.v1.routers.agents.process_skill",
                AsyncMock(return_value=(error_response, None)),
            )

            # Create a tracker for push_to_nexus calls
            push_to_nexus_called = False

            def mock_push_to_nexus(*args: Any, **kwargs: Any) -> Any:
                nonlocal push_to_nexus_called
                push_to_nexus_called = True
                return (True, None)

            mp.setattr(
                "app.api.v1.routers.agents.push_to_nexus",
                mock_push_to_nexus,
            )

            response = post_request_factory()

        assert response.status_code == status.HTTP_200_OK, "Should return 200 OK for streaming response"
        events = parse_streaming_response(response)

        # Verify that error response was sent
        error_events = [e for e in events if e.get("success") is False]
        assert error_events, "Should contain error event"
        assert "Error processing agents" in error_events[-1]["message"], "Should include error message"

        # Verify that processing stopped and never reached push_to_nexus
        assert not push_to_nexus_called, "Should not call push_to_nexus after skill processing fails"


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


class TestProcessSkill:
    """Tests for process_skill function."""

    # Common test data
    folder_zip: ClassVar[bytes] = b"test zip content"
    key: ClassVar[str] = TEST_SKILL_KEY
    project_uuid: ClassVar[str] = str(uuid4())
    agent_name: ClassVar[str] = TEST_AGENT
    skill_name: ClassVar[str] = TEST_SKILL
    skill_definition: ClassVar[dict[str, Any]] = {
        "agents": {
            TEST_AGENT: {
                "slug": TEST_AGENT,
                "skills": [
                    {
                        "slug": TEST_SKILL,
                        "source": {"entrypoint": "main.TestSkill"},
                    }
                ],
            }
        }
    }
    processed_count: ClassVar[int] = 1
    total_count: ClassVar[int] = 2
    expected_progress: ClassVar[float] = 0.55  # 1/2 * 0.7 + 0.2
    toolkit_version: ClassVar[str] = "1.0.0"  # Add default toolkit version

    def test_success(self) -> None:
        """Test successful processing of a skill."""
        import asyncio

        from app.api.v1.routers.agents import process_skill

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "app.api.v1.routers.agents.create_skill_zip", lambda *args, **kwargs: io.BytesIO(b"packaged content")
            )

            response, skill_zip = asyncio.run(
                process_skill(
                    self.folder_zip,
                    self.key,
                    self.project_uuid,
                    self.agent_name,
                    self.skill_name,
                    self.skill_definition,
                    self.processed_count,
                    self.total_count,
                    self.toolkit_version,  # Add toolkit version parameter
                )
            )

        assert response["success"] is True, "Should report success"
        assert skill_zip is not None, "Should return skill zip"
        assert response["data"] is not None, "Response should include data"
        assert response["data"]["skill_name"] == self.skill_name, "Should include skill name"
        assert response["progress"] == self.expected_progress, f"Progress should be {self.expected_progress}"

    def test_missing_agent_in_definition(self) -> None:
        """Test handling of missing agent in definition."""
        import asyncio

        from app.api.v1.routers.agents import process_skill

        # Use a non-existent agent name
        missing_agent = "non-existent-agent"

        response, skill_zip = asyncio.run(
            process_skill(
                self.folder_zip,
                f"{missing_agent}:{self.skill_name}",
                self.project_uuid,
                missing_agent,  # Use the missing agent name
                self.skill_name,
                self.skill_definition,  # Definition only contains TEST_AGENT
                self.processed_count,
                self.total_count,
                self.toolkit_version,  # Add toolkit version parameter
            )
        )

        assert response["success"] is False, "Should report failure"
        assert skill_zip is None, "Should not return skill zip"
        assert response["data"] is not None, "Response should include data"
        assert "Could not find agent" in response["data"]["error"], "Error should mention missing agent"
        assert missing_agent in response["data"]["error"], "Error should include the missing agent name"

    @pytest.mark.parametrize(
        "scenario, key, skill_name, expected_error",
        [
            ("missing_skill", "test-agent:missing-skill", "missing-skill", "Could not find skill"),
            ("exception", TEST_SKILL_KEY, TEST_SKILL, "Zip creation error"),
        ],
    )
    def test_error_scenarios(self, scenario: str, key: str, skill_name: str, expected_error: str) -> None:
        """Test various error scenarios in skill processing."""
        import asyncio

        from app.api.v1.routers.agents import process_skill

        with pytest.MonkeyPatch().context() as mp:
            if scenario == "exception":
                # Cause an exception
                mp.setattr(
                    "app.api.v1.routers.agents.create_skill_zip",
                    lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Zip creation error")),
                )

            response, skill_zip = asyncio.run(
                process_skill(
                    self.folder_zip,
                    key,
                    self.project_uuid,
                    self.agent_name,
                    skill_name,
                    self.skill_definition,
                    self.processed_count,
                    self.total_count,
                    self.toolkit_version,  # Add toolkit version parameter
                )
            )

        assert response["success"] is False, "Should report failure"
        assert response["data"] is not None, "Response should include data"
        assert expected_error in response["data"]["error"], f"Should include '{expected_error}' in error message"
        assert skill_zip is None, "Should not return skill zip"


class TestPushToNexus:
    """Tests for push_to_nexus function."""

    # Common test data
    project_uuid: ClassVar[str] = str(uuid4())
    definition: ClassVar[dict[str, Any]] = {"agents": {TEST_AGENT: {"skills": []}}}
    skill_mapping: ClassVar[dict[str, Any]] = {TEST_SKILL_KEY: io.BytesIO(b"skill content")}
    request_id: ClassVar[str] = str(uuid4())
    authorization: ClassVar[str] = TEST_TOKEN

    def test_success(self) -> None:
        """Test successful push to Nexus."""
        from app.api.v1.routers.agents import push_to_nexus

        with pytest.MonkeyPatch().context() as mp:
            # Create a mock response with status_code
            mock_response = type("MockResponse", (), {"status_code": status.HTTP_200_OK, "text": "Success"})
            mock_client = type("MockNexusClient", (), {"push_agents": lambda self, *args, **kwargs: mock_response()})
            mp.setattr("app.api.v1.routers.agents.NexusClient", lambda auth: mock_client())

            success, response = push_to_nexus(
                self.project_uuid, self.definition, self.skill_mapping, self.request_id, self.authorization
            )

        assert success is True, "Should report success"
        assert response is None, "Should not return a response"

    def test_exception(self) -> None:
        """Test exception handling in Nexus push."""
        from app.api.v1.routers.agents import push_to_nexus

        with pytest.MonkeyPatch().context() as mp:
            # Mock exception
            class MockNexusClient:
                def push_agents(self, *args: Any, **kwargs: Any) -> None:
                    raise RuntimeError("API error")

            mp.setattr("app.api.v1.routers.agents.NexusClient", lambda auth: MockNexusClient())

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
    def test_non_200_status_codes(self, status_code: int, status_text: str, expected_error_fragment: str) -> None:
        """Test handling of various non-200 status codes from Nexus API."""
        from app.api.v1.routers.agents import push_to_nexus

        with pytest.MonkeyPatch().context() as mp:
            # Create a mock response with the specified non-200 status code
            error_response = type("MockResponse", (), {"status_code": status_code, "text": status_text})
            mock_client = type("MockNexusClient", (), {"push_agents": lambda self, *args, **kwargs: error_response()})
            mp.setattr("app.api.v1.routers.agents.NexusClient", lambda auth: mock_client())

            success, response = push_to_nexus(
                self.project_uuid, self.definition, self.skill_mapping, self.request_id, self.authorization
            )

        assert success is False, f"Should report failure when status code is {status_code}"
        assert response is not None, "Should return an error response"
        assert response["success"] is False, "Response should indicate failure"
        assert response["data"] is not None, "Response should include data"
        assert expected_error_fragment in response["data"]["error"], "Should include status code in error"
        assert status_text in response["data"]["error"], "Should include response text in error message"
