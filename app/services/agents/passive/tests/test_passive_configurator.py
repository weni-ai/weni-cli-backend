import json
from typing import Any

import pytest
from fastapi import status
from fastapi.responses import StreamingResponse
from pytest_mock import MockerFixture

from app.core.response import CLIResponse
from app.services.agents.passive.configurator import PassiveAgentConfigurator

NEXUS_UPLOAD_ERROR_PROGRESS = 0.9
EXPECTED_CHUNKS_TOOL_FAILURE = 3
EXPECTED_CHUNKS_NEXUS_FAILURE = 5
EXPECTED_SEND_RESPONSE_CALLS_EMPTY_INPUT = 3


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockerFixture:
    return mocker.patch("app.services.agents.passive.configurator.logger", autospec=True)


@pytest.fixture
def mock_send_response(mocker: MockerFixture) -> Any:
    mock = mocker.patch("app.services.agents.passive.configurator.send_response", autospec=True)
    mock.side_effect = lambda data, request_id: (json.dumps(data) + "\n").encode("utf-8")
    return mock


@pytest.fixture
def mock_process_tool(mocker: MockerFixture) -> Any:
    async def _mock_process_tool(*args: Any, **kwargs: Any) -> tuple[CLIResponse, bytes | None]:
        key = args[1]  # Assuming key is the second argument
        processed_count = args[6] # Assuming processed_count is the seventh argument
        response: CLIResponse = {
            "message": f"Tool {key} processed successfully",
            "data": {"key": key, "status": "success"},
            "success": True,
            "progress": processed_count * 0.1, # Example progress
            "code": "TOOL_PROCESSED",
        }
        tool_zip_bytes = f"zip_content_for_{key}".encode()
        return response, tool_zip_bytes

    return mocker.patch("app.services.agents.passive.configurator.process_tool", side_effect=_mock_process_tool)


@pytest.fixture
def mock_nexus_client(mocker: MockerFixture) -> MockerFixture:
    mock_client_class = mocker.patch("app.services.agents.passive.configurator.NexusClient", autospec=True)
    mock_client_instance = mock_client_class.return_value
    mock_response = mocker.MagicMock()
    mock_response.status_code = status.HTTP_200_OK
    mock_client_instance.push_agents.return_value = mock_response
    return mock_client_class


@pytest.fixture
def configurator(
    mock_logger: MockerFixture,
    mock_send_response: Any,
    mock_process_tool: Any,
    mock_nexus_client: MockerFixture,
) -> PassiveAgentConfigurator:
    project_uuid = "test-project-uuid"
    # Define a minimal valid definition structure
    definition: dict[str, Any] = {
        "agents": {
            "agent1": {
                "tools": [
                    {"name": "tool1", "source": {"entrypoint": "old_entrypoint"}}
                ]
            }
        }
    }
    toolkit_version = "1.0.0"
    request_id = "test-request-id"
    authorization = "test-auth-token"
    return PassiveAgentConfigurator(project_uuid, definition, toolkit_version, request_id, authorization)


# --- Test Cases for configure_agents ---

@pytest.mark.asyncio
async def test_configure_agents_success(
    mocker: MockerFixture,
    configurator: PassiveAgentConfigurator,
    mock_send_response: Any,
    mock_process_tool: Any,
    mock_nexus_client: Any,
) -> None:
    """Tests the successful configuration and push of passive agents."""
    agent_resources_entries = [
        ("agent1:tool1", b"tool1_zip_content"),
        ("agent1:tool2", b"tool2_zip_content"),
    ]
    resource_count = len(agent_resources_entries)

    response = configurator.configure_agents(agent_resources_entries)
    assert isinstance(response, StreamingResponse)

    stream_content = b""
    async for chunk in response.body_iterator:
        assert isinstance(chunk, bytes), f"Expected bytes, got {type(chunk)}"
        stream_content += chunk

    # Verify initial message
    expected_initial_data: CLIResponse = {
        "message": "Processing agents...",
        "data": {"project_uuid": configurator.project_uuid, "total_files": resource_count},
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    }
    mock_send_response.assert_any_call(expected_initial_data, request_id=configurator.request_id)

    # Verify process_tool calls
    assert mock_process_tool.call_count == resource_count
    mock_process_tool.assert_any_call(
        b"tool1_zip_content",
        "agent1:tool1",
        configurator.project_uuid,
        "agent1",
        "tool1",
        configurator.definition,
        1, # processed_count for first call
        resource_count,
        configurator.toolkit_version,
    )
    mock_process_tool.assert_any_call(
        b"tool2_zip_content",
        "agent1:tool2",
        configurator.project_uuid,
        "agent1",
        "tool2",
        configurator.definition,
        2, # processed_count for second call
        resource_count,
        configurator.toolkit_version,
    )

    # Verify responses sent for each tool
    response1, _ = await mock_process_tool.side_effect(
        b"tool1_zip_content", "agent1:tool1", configurator.project_uuid, "agent1", "tool1",
        configurator.definition, 1, resource_count, configurator.toolkit_version
    )
    response2, _ = await mock_process_tool.side_effect(
        b"tool2_zip_content", "agent1:tool2", configurator.project_uuid, "agent1", "tool2",
        configurator.definition, 2, resource_count, configurator.toolkit_version
    )
    mock_send_response.assert_any_call(response1, request_id=configurator.request_id)
    mock_send_response.assert_any_call(response2, request_id=configurator.request_id)


    # Verify Nexus upload started message
    expected_nexus_start_data: CLIResponse = {
        "message": "Updating your agents...",
        "data": {"project_uuid": configurator.project_uuid, "tool_count": resource_count},
        "success": True,
        "code": "NEXUS_UPLOAD_STARTED",
        "progress": 0.99,
    }
    mock_send_response.assert_any_call(expected_nexus_start_data, request_id=configurator.request_id)

    # Verify Nexus client call
    mock_nexus_client.assert_called_once_with(configurator.authorization, configurator.project_uuid)
    expected_tool_mapping = {
        "agent1:tool1": b"zip_content_for_agent1:tool1",
        "agent1:tool2": b"zip_content_for_agent1:tool2",
    }
    # Check that entrypoint was modified before push
    expected_definition = configurator.definition.copy()
    expected_definition["agents"]["agent1"]["tools"][0]["source"]["entrypoint"] = "lambda_function.lambda_handler"
    mock_nexus_client.return_value.push_agents.assert_called_once_with(
       expected_definition, expected_tool_mapping
    )

    # Verify final message
    expected_final_data: CLIResponse = {
        "message": "Agents processed successfully",
        "data": {
            "project_uuid": configurator.project_uuid,
            "total_files": resource_count,
            "processed_files": resource_count,
            "status": "completed",
        },
        "success": True,
        "code": "PROCESSING_COMPLETED",
        "progress": 1.0,
    }
    mock_send_response.assert_any_call(expected_final_data, request_id=configurator.request_id)

    # Total calls: initial + 2 tools + nexus_start + final
    assert mock_send_response.call_count == 1 + resource_count + 1 + 1


@pytest.mark.asyncio
async def test_configure_agents_tool_processing_failure(
    mocker: MockerFixture,
    configurator: PassiveAgentConfigurator,
    mock_send_response: Any,
    mock_process_tool: Any,
    mock_nexus_client: Any,
) -> None:
    """Tests agent configuration when tool processing fails."""
    agent_resources_entries = [("agent1:tool1", b"tool1_zip_content")]
    resource_count = len(agent_resources_entries)
    expected_error_msg_from_tool = "Failed to process tool agent1:tool1: Simulated processing error"

    # Setup mock_process_tool to simulate failure
    failed_tool_response: CLIResponse = {
        "message": "Tool processing failed",
        "data": {"key": "agent1:tool1", "error": "Simulated processing error"},
        "success": False,
        "progress": 0.1,
        "code": "TOOL_PROCESSING_ERROR",
    }
    mock_process_tool.side_effect = [(failed_tool_response, None)] # Return no zip bytes on failure

    response = configurator.configure_agents(agent_resources_entries)

    # Consume the stream
    stream_chunks = []
    async for chunk in response.body_iterator:
        assert isinstance(chunk, bytes), f"Expected bytes, got {type(chunk)}"
        stream_chunks.append(json.loads(chunk.decode("utf-8")))

    # Check that the initial message and the failed tool response were sent
    expected_initial_data: CLIResponse = {
        "message": "Processing agents...",
        "data": {"project_uuid": configurator.project_uuid, "total_files": resource_count},
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    }
    # Check that expected_initial_data is in the stream_chunks
    assert any(item == expected_initial_data for item in stream_chunks)
    # Check that failed_tool_response is in the stream_chunks
    assert any(item == failed_tool_response for item in stream_chunks)


    # Check the final error message sent by the exception handler
    assert len(stream_chunks) > 0
    final_error_data = stream_chunks[-1] # The last response should be the overall error

    assert final_error_data["success"] is False
    assert final_error_data["code"] == "PROCESSING_ERROR"
    assert final_error_data["data"] is not None
    assert expected_error_msg_from_tool in final_error_data["data"]["error"]
    assert final_error_data["data"]["error_type"] == "Exception"

    # Total calls: initial + failed tool + final error
    # We check based on stream_chunks length as mock_send_response calls might be harder to sequence
    assert len(stream_chunks) == EXPECTED_CHUNKS_TOOL_FAILURE
    mock_nexus_client.assert_not_called() # Nexus should not be called if a tool fails


@pytest.mark.asyncio
async def test_configure_agents_nexus_push_failure(
    mocker: MockerFixture,
    configurator: PassiveAgentConfigurator,
    mock_send_response: Any,
    mock_process_tool: Any,
    mock_nexus_client: Any,
) -> None:
    """Tests agent configuration when pushing to Nexus fails."""
    agent_resources_entries = [("agent1:tool1", b"tool1_zip_content")]
    resource_count = len(agent_resources_entries)
    expected_error_msg_from_nexus = "Failed to push agents: 500 Nexus push failed miserably"

    # Setup Nexus client to simulate failure
    mock_client_instance = mock_nexus_client.return_value
    mock_response_obj = mocker.MagicMock()
    mock_response_obj.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_response_obj.text = "Nexus push failed miserably"
    mock_client_instance.push_agents.return_value = mock_response_obj

    response = configurator.configure_agents(agent_resources_entries)

    # Consume the stream
    stream_chunks = []
    async for chunk in response.body_iterator:
        assert isinstance(chunk, bytes), f"Expected bytes, got {type(chunk)}"
        stream_chunks.append(json.loads(chunk.decode("utf-8")))

    # We need to reconstruct the expected processed_tool_response as mock_process_tool.side_effect is an async function
    # For simplicity in this refactor, we'll check for the presence of key parts of the sequence
    assert any(item["code"] == "PROCESSING_STARTED" for item in stream_chunks)

    # The actual processed_tool_response depends on the mock_process_tool's side_effect
    # We can check that a "TOOL_PROCESSED" response was sent
    assert any(item["code"] == "TOOL_PROCESSED" and item["success"] is True for item in stream_chunks)

    expected_nexus_start_data: CLIResponse = {
        "message": "Updating your agents...",
        "data": {"project_uuid": configurator.project_uuid, "tool_count": resource_count},
        "success": True,
        "code": "NEXUS_UPLOAD_STARTED",
        "progress": 0.99,
    }
    assert any(item == expected_nexus_start_data for item in stream_chunks)

    # Verify the Nexus error response sent by push_to_nexus and yielded by configure_agents
    # This is the NEXUS_UPLOAD_ERROR response
    nexus_error_response_from_stream = next(
        (item for item in stream_chunks if item["code"] == "NEXUS_UPLOAD_ERROR"), None
    )
    assert nexus_error_response_from_stream is not None
    assert nexus_error_response_from_stream["success"] is False
    assert nexus_error_response_from_stream["data"] is not None
    assert expected_error_msg_from_nexus in nexus_error_response_from_stream["data"]["error"]
    assert nexus_error_response_from_stream["data"]["tools_processed"] == resource_count

    # Verify the final error message sent by the main exception handler in response_stream
    assert len(stream_chunks) > 0
    final_error_data = stream_chunks[-1] # The last response should be the overall error

    assert final_error_data["success"] is False
    assert final_error_data["code"] == "PROCESSING_ERROR"
    assert final_error_data["data"] is not None
    assert expected_error_msg_from_nexus in final_error_data["data"]["error"]
    assert final_error_data["data"]["error_type"] == "Exception"

    # Expected calls: initial + 1 tool + nexus_start + nexus_error (from push_to_nexus) + final_processing_error
    assert len(stream_chunks) == EXPECTED_CHUNKS_NEXUS_FAILURE


@pytest.mark.asyncio
async def test_configure_agents_empty_input(
    configurator: PassiveAgentConfigurator,
    mock_send_response: Any,
    mock_process_tool: Any,
    mock_nexus_client: Any,
) -> None:
    """Tests agent configuration with no input files."""
    agent_resources_entries: list[tuple[str, bytes]] = []
    resource_count = 0

    response = configurator.configure_agents(agent_resources_entries)
    assert isinstance(response, StreamingResponse)

    async for _ in response.body_iterator:
        pass

    # Verify initial message
    expected_initial_data: CLIResponse = {
        "message": "Processing agents...",
        "data": {"project_uuid": configurator.project_uuid, "total_files": resource_count},
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    }
    mock_send_response.assert_any_call(expected_initial_data, request_id=configurator.request_id)

    # Verify process_tool was not called
    mock_process_tool.assert_not_called()

    # Verify Nexus upload started message (even with 0 tools)
    expected_nexus_start_data: CLIResponse = {
        "message": "Updating your agents...",
        "data": {"project_uuid": configurator.project_uuid, "tool_count": 0},
        "success": True,
        "code": "NEXUS_UPLOAD_STARTED",
        "progress": 0.99,
    }
    mock_send_response.assert_any_call(expected_nexus_start_data, request_id=configurator.request_id)

    # Verify Nexus client call (with empty tool_mapping)
    mock_nexus_client.assert_called_once_with(configurator.authorization, configurator.project_uuid)
    expected_tool_mapping: dict[str, bytes] = {}
    # Check that entrypoint was modified before push (even if no tools?)
    # The current logic iterates definition['agents'], so it should modify if agents exist
    expected_definition = configurator.definition.copy()
    # Check if the agent and tool list exist before trying to modify
    if "agent1" in expected_definition["agents"] and expected_definition["agents"]["agent1"].get("tools"):
        expected_definition["agents"]["agent1"]["tools"][0]["source"]["entrypoint"] = "lambda_function.lambda_handler"

    mock_nexus_client.return_value.push_agents.assert_called_once_with(
        expected_definition, expected_tool_mapping
    )


    # Verify final message
    expected_final_data: CLIResponse = {
        "message": "Agents processed successfully",
        "data": {
            "project_uuid": configurator.project_uuid,
            "total_files": resource_count,
            "processed_files": 0,
            "status": "completed",
        },
        "success": True,
        "code": "PROCESSING_COMPLETED",
        "progress": 1.0,
    }
    mock_send_response.assert_any_call(expected_final_data, request_id=configurator.request_id)

    # Total calls: initial + nexus_start + final
    assert mock_send_response.call_count == EXPECTED_SEND_RESPONSE_CALLS_EMPTY_INPUT


# --- Test Cases for push_to_nexus ---

def test_push_to_nexus_success(
    configurator: PassiveAgentConfigurator,
    mock_nexus_client: Any,
    mock_logger: Any
) -> None:
    """Tests the successful push to Nexus."""
    tool_mapping = {"agent1:tool1": b"lambda1"}
    success, response_data = configurator.push_to_nexus(tool_mapping)

    assert success is True
    assert response_data is None
    mock_nexus_client.assert_called_once_with(configurator.authorization, configurator.project_uuid)

    # Check that entrypoint was modified
    expected_definition = configurator.definition.copy()
    expected_definition["agents"]["agent1"]["tools"][0]["source"]["entrypoint"] = "lambda_function.lambda_handler"

    mock_nexus_client.return_value.push_agents.assert_called_once_with(
        expected_definition, tool_mapping
    )
    mock_logger.info.assert_any_call(
        f"Sending {len(tool_mapping)} processed tools to Nexus for project {configurator.project_uuid}"
    )
    mock_logger.info.assert_any_call(f"Successfully pushed agents to Nexus for project {configurator.project_uuid}")


def test_push_to_nexus_http_error(
    mocker: MockerFixture,
    configurator: PassiveAgentConfigurator,
    mock_nexus_client: Any,
    mock_logger: Any
) -> None:
    """Tests push to Nexus when the HTTP request fails."""
    mock_client_instance = mock_nexus_client.return_value
    mock_response_obj = mocker.MagicMock()
    mock_response_obj.status_code = status.HTTP_400_BAD_REQUEST
    mock_response_obj.text = "Bad Request Data"
    mock_client_instance.push_agents.return_value = mock_response_obj

    tool_mapping = {"agent1:tool1": b"lambda1"}
    success, response_data = configurator.push_to_nexus(tool_mapping)

    assert success is False
    assert isinstance(response_data, dict)
    expected_error_msg = "Failed to push agents: 400 Bad Request Data"
    assert response_data["message"] == "Failed to push agents"
    assert response_data["data"] is not None # Add assertion for type checker
    assert response_data["data"]["error"] == expected_error_msg
    assert response_data["data"]["tools_processed"] == len(tool_mapping)
    assert response_data["success"] is False
    assert response_data["code"] == "NEXUS_UPLOAD_ERROR"
    assert response_data["progress"] == NEXUS_UPLOAD_ERROR_PROGRESS

    mock_logger.error.assert_called_once_with(
        f"Failed to push agents to Nexus: {expected_error_msg}", exc_info=True
    )


def test_push_to_nexus_exception(
    mocker: MockerFixture,
    configurator: PassiveAgentConfigurator,
    mock_nexus_client: Any,
    mock_logger: Any
) -> None:
    """Tests push to Nexus when the client raises an exception."""
    mock_client_instance = mock_nexus_client.return_value
    mock_client_instance.push_agents.side_effect = Exception("Connection Refused")

    tool_mapping = {"agent1:tool1": b"lambda1"}
    success, response_data = configurator.push_to_nexus(tool_mapping)

    assert success is False
    assert isinstance(response_data, dict)
    expected_error_msg = "Connection Refused"
    assert response_data["message"] == "Failed to push agents"
    assert response_data["data"] is not None # Add assertion for type checker
    assert response_data["data"]["error"] == expected_error_msg
    assert response_data["data"]["tools_processed"] == len(tool_mapping)
    assert response_data["success"] is False
    assert response_data["code"] == "NEXUS_UPLOAD_ERROR"
    assert response_data["progress"] == NEXUS_UPLOAD_ERROR_PROGRESS

    mock_logger.error.assert_called_once_with(
        f"Failed to push agents to Nexus: {expected_error_msg}", exc_info=True
    )
