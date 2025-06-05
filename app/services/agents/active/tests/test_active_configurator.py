import json
from io import BytesIO
from typing import Any

import pytest
from fastapi import status
from fastapi.responses import StreamingResponse
from pytest_mock import MockerFixture

from app.core.response import CLIResponse
from app.services.agents.active.configurator import (
    PREPROCESSOR_OUTPUT_EXAMPLE_KEY,
    PREPROCESSOR_RESOURCE_KEY,
    ActiveAgentConfigurator,
)
from app.services.agents.active.models import ActiveAgentResourceModel, Resource, RuleResource

EXPECTED_SEND_RESPONSE_CALLS_SUCCESS = 2


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MockerFixture:
    return mocker.patch("app.services.agents.active.configurator.logger", autospec=True)


@pytest.fixture
def mock_send_response(mocker: MockerFixture) -> Any:
    mock = mocker.patch("app.services.agents.active.configurator.send_response", autospec=True)
    mock.side_effect = lambda data, request_id: (json.dumps(data) + "\n").encode("utf-8")
    return mock


@pytest.fixture
def mock_active_agent_processor(mocker: MockerFixture) -> MockerFixture:
    mock_processor_class = mocker.patch("app.services.agents.active.configurator.ActiveAgentProcessor", autospec=True)
    mock_processor_instance = mock_processor_class.return_value
    mock_processor_instance.process.return_value = BytesIO(b"lambda_zip_content")
    return mock_processor_class


@pytest.fixture
def mock_gallery_client(mocker: MockerFixture) -> MockerFixture:
    mock_client_class = mocker.patch("app.services.agents.active.configurator.GalleryClient", autospec=True)
    mock_client_instance = mock_client_class.return_value
    mock_response = mocker.MagicMock()
    mock_response.status_code = status.HTTP_201_CREATED
    mock_client_instance.push_agents.return_value = mock_response
    return mock_client_class


@pytest.fixture
def configurator(
    mock_logger: MockerFixture,
    mock_send_response: Any,
    mock_active_agent_processor: MockerFixture,
    mock_gallery_client: MockerFixture,
) -> ActiveAgentConfigurator:
    project_uuid = "test-project-uuid"
    definition: dict[str, Any] = {
        "agents": {
            "agent1": {
                "rules": {
                    "rule1": {
                        "template": "rule1_template",
                        "source": {
                            "entrypoint": "rule1.Rule1",
                            "path": "rule1_entrypoint_path",
                        },
                    }
                },
                "pre-processing": {"source": {"entrypoint": "preprocessor.PreProcessor", "path": "preprocessor_path"}},
            }
        }
    }
    toolkit_version = "1.0.0"
    request_id = "test-request-id"
    authorization = "test-auth-token"
    return ActiveAgentConfigurator(project_uuid, definition, toolkit_version, request_id, authorization)


@pytest.mark.asyncio
async def test_configure_agents_success(
    mocker: MockerFixture,
    configurator: ActiveAgentConfigurator,
    mock_send_response: Any,
    mock_active_agent_processor: Any,
    mock_gallery_client: Any,
) -> None:
    """Tests the successful configuration of agents."""
    agent_resources_entries = [
        ("agent1:rule1", b"rule1_content"),
        (f"agent1:{PREPROCESSOR_RESOURCE_KEY}", b"preprocessor_content"),
        (f"agent1:{PREPROCESSOR_OUTPUT_EXAMPLE_KEY}", b"example_content"),
    ]
    resource_count = len(agent_resources_entries)

    response = configurator.configure_agents(agent_resources_entries)
    assert isinstance(response, StreamingResponse)

    stream_content = b""
    if hasattr(response.body_iterator, "__aiter__"):
        async for chunk in response.body_iterator:
            assert isinstance(chunk, bytes)
            stream_content += chunk
    elif hasattr(response.body_iterator, "__iter__"):
        for chunk in response.body_iterator:
            assert isinstance(chunk, bytes)
            stream_content += chunk

    expected_initial_data: CLIResponse = {
        "message": "Processing agents...",
        "data": {"project_uuid": configurator.project_uuid, "total_files": resource_count},
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    }
    expected_final_data: CLIResponse = {
        "message": "Agents processed successfully",
        "data": {
            "project_uuid": configurator.project_uuid,
            "total_files": resource_count,
            "processed_files": 1,
            "status": "completed",
        },
        "success": True,
        "code": "PROCESSING_COMPLETED",
        "progress": 1.0,
    }

    mock_send_response.assert_any_call(expected_initial_data, request_id=configurator.request_id)
    mock_send_response.assert_any_call(expected_final_data, request_id=configurator.request_id)
    assert mock_send_response.call_count == EXPECTED_SEND_RESPONSE_CALLS_SUCCESS

    expected_resource_model = ActiveAgentResourceModel(
        preprocessor=Resource(content=b"preprocessor_content", module_name="preprocessor", class_name="PreProcessor"),
        rules=[
            RuleResource(
                key="rule1",
                content=b"rule1_content",
                module_name="rule1",
                class_name="Rule1",
                template="rule1_template",
            )
        ],
        preprocessor_example=b"example_content",
    )
    mock_active_agent_processor.assert_called_once_with(
        configurator.project_uuid, configurator.toolkit_version, expected_resource_model
    )
    mock_active_agent_processor.return_value.process.assert_called_once_with("agent1")

    assert configurator.definition["agents"]["agent1"]["pre-processing"]["result_example"] == b"example_content"

    mock_gallery_client.assert_called_once_with(configurator.project_uuid, configurator.authorization)
    expected_lambda_map = {"agent1": BytesIO(b"lambda_zip_content")}
    call_args, call_kwargs = mock_gallery_client.return_value.push_agents.call_args
    pushed_definition, pushed_lambda_map = call_args
    assert pushed_definition == configurator.definition
    assert list(pushed_lambda_map.keys()) == list(expected_lambda_map.keys())
    assert pushed_lambda_map["agent1"].getvalue() == expected_lambda_map["agent1"].getvalue()


@pytest.mark.asyncio
async def test_configure_agents_push_failure(
    mocker: MockerFixture, configurator: ActiveAgentConfigurator, mock_send_response: Any, mock_gallery_client: Any
) -> None:
    """Tests agent configuration when pushing to gallery fails."""
    agent_resources_entries = [("agent1:rule1", b"rule1_content")]
    resource_count = len(agent_resources_entries)

    mock_client_instance = mock_gallery_client.return_value
    mock_response_obj = mocker.MagicMock()
    mock_response_obj.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_response_obj.text = "Gallery push failed"
    mock_client_instance.push_agents.return_value = mock_response_obj

    response = configurator.configure_agents(agent_resources_entries)

    stream_content = b""
    with pytest.raises(Exception, match="Unknown error while pushing agents..."):
        if hasattr(response.body_iterator, "__aiter__"):
            async for chunk in response.body_iterator:
                assert isinstance(chunk, bytes)
                stream_content += chunk
        elif hasattr(response.body_iterator, "__iter__"):
            for chunk in response.body_iterator:
                assert isinstance(chunk, bytes)
                stream_content += chunk

    expected_initial_data: CLIResponse = {
        "message": "Processing agents...",
        "data": {"project_uuid": configurator.project_uuid, "total_files": resource_count},
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    }
    mock_send_response.assert_any_call(expected_initial_data, request_id=configurator.request_id)


@pytest.mark.asyncio
async def test_configure_agents_processing_error(
    mocker: MockerFixture,
    configurator: ActiveAgentConfigurator,
    mock_active_agent_processor: Any,
    mock_send_response: Any,
) -> None:
    """Tests agent configuration when the agent processor fails."""
    agent_resources_entries = [("agent1:rule1", b"rule1_content")]
    resource_count = len(agent_resources_entries)

    mock_active_agent_processor.return_value.process.return_value = None

    response = configurator.configure_agents(agent_resources_entries)

    with pytest.raises(Exception, match="Error processing agent agent1"):
        if hasattr(response.body_iterator, "__aiter__"):
            async for chunk in response.body_iterator:
                assert isinstance(chunk, bytes)
                pass
        elif hasattr(response.body_iterator, "__iter__"):
            for chunk in response.body_iterator:
                assert isinstance(chunk, bytes)
                pass

    expected_initial_data: CLIResponse = {
        "message": "Processing agents...",
        "data": {"project_uuid": configurator.project_uuid, "total_files": resource_count},
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    }
    mock_send_response.assert_called_once_with(expected_initial_data, request_id=configurator.request_id)
    mock_active_agent_processor.return_value.process.assert_called_once_with("agent1")


def test_push_to_gallery_success(
    mocker: MockerFixture, configurator: ActiveAgentConfigurator, mock_gallery_client: Any, mock_logger: Any
) -> None:
    """Tests the successful push to gallery."""
    agents_lambda_map = {"agent1": BytesIO(b"lambda1")}
    success, response_data = configurator.push_to_gallery(agents_lambda_map)

    assert success is True
    assert response_data is None
    mock_gallery_client.assert_called_once_with(configurator.project_uuid, configurator.authorization)
    mock_gallery_client.return_value.push_agents.assert_called_once_with(configurator.definition, agents_lambda_map)
    mock_logger.info.assert_any_call(
        f"Sending {len(agents_lambda_map)} processed agents to Gallery for project {configurator.project_uuid}"
    )
    mock_logger.info.assert_any_call(f"Successfully pushed agents to Gallery for project {configurator.project_uuid}")


def test_push_to_gallery_http_error(
    mocker: MockerFixture, configurator: ActiveAgentConfigurator, mock_gallery_client: Any, mock_logger: Any
) -> None:
    """Tests push to gallery when the HTTP request fails with 400 Bad Request."""
    mock_client_instance = mock_gallery_client.return_value
    mock_response_obj = mocker.MagicMock()
    mock_response_obj.status_code = status.HTTP_400_BAD_REQUEST
    mock_response_obj.text = "Bad Request"
    mock_client_instance.push_agents.return_value = mock_response_obj

    agents_lambda_map = {"agent1": BytesIO(b"lambda1")}
    success, response_data = configurator.push_to_gallery(agents_lambda_map)

    assert success is False
    assert isinstance(response_data, dict)
    assert response_data["message"] == "Failed to push agents to Gallery: 400 Bad Request"
    assert response_data["success"] is False
    mock_logger.error.assert_called_once()


def test_push_to_gallery_400_bad_request_with_details(
    mocker: MockerFixture, configurator: ActiveAgentConfigurator, mock_gallery_client: Any, mock_logger: Any
) -> None:
    """Tests push to gallery with a 400 Bad Request containing detailed error message."""
    mock_client_instance = mock_gallery_client.return_value
    mock_response_obj = mocker.MagicMock()
    mock_response_obj.status_code = status.HTTP_400_BAD_REQUEST
    mock_response_obj.text = json.dumps({"detail": "Invalid agent configuration", "code": "INVALID_CONFIG"})
    mock_client_instance.push_agents.return_value = mock_response_obj

    agents_lambda_map = {"agent1": BytesIO(b"lambda1")}
    success, response_data = configurator.push_to_gallery(agents_lambda_map)

    assert success is False
    assert isinstance(response_data, dict)
    assert response_data["message"] == f"Failed to push agents to Gallery: 400 {mock_response_obj.text}"
    assert response_data["success"] is False
    assert "Invalid agent configuration" in response_data["message"]
    mock_logger.error.assert_called_once()


def test_push_to_gallery_exception(
    mocker: MockerFixture, configurator: ActiveAgentConfigurator, mock_gallery_client: Any, mock_logger: Any
) -> None:
    """Tests push to gallery when the client raises an exception."""
    mock_client_instance = mock_gallery_client.return_value
    mock_client_instance.push_agents.side_effect = Exception("Network Error")

    agents_lambda_map = {"agent1": BytesIO(b"lambda1")}
    success, response_data = configurator.push_to_gallery(agents_lambda_map)

    assert success is False
    assert isinstance(response_data, dict)
    assert response_data["message"] == "Network Error"
    assert response_data["success"] is False
    mock_logger.error.assert_called_once_with("Error pushing agents to Gallery: Network Error", exc_info=True)


# TODO: Add test for the case where agent_key is not found in agents_resources
# (though current logic initializes it, defensive testing might be good)
# Example: test_configure_agents_key_error_handling
# Requires adjusting the loop logic slightly or mocking the split differently.

# TODO: Test edge case with empty agent_resources_entries

@pytest.mark.asyncio
async def test_configure_agents_syntax_error(
    mocker: MockerFixture,
    configurator: ActiveAgentConfigurator,
    mock_active_agent_processor: Any,
    mock_send_response: Any,
) -> None:
    """Tests agent configuration when a syntax error occurs."""
    agent_resources_entries = [("agent1:rule1", b"rule1_content")]
    resource_count = len(agent_resources_entries)

    # Setup mock to raise SyntaxError
    syntax_error_msg = (
        "invalid syntax in rule_StatusAprovado.py, line 25\n" +
        "    if status == 'APPROVED'\n" +
        "                          ^\n" +
        "SyntaxError: expected ':'"
    )
    mock_active_agent_processor.return_value.process.side_effect = SyntaxError(syntax_error_msg)

    response_stream = configurator.configure_agents(agent_resources_entries)
    responses = [r async for r in response_stream]

    assert len(responses) == 2  # Initial message + error message
    assert responses[0].body.decode() == json.dumps({
        "message": "Starting agent processing...",
        "data": {
            "project_uuid": str(configurator.project_uuid),
            "total_files": resource_count,
        },
        "success": True,
        "progress": 0.01,
        "code": "PROCESSING_STARTED",
    })

    error_response = json.loads(responses[1].body.decode())
    assert error_response["success"] is False
    assert error_response["message"] == f"Syntax error processing agent agent1: {syntax_error_msg}"
    assert error_response["data"]["error_code"] == "SYNTAX_ERROR"
    assert error_response["data"]["status_code"] == 400
    assert error_response["data"]["agent_key"] == "agent1"
    assert error_response["data"]["error_details"] == syntax_error_msg
