"""Run strategy for passive agents (Tools)."""

import json
import logging
from collections.abc import AsyncIterator

from app.api.v1.models.requests import RunRequestModel
from app.clients.aws import AWSLambdaClient
from app.clients.aws.lambda_client import LambdaFunction
from app.core.config import settings
from app.core.response import CLIResponse, send_response
from app.services.jwt_generator import JWT_PROJECT_KEY, generate_jwt_token
from app.services.tool.packager import process_tool

logger = logging.getLogger(__name__)


async def run(  # noqa: PLR0915
    data: RunRequestModel,
    folder_zip: bytes,
    function_name: str,
    request_id: str,
    lambda_client: AWSLambdaClient,
) -> AsyncIterator[bytes]:
    """Stream NDJSON responses while running test cases for a tool (passive agent)."""

    if not data.tool_key:
        raise ValueError("tool_key is required for passive (tool) run strategy")
    tool_key = data.tool_key

    initial_data: CLIResponse = {
        "message": "Processing test run...",
        "data": {
            "project_uuid": str(data.project_uuid),
        },
        "success": True,
        "code": "PROCESSING_STARTED",
    }
    yield send_response(initial_data, request_id=request_id)
    logger.info(f"Starting test run for tool {tool_key} by {data.agent_key}")

    agent_info = data.definition["agents"].get(data.agent_key)
    if not agent_info:
        raise ValueError(f"Could not find agent {data.agent_key} in definition")

    logger.debug(f"Agent info: {agent_info}")

    tool_info = next(
        (tool for tool in agent_info["tools"] if tool["key"] == tool_key),
        None,
    )

    if not tool_info:
        raise ValueError(f"Could not find tool {tool_key} for agent {data.agent_key} in definition")

    logger.debug(f"Tool info: {tool_info}")

    response, tool_zip_bytes = await process_tool(
        folder_zip,
        f"{data.agent_key}:{tool_key}",
        str(data.project_uuid),
        data.agent_key,
        tool_key,
        data.definition,
        1,
        1,
        data.toolkit_version,
    )

    yield send_response(response, request_id=request_id)

    if not tool_zip_bytes:
        raise ValueError("Failed to process tool, aborting test run")

    creating_response: CLIResponse = {
        "message": "Creating tool",
        "data": {
            "project_uuid": str(data.project_uuid),
            "function_name": function_name,
        },
        "success": True,
        "code": "LAMBDA_FUNCTION_CREATING",
    }
    yield send_response(creating_response, request_id=request_id)

    lambda_function: LambdaFunction = lambda_client.create_function(
        function_name=function_name,
        handler="lambda_function.lambda_handler",
        code=tool_zip_bytes,
        description=f"CLI run for tool {tool_key} by {data.agent_key}. Project: {data.project_uuid}",
    )

    if not lambda_function.arn or not lambda_function.name:
        raise ValueError("Failed to create tool")

    if not await lambda_client.wait_for_function_active(lambda_function.arn):
        raise ValueError("Tool did not became active")

    logger.info(f"Tool {lambda_function.arn} active, invoking...")

    starting_response: CLIResponse = {
        "message": "Running test cases",
        "data": {
            "project_uuid": str(data.project_uuid),
            "function_name": function_name,
        },
        "success": True,
        "code": "STARTING_TEST_CASES",
    }
    yield send_response(starting_response, request_id=request_id)

    for test_case, test_data in data.test_definition.get("tests", {}).items():
        logger.info(f"Running test case {test_case} for tool {tool_key} by {data.agent_key}")

        running_response: CLIResponse = {
            "message": f"Running test case {test_case}",
            "data": {
                "project_uuid": str(data.project_uuid),
                "function_name": function_name,
                "test_case": test_case,
            },
            "success": True,
            "code": "TEST_CASE_RUNNING",
        }
        yield send_response(running_response, request_id=request_id)

        parameters = []
        for key, value in test_data.get("parameters", {}).items():
            parameters.append({"name": key, "value": value})

        project = test_data.get("project", {})
        if isinstance(project, str):
            project = json.loads(project)

        token = generate_jwt_token(str(data.project_uuid), settings.JWT_SECRET_KEY)
        project[JWT_PROJECT_KEY] = token

        test_event = {
            "agent_key": data.agent_key,
            "action_group": lambda_function.name,
            "function": lambda_function.name,
            "parameters": parameters,
            "sessionAttributes": {
                "project": json.dumps(project),
                "contact": json.dumps(test_data.get("contact", {})),
                "credentials": json.dumps(test_data.get("credentials", data.tool_credentials or {})),
                "globals": json.dumps(test_data.get("globals", data.tool_globals or {})),
            },
        }

        invoke_result, invoke_start_time, invoke_end_time = lambda_client.invoke_function(
            lambda_function.arn, test_event
        )

        response_data = invoke_result.get("response", {})
        has_error = (
            response_data.get("errorMessage") is not None
            or response_data.get("errorType") is not None
            or response_data.get("error") is not None
        )

        test_response: CLIResponse = {
            "message": "Test case completed",
            "data": {
                "test_case": test_case,
                "logs": invoke_result.get("logs"),
                "duration": invoke_end_time - invoke_start_time,
                "test_status_code": 400 if has_error else invoke_result.get("status_code", 200),
                "test_response": invoke_result.get("response"),
            },
            "success": True,
            "code": "TEST_CASE_COMPLETED",
        }

        yield send_response(test_response, request_id=request_id)
