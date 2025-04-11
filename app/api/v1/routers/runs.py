"""
Tool runs endpoints for handling file uploads and processing.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from app.api.v1.models.requests import RunToolRequestModel
from app.clients.aws import AWSLambdaClient
from app.core.response import CLIResponse, send_response
from app.services.tool.packager import process_tool

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def run_tool_test(  # noqa: PLR0915
    request: Request,
    data: Annotated[RunToolRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    request_id = str(uuid4())
    logger.info(f"Processing test run for project {data.project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {data.definition}")

    # Access the form data with files
    form = await request.form()

    # Extract and process tool files
    tool_folder_zip = form.get("tool")

    if not tool_folder_zip or not isinstance(tool_folder_zip, UploadFile):
        raise HTTPException(status_code=400, detail="Tool folder zip is required")

    folder_zip = await tool_folder_zip.read()

    logger.info(f"Found tool folder to process for project {data.project_uuid}")
    logger.debug(f"Tool folder zip: {tool_folder_zip}")

    function_name = f"cli-{str(uuid4())}"
    lambda_client = AWSLambdaClient()

    async def response_stream() -> AsyncIterator[bytes]:
        try:
            initial_data: CLIResponse = {
                "message": "Processing test run...",
                "data": {
                    "project_uuid": str(data.project_uuid),
                },
                "success": True,
                "code": "PROCESSING_STARTED",
            }
            yield send_response(initial_data, request_id=request_id)
            logger.info(f"Starting test run for tool {data.tool_key} by {data.agent_key}")

            # Get tool entrypoint
            agent_info = data.definition["agents"].get(data.agent_key)

            if not agent_info:
                raise ValueError(f"Could not find agent {data.agent_key} in definition")

            logger.debug(f"Agent info: {agent_info}")

            tool_info = next(
                (tool for tool in agent_info["tools"] if tool["key"] == data.tool_key),
                None,
            )

            if not tool_info:
                raise ValueError(f"Could not find tool {data.tool_key} for agent {data.agent_key} in definition")

            logger.debug(f"Tool info: {tool_info}")

            # Process the tool folder zip
            response, tool_zip_bytes = await process_tool(
                folder_zip,
                f"{data.agent_key}:{data.tool_key}",
                str(data.project_uuid),
                data.agent_key,
                data.tool_key,
                data.definition,
                1,
                1,
                data.toolkit_version,
            )

            # Send response for this tool
            yield send_response(response, request_id=request_id)

            if not tool_zip_bytes:
                raise ValueError("Failed to process tool, aborting test run")

            response = {
                "message": "Creating lambda function",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "function_name": function_name,
                },
                "success": True,
                "code": "LAMBDA_FUNCTION_CREATING",
            }
            yield send_response(response, request_id=request_id)

            # Create lambda function
            lambda_function = lambda_client.create_function(
                function_name=function_name,
                handler="lambda_function.lambda_handler",
                code=tool_zip_bytes,
                description=f"CLI run for tool {data.tool_key} by {data.agent_key}. Project: {data.project_uuid}",
            )

            if not lambda_function.function_arn or not lambda_function.function_name:
                raise ValueError("Failed to create lambda function")

            if not await lambda_client.wait_for_function_active(lambda_function.function_arn):
                raise ValueError("Lambda function did not became active")

            logger.info(f"Lambda function {lambda_function.function_arn} active, invoking...")

            response = {
                "message": "Running test cases",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "function_name": function_name,
                },
                "success": True,
                "code": "STARTING_TEST_CASES",
            }
            yield send_response(response, request_id=request_id)

            for test_case, test_data in data.test_definition.get("tests", {}).items():
                logger.info(f"Running test case {test_case} for tool {data.tool_key} by {data.agent_key}")

                response = {
                    "message": f"Running test case {test_case}",
                    "data": {
                        "project_uuid": str(data.project_uuid),
                        "function_name": function_name,
                        "test_case": test_case,
                    },
                    "success": True,
                    "code": "TEST_CASE_RUNNING",
                }
                yield send_response(response, request_id=request_id)

                parameters = []
                for key, value in test_data.get("parameters", {}).items():
                    parameters.append({"name": key, "value": value})

                test_event = {
                    "agent_key": data.agent_key,
                    "action_group": lambda_function.function_name,
                    "function": lambda_function.function_name,
                    "parameters": parameters,
                    "sessionAttributes": {
                        "credentials": json.dumps(test_data.get("credentials", data.tool_credentials)),
                        "globals": json.dumps(test_data.get("globals", data.tool_globals)),
                    },
                }

                # Invoke lambda function
                invoke_result, invoke_start_time, invoke_end_time = lambda_client.invoke_function(
                    lambda_function.function_arn, test_event
                )

                test_response: CLIResponse = {
                    "message": "Test case completed",
                    "data": {
                        "test_case": test_case,
                        "logs": invoke_result.get("logs"),
                        "duration": invoke_end_time - invoke_start_time,
                        "test_status_code": invoke_result.get("status_code"),
                        "test_response": invoke_result.get("response"),
                    },
                    "success": True,
                    "code": "TEST_CASE_COMPLETED",
                }

                yield send_response(test_response, request_id=request_id)

        except Exception as e:
            logger.error(
                f"Error processing test run for project {data.project_uuid}: {str(e)} - request_id: {request_id}"
            )
            error_data: CLIResponse = {
                "message": "Error processing test run",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "error": str(e),
                },
                "success": False,
                "code": "PROCESSING_ERROR",
            }
            yield send_response(error_data, request_id=request_id)
        finally:
            # Delete lambda function
            try:
                lambda_client.delete_function(function_name=function_name)
            except Exception as e:
                logger.error(f"Error deleting lambda function {function_name}: {str(e)}")

    return StreamingResponse(response_stream(), media_type="application/x-ndjson")
