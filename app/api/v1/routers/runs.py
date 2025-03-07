"""
Skill runs endpoints for handling file uploads and processing.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from app.api.v1.models.requests import RunSkillRequestModel
from app.clients.aws import AWSLambdaClient, AWSLogsClient
from app.core.response import CLIResponse, send_response
from app.services.skill.packager import process_skill

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def run_skill_test(  # noqa: PLR0915
    request: Request,
    data: Annotated[RunSkillRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    request_id = str(uuid4())
    logger.info(f"Processing test run for project {data.project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {data.definition}")

    # Access the form data with files
    form = await request.form()

    # Extract and process skill files
    skill_folder_zip = form.get("skill")

    if not skill_folder_zip or not isinstance(skill_folder_zip, UploadFile):
        raise HTTPException(status_code=400, detail="Skill folder zip is required")

    folder_zip = await skill_folder_zip.read()

    logger.info(f"Found skill folder to process for project {data.project_uuid}")
    logger.debug(f"Skill folder zip: {skill_folder_zip}")

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
            logger.info(f"Starting test run for skill {data.skill_name} by {data.agent_name}")

            # Get skill entrypoint
            agent_info = next(
                (agent for agent in data.definition["agents"].values() if agent.get("name") == data.agent_name),
                None,
            )

            if not agent_info:
                raise ValueError(f"Could not find agent {data.agent_name} in definition")

            logger.debug(f"Agent info: {agent_info}")

            skill_info = next(
                (skill for skill in agent_info["skills"] if skill["name"] == data.skill_name),
                None,
            )

            if not skill_info:
                raise ValueError(f"Could not find skill {data.skill_name} for agent {data.agent_name} in definition")

            logger.debug(f"Skill info: {skill_info}")

            # Process the skill folder zip
            response, skill_zip_bytes = await process_skill(
                folder_zip,
                f"{agent_info.get('slug')}:{skill_info.get('slug')}",
                str(data.project_uuid),
                agent_info.get("slug"),
                skill_info.get("slug"),
                data.definition,
                1,
                1,
                data.toolkit_version,
            )

            # Send response for this skill
            yield send_response(response, request_id=request_id)

            if not skill_zip_bytes:
                raise ValueError("Failed to process skill, aborting test run")

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
                code=skill_zip_bytes,
                description=f"CLI run for skill {data.skill_name} by {data.agent_name}. Project: {data.project_uuid}",
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
                logger.info(f"Running test case {test_case} for skill {data.skill_name} by {data.agent_name}")

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
                    "agent_name": data.agent_name,
                    "action_group": lambda_function.function_name,
                    "function": lambda_function.function_name,
                    "parameters": parameters,
                    "sessionAttributes": {
                        "credentials": json.dumps(test_data.get("credentials", data.skill_credentials)),
                        "globals": json.dumps(test_data.get("globals", data.skill_globals)),
                    },
                }

                # Invoke lambda function
                invoke_result, invoke_start_time, invoke_end_time = lambda_client.invoke_function(
                    lambda_function.function_arn, test_event
                )

                # Get lambda function logs
                logs_client = AWSLogsClient()
                logs = await logs_client.get_function_logs(
                    function_name=lambda_function.function_name,
                    request_id=invoke_result.get("request_id"),
                    start_time=invoke_start_time,
                )

                test_response: CLIResponse = {
                    "message": "Test case completed",
                    "data": {
                        "test_case": test_case,
                        "logs": logs,
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
