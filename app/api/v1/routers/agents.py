"""
Agents endpoints for handling file uploads and processing.
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Form, Header, Request, status
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from app.api.v1.models.requests import BaseRequestModel
from app.clients.nexus_client import NexusClient
from app.core.response import CLIResponse, send_response
from app.services.tool.packager import process_tool

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def configure_agents(
    request: Request,
    data: Annotated[BaseRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    """
    Upload multiple files for tool processing.

    Args:
        request: FastAPI request object to access form data and files
        project_uuid: UUID of the project
        definition: JSON containing tool definitions
        authorization: Authorization header for Nexus API calls

    Returns:
        StreamingResponse: Streaming response for future result handling
    """
    request_id = str(uuid4())
    logger.info(f"Processing agent configuration for project {data.project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {data.definition}")

    # Access the form data with files
    form = await request.form()

    # Extract and process tool files
    tools_folders_zips = await extract_tool_files(form)
    tool_count = len(tools_folders_zips)
    logger.info(f"Found {tool_count} tool folders to process for project {data.project_uuid}")
    logger.debug(f"Tool keys: {list(tools_folders_zips.keys())}")

    # Create file names list for streaming
    tools_folders_zips_entries = await read_tools_content(tools_folders_zips)

    # Async generator for streaming responses
    async def response_stream() -> AsyncIterator[bytes]:
        try:
            # Initial message
            initial_data: CLIResponse = {
                "message": "Processing agents...",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "total_files": tool_count,
                },
                "success": True,
                "progress": 0.01,
                "code": "PROCESSING_STARTED",
            }
            logger.info(f"Starting processing for {tool_count} tools")
            yield send_response(initial_data, request_id=request_id)

            tool_mapping = {}
            processed_count = 0

            # Process each tool file
            for key, folder_zip in tools_folders_zips_entries:
                agent_key, tool_key = key.split(":")
                processed_count += 1

                # Process the tool
                response, tool_zip_bytes = await process_tool(
                    folder_zip,
                    key,
                    str(data.project_uuid),
                    agent_key,
                    tool_key,
                    data.definition,
                    processed_count,
                    tool_count,
                    data.toolkit_version,
                )

                # Send response for this tool
                yield send_response(response, request_id=request_id)

                # If tool processing failed, stop and raise an exception
                if not tool_zip_bytes:
                    response_data: dict = response.get("data") or {}
                    error_message = response_data.get("error", "Unknown error processing tool")
                    raise Exception(f"Failed to process tool {key}: {error_message}")

                # Add to mapping if successful (we'll only get here if processing succeeded)
                tool_mapping[key] = tool_zip_bytes

            # Send progress update for Nexus upload
            nexus_response: CLIResponse = {
                "message": "Updating your agents...",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "tool_count": len(tool_mapping),
                },
                "success": True,
                "code": "NEXUS_UPLOAD_STARTED",
                "progress": 0.99,
            }
            yield send_response(nexus_response, request_id=request_id)

            # Push to Nexus - always push, even if no tools
            success, error_response = push_to_nexus(
                str(data.project_uuid), data.definition, tool_mapping, request_id, authorization
            )

            if not success and error_response is not None:
                # Type cast error_response to CLIResponse for type checker
                typed_error_response = cast(CLIResponse, error_response)
                yield send_response(typed_error_response, request_id=request_id)

                response_data = typed_error_response.get("data") or {}
                error_message = str(response_data.get("error", "Unknown error while pushing agents..."))
                raise Exception(error_message)

            # Final message
            final_data: CLIResponse = {
                "message": "Agents processed successfully",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "total_files": tool_count,
                    "processed_files": len(tool_mapping),
                    "status": "completed",
                },
                "success": True,
                "code": "PROCESSING_COMPLETED",
                "progress": 1.0,
            }
            yield send_response(final_data, request_id=request_id)

        except Exception as e:
            logger.error(f"Error processing agents for project {data.project_uuid}: {str(e)}", exc_info=True)
            error_data: CLIResponse = {
                "message": "Error processing agents",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                },
                "success": False,
                "code": "PROCESSING_ERROR",
            }
            yield send_response(error_data, request_id=request_id)

    return StreamingResponse(response_stream(), media_type="application/x-ndjson")


# Utility functions below this point
async def extract_tool_files(form: Any) -> dict[str, UploadFile]:
    """
    Extract tool files from form data.

    Args:
        form: The form data containing files (FormData from FastAPI)

    Returns:
        Dictionary of agent:tool keys to UploadFile objects
    """
    tools_folders_zips = {}
    for key, value in form.items():
        if isinstance(value, UploadFile) and ":" in key:
            tools_folders_zips[key] = value

    return tools_folders_zips


async def read_tools_content(tools_folders_zips: dict[str, UploadFile]) -> list[tuple[str, bytes]]:
    """
    Read the content of each tool file.

    Args:
        tools_folders_zips: Dictionary of tool keys to file objects

    Returns:
        List of tuples containing (key, file_content)
    """
    return [(key, await file.read()) for key, file in tools_folders_zips.items()]


def push_to_nexus(
    project_uuid: str,
    definition: dict[str, Any],
    tool_mapping: dict[str, Any],
    request_id: str,
    authorization: str,
) -> tuple[bool, CLIResponse | None]:
    """
    Push processed tools to Nexus.

    Args:
        project_uuid: The UUID of the project
        definition: The agent definition
        tool_mapping: Dictionary of processed tools
        request_id: The request ID for correlation
        authorization: The authorization header value

    Returns:
        Tuple of (success, response)
    """
    try:
        logger.info(f"Sending {len(tool_mapping)} processed tools to Nexus for project {project_uuid}")
        nexus_client = NexusClient(authorization)

        # We need to change the entrypoint to the lambda function we've created
        for _, agent_data in definition["agents"].items():
            for tool in agent_data["tools"]:
                tool["source"]["entrypoint"] = "lambda_function.lambda_handler"

        response = nexus_client.push_agents(str(project_uuid), definition, tool_mapping)

        if response.status_code != status.HTTP_200_OK:
            raise Exception(f"Failed to push agents: {response.status_code} {response.text}")

        logger.info(f"Successfully pushed agents to Nexus for project {project_uuid}")

        return True, None

    except Exception as e:
        logger.error(f"Failed to push agents to Nexus: {str(e)}", exc_info=True)
        nexus_error: CLIResponse = {
            "message": "Failed to push agents",
            "data": {
                "project_uuid": str(project_uuid),
                "error": str(e),
                "tools_processed": len(tool_mapping),
            },
            "success": False,
            "code": "NEXUS_UPLOAD_ERROR",
            "progress": 0.9,
        }

        return False, nexus_error
