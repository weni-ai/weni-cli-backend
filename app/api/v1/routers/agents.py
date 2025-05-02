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
from app.services.skill.packager import process_skill

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def configure_agents(
    request: Request,
    data: Annotated[BaseRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    """
    Upload multiple files for skill processing.

    Args:
        request: FastAPI request object to access form data and files
        project_uuid: UUID of the project
        definition: JSON containing agent definitions
        authorization: Authorization header for Nexus API calls

    Returns:
        StreamingResponse: Streaming response for future result handling
    """
    request_id = str(uuid4())
    logger.info(f"Processing agent configuration for project {data.project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {data.definition}")

    # Access the form data with files
    form = await request.form()

    # Extract and process skill files
    skills_folders_zips = await extract_skill_files(form)
    skill_count = len(skills_folders_zips)
    logger.info(f"Found {skill_count} skill folders to process for project {data.project_uuid}")
    logger.debug(f"Skill keys: {list(skills_folders_zips.keys())}")

    # Create file names list for streaming
    skills_folders_zips_entries = await read_skills_content(skills_folders_zips)

    # Async generator for streaming responses
    async def response_stream() -> AsyncIterator[bytes]:
        try:
            # Initial message
            initial_data: CLIResponse = {
                "message": "Processing agents...",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "total_files": skill_count,
                },
                "success": True,
                "progress": 0.01,
                "code": "PROCESSING_STARTED",
            }
            logger.info(f"Starting processing for {skill_count} skills")
            yield send_response(initial_data, request_id=request_id)

            skill_mapping = {}
            processed_count = 0

            # Process each skill file
            for key, folder_zip in skills_folders_zips_entries:
                agent_slug, skill_slug = key.split(":")
                processed_count += 1

                # Process the skill
                response, skill_zip_bytes = await process_skill(
                    folder_zip,
                    key,
                    str(data.project_uuid),
                    agent_slug,
                    skill_slug,
                    data.definition,
                    processed_count,
                    skill_count,
                    data.toolkit_version,
                )

                # Send response for this skill
                yield send_response(response, request_id=request_id)

                # If skill processing failed, stop and raise an exception
                if not skill_zip_bytes:
                    response_data: dict = response.get("data") or {}
                    error_message = response_data.get("error", "Unknown error processing skill")
                    raise Exception(f"Failed to process skill {key}: {error_message}")

                # Add to mapping if successful (we'll only get here if processing succeeded)
                skill_mapping[key] = skill_zip_bytes

            # Send progress update for Nexus upload
            nexus_response: CLIResponse = {
                "message": "Updating your agents...",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "skill_count": len(skill_mapping),
                },
                "success": True,
                "code": "NEXUS_UPLOAD_STARTED",
                "progress": 0.99,
            }
            yield send_response(nexus_response, request_id=request_id)

            # Push to Nexus - always push, even if no skills
            success, error_response = push_to_nexus(
                str(data.project_uuid), data.definition, skill_mapping, request_id, authorization
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
                    "total_files": skill_count,
                    "processed_files": len(skill_mapping),
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
async def extract_skill_files(form: Any) -> dict[str, UploadFile]:
    """
    Extract skill files from form data.

    Args:
        form: The form data containing files (FormData from FastAPI)

    Returns:
        Dictionary of agent:skill keys to UploadFile objects
    """
    skills_folders_zips = {}
    for key, value in form.items():
        if isinstance(value, UploadFile) and ":" in key:
            skills_folders_zips[key] = value

    return skills_folders_zips


async def read_skills_content(skills_folders_zips: dict[str, UploadFile]) -> list[tuple[str, bytes]]:
    """
    Read the content of each skill file.

    Args:
        skills_folders_zips: Dictionary of skill keys to file objects

    Returns:
        List of tuples containing (key, file_content)
    """
    return [(key, await file.read()) for key, file in skills_folders_zips.items()]


def push_to_nexus(
    project_uuid: str,
    definition: dict[str, Any],
    skill_mapping: dict[str, Any],
    request_id: str,
    authorization: str,
) -> tuple[bool, CLIResponse | None]:
    """
    Push processed skills to Nexus.

    Args:
        project_uuid: The UUID of the project
        definition: The agent definition
        skill_mapping: Dictionary of processed skills
        request_id: The request ID for correlation
        authorization: The authorization header value

    Returns:
        Tuple of (success, response)
    """
    try:
        logger.info(f"Sending {len(skill_mapping)} processed skills to Nexus for project {project_uuid}")
        nexus_client = NexusClient(authorization, str(project_uuid))

        # We need to change the entrypoint to the lambda function we've created
        for _, agent_data in definition["agents"].items():
            for skill in agent_data["skills"]:
                skill["source"]["entrypoint"] = "lambda_function.lambda_handler"

        response = nexus_client.push_agents(definition, skill_mapping)

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
                "skills_processed": len(skill_mapping),
            },
            "success": False,
            "code": "NEXUS_UPLOAD_ERROR",
            "progress": 0.9,
        }

        return False, nexus_error
