"""
Skill endpoints for handling file uploads and processing.
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Form, Header, Request, status
from fastapi.responses import StreamingResponse
from pydantic import UUID4, Json
from starlette.datastructures import UploadFile

from app.clients.nexus_client import NexusClient
from app.core.response import CLIResponse, send_response
from app.services.skill.packager import create_skill_zip

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def configure_agents(
    request: Request,
    project_uuid: Annotated[UUID4, Form()],
    definition: Annotated[Json, Form()],
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
    logger.info(f"Processing agent configuration for project {project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {definition}")

    # Access the form data with files
    form = await request.form()

    # Extract and process skill files
    skills_folders_zips = await extract_skill_files(form)
    skill_count = len(skills_folders_zips)
    logger.info(f"Found {skill_count} skill folders to process for project {project_uuid}")
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
                    "project_uuid": str(project_uuid),
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
                    str(project_uuid),
                    agent_slug,
                    skill_slug,
                    definition,
                    processed_count,
                    skill_count,
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
                "message": "Sending agents to Nexus...",
                "data": {
                    "project_uuid": str(project_uuid),
                    "skill_count": len(skill_mapping),
                },
                "success": True,
                "code": "NEXUS_UPLOAD_STARTED",
                "progress": 0.9,
            }
            yield send_response(nexus_response, request_id=request_id)

            # Push to Nexus - always push, even if no skills
            success, error_response = push_to_nexus(
                str(project_uuid), definition, skill_mapping, request_id, authorization
            )

            if not success and error_response is not None:
                # Type cast error_response to CLIResponse for type checker
                typed_error_response = cast(CLIResponse, error_response)
                yield send_response(typed_error_response, request_id=request_id)

                response_data = typed_error_response.get("data") or {}
                error_message = str(response_data.get("error", "Unknown error pushing to Nexus"))
                raise Exception(error_message)

            # Final message
            final_data: CLIResponse = {
                "message": "Agents processed successfully",
                "data": {
                    "project_uuid": str(project_uuid),
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
            logger.error(f"Error processing agents for project {project_uuid}: {str(e)}", exc_info=True)
            error_data: CLIResponse = {
                "message": "Error processing agents",
                "data": {
                    "project_uuid": str(project_uuid),
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


async def process_skill(  # noqa: PLR0913
    folder_zip: bytes,
    key: str,
    project_uuid: str,
    agent_slug: str,
    skill_slug: str,
    definition: dict[str, Any],
    processed_count: int,
    total_count: int,
) -> tuple[CLIResponse, Any | None]:
    """
    Process a single skill file.

    Args:
        folder_zip: The skill folder zip file content
        key: The skill key (agent:skill)
        project_uuid: The UUID of the project
        agent_slug: The slug of the agent
        skill_slug: The name of the skill
        definition: The definition data
        processed_count: The number of skills processed so far
        total_count: The total number of skills to process
        request_id: The request ID for correlation

    Returns:
        Tuple of (response, skill_zip_bytes or None if error)
    """
    # Reduce the progress to be always between 0.2 and 0.9
    progress = 0.2 + (processed_count / total_count) * 0.7
    logger.info(f"Processing skill {skill_slug} for agent {agent_slug} ({processed_count}/{total_count})")

    try:
        # Get skill entrypoint
        agent_info = next((agent for agent in definition["agents"].values() if agent["slug"] == agent_slug), None)

        if not agent_info:
            raise ValueError(f"Could not find agent {agent_slug} in definition")

        skill_info = next(
            (skill for skill in agent_info["skills"] if skill["slug"] == skill_slug),
            None,
        )

        if not skill_info:
            raise ValueError(f"Could not find skill {skill_slug} for agent {agent_slug} in definition")

        skill_entrypoint = skill_info["source"]["entrypoint"]
        skill_entrypoint_module = skill_entrypoint.split(".")[0]
        skill_entrypoint_class = skill_entrypoint.split(".")[1]
        logger.debug(f"Skill entrypoint: {skill_entrypoint_module}.{skill_entrypoint_class}")

        # Create zip package
        logger.info(f"Creating zip package for skill {skill_slug}")
        skill_zip_bytes = create_skill_zip(
            folder_zip, key, str(project_uuid), skill_entrypoint_module, skill_entrypoint_class
        )

        file_size_kb = len(skill_zip_bytes.getvalue()) / 1024
        logger.debug(f"Successfully created zip for {key}, size: {file_size_kb:.2f} KB")

        # Prepare success response
        response: CLIResponse = {
            "message": f"Skill {skill_slug} processed successfully ({processed_count}/{total_count})",
            "data": {
                "skill_name": skill_slug,
                "agent_name": agent_slug,
                "size_kb": round(file_size_kb, 2),
                "processed": processed_count,
                "total": total_count,
            },
            "success": True,
            "code": "SKILL_PROCESSED",
            "progress": progress,
        }

        return response, skill_zip_bytes

    except Exception as e:
        logger.error(f"Failed to process skill {key}: {str(e)}")
        error_response: CLIResponse = {
            "message": f"Failed to process skill {skill_slug}",
            "data": {
                "skill_name": skill_slug,
                "agent_name": agent_slug,
                "error": str(e),
                "processed": processed_count,
                "total": total_count,
            },
            "success": False,
            "code": "SKILL_PROCESSING_ERROR",
            "progress": progress,
        }
        return error_response, None


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
        nexus_client = NexusClient(authorization)

        # We need to change the entrypoint to the lambda function we've created
        for _, agent_data in definition["agents"].items():
            for skill in agent_data["skills"]:
                skill["source"]["entrypoint"] = "lambda_function.lambda_handler"

        response = nexus_client.push_agents(str(project_uuid), definition, skill_mapping)

        if response.status_code != status.HTTP_200_OK:
            raise Exception(f"Failed to push agents to Nexus: {response.status_code} {response.text}")

        logger.info(f"Successfully pushed agents to Nexus for project {project_uuid}")

        return True, None

    except Exception as e:
        logger.error(f"Failed to push agents to Nexus: {str(e)}", exc_info=True)
        nexus_error: CLIResponse = {
            "message": "Failed to push agents to Nexus",
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
