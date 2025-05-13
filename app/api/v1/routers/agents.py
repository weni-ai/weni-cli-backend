"""
Agents endpoints for handling file uploads and processing.
"""

import logging
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Form, Header, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from app.api.v1.models.requests import ConfigureAgentsRequestModel
from app.services.agents.active.configurator import ActiveAgentConfigurator
from app.services.agents.passive.configurator import PassiveAgentConfigurator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def configure_agents(
    request: Request,
    data: Annotated[ConfigureAgentsRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    """
    Upload multiple files for tool processing.

    Args:
        request: FastAPI request object to access form data and files
        project_uuid: UUID of the project
        definition: JSON containing tool definitions
        authorization: Authorization header for Nexus API calls
        type: Type of agents to configure (active or passive)

    Returns:
        StreamingResponse: Streaming response for future result handling
    """
    request_id = str(uuid4())
    logger.info(f"Processing agent configuration for project {data.project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {data.definition}")

    # Access the form data with files
    form = await request.form()

    # Extract and process agent resources files
    agent_resources_folders_zips = await extract_agent_resources_files(form)
    resource_count = len(agent_resources_folders_zips)
    logger.info(f"Found {resource_count} resource to process for project {data.project_uuid}")
    logger.debug(f"Resource keys: {list(agent_resources_folders_zips.keys())}")

    # Create file names list for streaming
    agent_resources_folders_zips_entries = await read_agent_resources_content(agent_resources_folders_zips)

    agent_configurators = {
        "active": ActiveAgentConfigurator,
        "passive": PassiveAgentConfigurator,
    }

    if configurator_cls := agent_configurators.get(data.type):
        configurator_instance = configurator_cls(
            str(data.project_uuid),
            data.definition,
            data.toolkit_version,
            request_id,
            authorization
        )

        return configurator_instance.configure_agents(agent_resources_folders_zips_entries)

    raise ValueError(f"Invalid agent type: {data.type}")

async def extract_agent_resources_files(form: Any) -> dict[str, UploadFile]:
    """
    Extract agent resources files from form data.

    Args:
        form: The form data containing files (FormData from FastAPI)

    Returns:
        Dictionary of agent:tool keys to UploadFile objects
    """
    agent_resources_folders_zips = {}
    for key, value in form.items():
        if isinstance(value, UploadFile) and ":" in key:
            agent_resources_folders_zips[key] = value

    return agent_resources_folders_zips

async def read_agent_resources_content(
    agent_resources_folders_zips: dict[str, UploadFile]
) -> list[tuple[str, bytes]]:
    """
    Read the content of each agent resources file.

    Args:
        agent_resources_folders_zips: Dictionary of resource keys to file objects

    Returns:
        List of tuples containing (key, file_content)
    """
    return [(key, await file.read()) for key, file in agent_resources_folders_zips.items()]
