"""
Agents endpoints for handling file uploads and processing.
"""

import json
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


def detect_agent_type(definition: dict[str, Any]) -> str:
    """
    Automatically detect if agents are active or passive based on definition structure.
    
    Args:
        definition: The agent definition dictionary
        
    Returns:
        str: "active" if agents have rules, "passive" if they have tools
        
    Raises:
        ValueError: If unable to determine agent type
    """
    agents = definition.get("agents", {})
    
    if not agents:
        raise ValueError("No agents found in definition")
    
    # Check the first agent to determine type
    for agent_key, agent_data in agents.items():
        # Active agents have "rules" field
        if "rules" in agent_data:
            logger.info(f"Detected active agent type for agent '{agent_key}' (has 'rules' field)")
            return "active"
        # Passive agents have "tools" field  
        elif "tools" in agent_data:
            logger.info(f"Detected passive agent type for agent '{agent_key}' (has 'tools' field)")
            return "passive"
    
    raise ValueError("Unable to determine agent type: no 'rules' or 'tools' found in agent definition")


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

    # Parse the definition to detect agent type automatically
    try:
        definition_dict = json.loads(data.definition) if isinstance(data.definition, str) else data.definition
        detected_type = detect_agent_type(definition_dict)
        
        # Use detected type, with optional validation against provided type
        if data.type is not None and data.type != detected_type:
            logger.warning(
                f"Type mismatch: provided type '{data.type}' differs from detected type '{detected_type}'. "
                f"Using detected type '{detected_type}'"
            )
        elif data.type is None:
            logger.info(f"No type provided, using auto-detected type: '{detected_type}'")
        
        agent_type = detected_type
        
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to provided type if detection fails
        if data.type is not None:
            logger.warning(f"Failed to detect agent type automatically: {e}. Using provided type: {data.type}")
            agent_type = data.type
        else:
            logger.error(f"Failed to detect agent type automatically and no type provided: {e}")
            raise ValueError(f"Unable to determine agent type: {e}")
    
    logger.info(f"Processing {agent_type} agent for project {data.project_uuid}")

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

    if configurator_cls := agent_configurators.get(agent_type):
        configurator_instance = configurator_cls(
            str(data.project_uuid),
            definition_dict if 'definition_dict' in locals() else json.loads(data.definition),
            data.toolkit_version,
            request_id,
            authorization
        )

        return configurator_instance.configure_agents(agent_resources_folders_zips_entries)

    raise ValueError(f"Invalid agent type: {agent_type}")

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
