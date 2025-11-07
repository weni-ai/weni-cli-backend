"""
Channels endpoints for creating channels in Flows.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from app.api.v1.models.requests import CreateChannelRequestModel
from app.clients.flows_client import FlowsClient

router = APIRouter()
logger = logging.getLogger(__name__)

# HTTP status code constants
HTTP_BAD_REQUEST = 400


@router.post("")
async def create_channel(
    data: CreateChannelRequestModel,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """
    Create a new channel in Flows.

    Args:
        data: Channel creation data including project_uuid and channel_definition
        authorization: Authorization token from weni login
        x_project_uuid: Project UUID from header

    Returns:
        JSONResponse: Response from Flows API
    """
    logger.info(f"Creating channel for project {data.project_uuid}")
    logger.debug(f"Channel definition: {data.channel_definition}")

    try:
        # Create Flows client instance
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=str(data.project_uuid))

        # Call Flows API to create channel
        response = flows_client.create_channel(data.channel_definition)

        # Check response status
        if response.status_code >= HTTP_BAD_REQUEST:
            logger.error(f"Error creating channel: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Failed to create channel: {response.text}")

        logger.info(f"Channel created successfully for project {data.project_uuid}")
        return JSONResponse(status_code=response.status_code, content=response.json())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error creating channel: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e
