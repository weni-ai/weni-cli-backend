"""
Channels endpoints that proxy the Flows internal channel API.
"""

import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse
from requests import Response

from app.api.v1.models.requests import CreateChannelRequestModel, UpdateChannelRequestModel
from app.api.v1.project_binding import bound_channel_request
from app.clients.flows_client import FlowsClient

router = APIRouter()
logger = logging.getLogger(__name__)

# HTTP status code constants
HTTP_BAD_REQUEST = 400


def _proxy_flows_response(response: Response, action: str) -> JSONResponse:
    if response.status_code >= HTTP_BAD_REQUEST:
        logger.error("Error %s channel: %s - %s", action, response.status_code, response.text)
        raise HTTPException(status_code=response.status_code, detail=f"Failed to {action} channel: {response.text}")

    if not response.content:
        return JSONResponse(status_code=response.status_code, content={})

    return JSONResponse(status_code=response.status_code, content=response.json())


def _load_project_channel(flows_client: FlowsClient, channel_uuid: str, project_uuid: str) -> dict[str, Any]:
    """Read a channel and keep it inside the project authorized by X-Project-Uuid."""
    response = flows_client.get_channel(channel_uuid)
    if response.status_code >= HTTP_BAD_REQUEST:
        logger.error("Error reading channel %s: %s - %s", channel_uuid, response.status_code, response.text)
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to read channel: {response.text}",
        )

    content: Any = response.json()
    org = content.get("org") if isinstance(content, dict) else None
    if not isinstance(content, dict) or str(org) != project_uuid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found")

    return content


@router.get("")
async def list_channels(
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
    channel_type: Annotated[str | None, Query()] = None,
    exclude_wpp_demo: Annotated[bool, Query()] = False,
) -> JSONResponse:
    """
    List channels of the authorized project.

    Flows filters with the ``org`` query param, which this route always sets
    from ``X-Project-Uuid``.
    """
    logger.info("Listing channels for project %s", x_project_uuid)

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        response = flows_client.list_channels(channel_type=channel_type, exclude_wpp_demo=exclude_wpp_demo)
        return _proxy_flows_response(response, "list")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error listing channels: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e


@router.get("/{channel_uuid}")
async def get_channel(
    channel_uuid: UUID,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """Return one channel when it belongs to the authorized project."""
    logger.info("Reading channel %s for project %s", channel_uuid, x_project_uuid)

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        channel = _load_project_channel(flows_client, str(channel_uuid), x_project_uuid)
        return JSONResponse(status_code=status.HTTP_200_OK, content=channel)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error reading channel: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e


@router.patch("/{channel_uuid}")
async def update_channel(
    channel_uuid: UUID,
    data: UpdateChannelRequestModel,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """
    Partially update a channel.

    Flows merges ``config`` on PATCH. ``uuid``, org, type and ``is_active`` stay unchanged.
    """
    logger.info("Updating channel %s for project %s", channel_uuid, x_project_uuid)

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        _load_project_channel(flows_client, str(channel_uuid), x_project_uuid)
        response = flows_client.update_channel(
            str(channel_uuid),
            name=data.name,
            address=data.address,
            config=data.config,
        )
        return _proxy_flows_response(response, "update")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error updating channel: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e


@router.delete("/{channel_uuid}")
async def delete_channel(
    channel_uuid: UUID,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """
    Soft-delete a channel.

    Flows releases the channel: ``is_active`` becomes false and a DELETE event is published.
    """
    logger.info("Releasing channel %s for project %s", channel_uuid, x_project_uuid)

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        _load_project_channel(flows_client, str(channel_uuid), x_project_uuid)
        response = flows_client.release_channel(str(channel_uuid))
        return _proxy_flows_response(response, "release")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error releasing channel: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e


@router.post("")
async def create_channel(
    data: Annotated[CreateChannelRequestModel, Depends(bound_channel_request)],
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
