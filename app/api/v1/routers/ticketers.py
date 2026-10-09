"""
Ticketers endpoints for creating ticketers in Flows.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from requests import Response as RequestsResponse

from app.api.v1.models.requests import CreateTicketerRequestModel
from app.api.v1.project_binding import bound_ticketer_request
from app.clients.flows_client import FlowsClient

router = APIRouter()
logger = logging.getLogger(__name__)

# HTTP status code constants
HTTP_BAD_REQUEST = 400
HTTP_NO_CONTENT = 204


def _raise_if_flows_error(response: RequestsResponse, action: str) -> None:
    if response.status_code >= HTTP_BAD_REQUEST:
        logger.error(f"Error {action} ticketer: {response.status_code} - {response.text}")
        raise HTTPException(status_code=response.status_code, detail=f"Failed to {action} ticketer: {response.text}")


@router.post("")
async def create_ticketer(
    data: Annotated[CreateTicketerRequestModel, Depends(bound_ticketer_request)],
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """
    Create a new ticketer in Flows.

    Args:
        data: Ticketer creation data including project_uuid and ticketer_definition
        authorization: Authorization token from weni login
        x_project_uuid: Project UUID from header

    Returns:
        JSONResponse: Response from Flows API
    """
    logger.info(f"Creating ticketer for project {data.project_uuid}")
    logger.debug(f"Ticketer definition: {data.ticketer_definition}")

    try:
        # Create Flows client instance
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=str(data.project_uuid))

        # Call Flows API to create ticketer
        response = flows_client.create_ticketer(data.ticketer_definition)

        _raise_if_flows_error(response, "create")

        logger.info(f"Ticketer created successfully for project {data.project_uuid}")
        return JSONResponse(status_code=response.status_code, content=response.json())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error creating ticketer: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.get("")
async def list_ticketers(
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """List ticketers for the project."""
    logger.info(f"Listing ticketers for project {x_project_uuid}")

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        response = flows_client.list_ticketers()
        _raise_if_flows_error(response, "list")
        return JSONResponse(status_code=response.status_code, content=response.json())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error listing ticketers: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.get("/{ticketer_uuid}")
async def get_ticketer(
    ticketer_uuid: str,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """Get a ticketer by UUID."""
    logger.info(f"Getting ticketer {ticketer_uuid} for project {x_project_uuid}")

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        response = flows_client.get_ticketer(str(ticketer_uuid))
        _raise_if_flows_error(response, "get")
        return JSONResponse(status_code=response.status_code, content=response.json())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error getting ticketer: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.put("/{ticketer_uuid}")
async def update_ticketer(
    ticketer_uuid: str,
    data: Annotated[CreateTicketerRequestModel, Depends(bound_ticketer_request)],
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> JSONResponse:
    """Update a ticketer."""
    logger.info(f"Updating ticketer {ticketer_uuid} for project {data.project_uuid}")

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=str(data.project_uuid))
        response = flows_client.update_ticketer(str(ticketer_uuid), data.ticketer_definition)
        _raise_if_flows_error(response, "update")
        return JSONResponse(status_code=response.status_code, content=response.json())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error updating ticketer: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.delete("/{ticketer_uuid}", response_model=None)
async def delete_ticketer(
    ticketer_uuid: str,
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
) -> Response:
    """Delete a ticketer."""
    logger.info(f"Deleting ticketer {ticketer_uuid} for project {x_project_uuid}")

    try:
        flows_client = FlowsClient(user_auth_token=authorization, project_uuid=x_project_uuid)
        response = flows_client.delete_ticketer(str(ticketer_uuid))
        _raise_if_flows_error(response, "delete")
        if response.status_code == HTTP_NO_CONTENT:
            return Response(status_code=HTTP_NO_CONTENT)
        return JSONResponse(status_code=response.status_code, content=response.json())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error deleting ticketer: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e
