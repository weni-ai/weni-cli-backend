"""
Tool Logs endpoints
"""

import json
from typing import Annotated

from fastapi import APIRouter, Header, Query, status
from fastapi.responses import JSONResponse

from app.api.v1.models.requests import GetLogsRequestModel
from app.clients.aws.logs_client import AWSLogsClient
from app.clients.nexus_client import NexusClient

router = APIRouter()


@router.get("/")
async def get_logs(
    # Header parameters
    authorization: Annotated[str, Header()],
    x_project_uuid: Annotated[str, Header()],
    # Query parameters
    data: Annotated[GetLogsRequestModel, Query()],
) -> JSONResponse:
    nexus_client = NexusClient(authorization, x_project_uuid)
    log_group = nexus_client.get_log_group(data.agent_key, data.tool_key)

    if log_group.status_code != status.HTTP_200_OK:
        return JSONResponse(
            status_code=log_group.status_code,
            content={"status": "error", "message": f"Error getting log group: {log_group.text}"},
        )

    log_group_data = log_group.json()

    try:
        log_group_arn = log_group_data["log_group"]["log_group_arn"]
    except KeyError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "error", "message": "Log group not found in Nexus response"},
        )

    logs_client = AWSLogsClient()
    logs, next_token = await logs_client.get_function_logs(
        log_group_arn, data.start_time, data.end_time, data.pattern, data.next_token
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "logs": json.loads(json.dumps(logs, default=str)),
            "next_token": next_token,
        },
    )
