import logging
from typing import Annotated, cast
from uuid import uuid4

from fastapi import Form, Header, Request, status

from app.api.v1.models.requests import (
    ConfigureAgentsRequestModel,
    CreateChannelRequestModel,
    CreateTicketerRequestModel,
    RunRequestModel,
)
from app.api.v1.rejections import RequestRejectedError
from app.api.v1.user_identity import read_user_email
from app.core.log_events import format_log_event

logger = logging.getLogger(__name__)

PROJECT_MISMATCH_MESSAGE = "The project in the request does not match the authorized project."
PROJECT_MISMATCH_CODE = "PROJECT_MISMATCH"
PROJECT_MISMATCH_EVENT = "project_mismatch_rejected"


class ProjectMismatchError(RequestRejectedError):
    def __init__(self, request_id: str) -> None:
        super().__init__(
            http_status=status.HTTP_403_FORBIDDEN,
            code=PROJECT_MISMATCH_CODE,
            message=PROJECT_MISMATCH_MESSAGE,
            request_id=request_id,
        )


def ensure_body_project_is_authorized(
    request: Request,
    authorized_project_uuid: str,
    requested_project_uuid_raw: str,
) -> None:
    # UUID4 validation normalizes letter case, and the clarification requires exact string comparison.
    if authorized_project_uuid == requested_project_uuid_raw:
        return
    request_id = str(uuid4())
    logger.warning(
        format_log_event(
            PROJECT_MISMATCH_EVENT,
            {
                "header_project_uuid": authorized_project_uuid,
                "body_project_uuid": requested_project_uuid_raw,
                "endpoint": request.url.path,
                "request_id": request_id,
                "user_email": read_user_email(request.headers.get("Authorization")),
            },
        )
    )
    raise ProjectMismatchError(request_id)


# FastAPI validates the body from each dependency's parameter annotation, which is what
# keeps an invalid body at 422. One shared function would drop that model. Runs and agents
# re-read the multipart field, and channels and ticketers re-read the JSON field, because
# data.project_uuid is already normalized.
async def bound_run_request(
    request: Request,
    data: Annotated[RunRequestModel, Form()],
    x_project_uuid: Annotated[str, Header()],
) -> RunRequestModel:
    raw_project_uuid = cast(str, (await request.form())["project_uuid"])
    ensure_body_project_is_authorized(request, x_project_uuid, raw_project_uuid)
    return data


async def bound_agents_request(
    request: Request,
    data: Annotated[ConfigureAgentsRequestModel, Form()],
    x_project_uuid: Annotated[str, Header()],
) -> ConfigureAgentsRequestModel:
    raw_project_uuid = cast(str, (await request.form())["project_uuid"])
    ensure_body_project_is_authorized(request, x_project_uuid, raw_project_uuid)
    return data


async def bound_channel_request(
    request: Request,
    data: CreateChannelRequestModel,
    x_project_uuid: Annotated[str, Header()],
) -> CreateChannelRequestModel:
    raw_project_uuid = cast(str, (await request.json())["project_uuid"])
    ensure_body_project_is_authorized(request, x_project_uuid, raw_project_uuid)
    return data


async def bound_ticketer_request(
    request: Request,
    data: CreateTicketerRequestModel,
    x_project_uuid: Annotated[str, Header()],
) -> CreateTicketerRequestModel:
    raw_project_uuid = cast(str, (await request.json())["project_uuid"])
    ensure_body_project_is_authorized(request, x_project_uuid, raw_project_uuid)
    return data
