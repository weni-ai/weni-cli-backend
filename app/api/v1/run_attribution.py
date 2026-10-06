import logging
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, Header, Request, status

from app.api.v1.models.requests import RunRequestModel
from app.api.v1.project_binding import bound_run_request
from app.api.v1.rejections import RequestRejectedError
from app.api.v1.user_identity import read_user_email
from app.core.log_events import format_log_event

logger = logging.getLogger(__name__)

RUN_NOT_ATTRIBUTABLE_MESSAGE = "This run could not be attributed to a user."
RUN_NOT_ATTRIBUTABLE_CODE = "RUN_NOT_ATTRIBUTABLE"
RUN_NOT_ATTRIBUTABLE_EVENT = "run_not_attributable"


@dataclass
class AttributedRun:
    request: RunRequestModel
    authorized_project_uuid: str
    user_email: str


class RunNotAttributableError(RequestRejectedError):
    def __init__(self, request_id: str) -> None:
        super().__init__(
            http_status=status.HTTP_403_FORBIDDEN,
            code=RUN_NOT_ATTRIBUTABLE_CODE,
            message=RUN_NOT_ATTRIBUTABLE_MESSAGE,
            request_id=request_id,
        )


async def attributed_run_request(
    request: Request,
    data: Annotated[RunRequestModel, Depends(bound_run_request)],
    x_project_uuid: Annotated[str, Header()],
) -> AttributedRun:
    email = read_user_email(request.headers.get("Authorization"))
    if email is None:
        request_id = str(uuid4())
        logger.warning(
            format_log_event(
                RUN_NOT_ATTRIBUTABLE_EVENT,
                {
                    "header_project_uuid": x_project_uuid,
                    "endpoint": request.url.path,
                    "request_id": request_id,
                },
            )
        )
        raise RunNotAttributableError(request_id)
    return AttributedRun(
        request=data,
        authorized_project_uuid=str(UUID(x_project_uuid)),
        user_email=email,
    )
