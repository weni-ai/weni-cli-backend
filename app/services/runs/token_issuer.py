import logging
from dataclasses import dataclass
from typing import Literal

from app.core.config import settings
from app.core.log_events import format_log_event
from app.services.jwt_generator import generate_jwt_token

logger = logging.getLogger(__name__)

RUN_TOKEN_MINTED_EVENT = "run_token_minted"
RunType = Literal["passive", "active"]


@dataclass(kw_only=True)
class RunTokenIssuer:
    authorized_project_uuid: str
    user_email: str
    agent_key: str
    tool_key: str | None
    run_type: RunType
    request_id: str

    def issue(self) -> str:
        token = generate_jwt_token(self.authorized_project_uuid, settings.JWT_SECRET_KEY)
        logger.info(
            format_log_event(
                RUN_TOKEN_MINTED_EVENT,
                {
                    "user_email": self.user_email,
                    "project_uuid": self.authorized_project_uuid,
                    "agent_key": self.agent_key,
                    "tool_key": self.tool_key,
                    "run_type": self.run_type,
                    "request_id": self.request_id,
                },
            )
        )
        return token
