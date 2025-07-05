from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import UUID4, BaseModel, Json

from app.core.config import settings


class BaseRequestModel(BaseModel):
    """Base request model."""

    project_uuid: UUID4
    definition: Json
    toolkit_version: str


class ConfigureAgentsRequestModel(BaseRequestModel):
    """Configure agents request model."""

    # type can be "active" or "passive", but is optional since we can auto-detect
    type: Literal["active", "passive"] | None = None


class RunToolRequestModel(BaseRequestModel):
    """Run tool request model for passive agents."""

    test_definition: Json
    tool_key: str
    agent_key: str
    tool_credentials: Json
    tool_globals: Json


class RunActiveAgentRequestModel(BaseRequestModel):
    """Run agent request model for active agents."""

    test_definition: Json
    agent_key: str
    rule_key: str
    rule_credentials: Json = {}  # Used as default credentials
    rule_globals: Json = {}
    type: str = "active"  # CLI sends this field
    payload_path: str | None = None  # Path to webhook payload JSON file
    webhook_data: Json | None = None  # Webhook data sent directly by CLI


class VerifyPermissionRequestModel(BaseModel):
    """Verify permission request model."""

    project_uuid: UUID4


class GetLogsRequestModel(BaseModel):
    """Get logs request model."""

    tool_key: str
    agent_key: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    pattern: str | None = None
    next_token: str | None = None
