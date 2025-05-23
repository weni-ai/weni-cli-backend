from datetime import datetime
from typing import Literal

from pydantic import UUID4, BaseModel, Json


class BaseRequestModel(BaseModel):
    """Base request model."""

    project_uuid: UUID4
    definition: Json
    toolkit_version: str


class ConfigureAgentsRequestModel(BaseRequestModel):
    """Configure agents request model."""

    # type can only be "active" or "passive"
    type: Literal["active", "passive"]


class RunToolRequestModel(BaseRequestModel):
    """Run tool request model."""

    test_definition: Json
    tool_key: str
    agent_key: str
    tool_credentials: Json
    tool_globals: Json


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
