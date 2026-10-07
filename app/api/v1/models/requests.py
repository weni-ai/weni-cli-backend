from datetime import datetime
from typing import Any, Literal, Self

from pydantic import UUID4, BaseModel, Field, Json, model_validator


class BaseRequestModel(BaseModel):
    """Base request model."""

    project_uuid: UUID4
    definition: Json
    toolkit_version: str


class ConfigureAgentsRequestModel(BaseRequestModel):
    """Configure agents request model."""

    # type can only be "active" or "passive"
    type: Literal["active", "passive"]
    apm_instrumentation: Literal["enabled", "disabled"] | None = None


class RunRequestModel(BaseRequestModel):
    """Run request model — supports both passive (tool) and active (preprocessor + rules) agents."""

    test_definition: Json
    agent_key: str
    type: Literal["active", "passive"] = "passive"
    tool_key: str | None = None
    tool_credentials: Json | None = None
    tool_globals: Json | None = None


# Backward-compatible alias for tests/imports referencing the old name
RunToolRequestModel = RunRequestModel


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


class CreateChannelRequestModel(BaseModel):
    """Create channel request model."""

    project_uuid: UUID4
    channel_definition: dict[str, Any]


class UpdateChannelRequestModel(BaseModel):
    """Partial channel update. Fields left out stay as they are in Flows."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    address: str | None = Field(default=None, max_length=255)
    config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_writable_field(self) -> Self:
        if self.name is None and self.address is None and self.config is None:
            message = "At least one of name, address or config is required."
            raise ValueError(message)
        return self


class CreateTicketerRequestModel(BaseModel):
    """Create ticketer request model."""

    project_uuid: UUID4
    ticketer_definition: dict[str, Any]


class RunEvaluationRequestModel(BaseModel):
    """Run evaluation request model."""

    evaluator: dict[str, Any] = {}
    target: dict[str, Any] = {}
    tests: dict[str, Any]
    filter: str | None = None
