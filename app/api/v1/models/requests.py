from pydantic import UUID4, BaseModel, Json


class BaseRequestModel(BaseModel):
    """Base request model."""

    project_uuid: UUID4
    definition: Json
    toolkit_version: str


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
