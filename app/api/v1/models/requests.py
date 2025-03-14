from pydantic import UUID4, BaseModel, Json


class BaseRequestModel(BaseModel):
    """Base request model."""

    project_uuid: UUID4
    definition: Json
    toolkit_version: str


class RunSkillRequestModel(BaseRequestModel):
    """Run skill request model."""

    test_definition: Json
    skill_name: str
    agent_name: str
    skill_credentials: Json
    skill_globals: Json


class VerifyPermissionRequestModel(BaseModel):
    """Verify permission request model."""

    project_uuid: UUID4
