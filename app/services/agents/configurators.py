import logging
from dataclasses import dataclass
from typing import Any

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentConfiguratorContext:
    project_uuid: str
    definition: dict[str, Any]
    toolkit_version: str
    request_id: str
    authorization: str
    apm_instrumentation: str | None = None


class AgentConfigurator:
    def __init__(self, context: AgentConfiguratorContext):
        self.project_uuid = context.project_uuid
        self.definition = context.definition
        self.toolkit_version = context.toolkit_version
        self.request_id = context.request_id
        self.authorization = context.authorization
        self.apm_instrumentation = context.apm_instrumentation

    def configure_agents(self, agent_resources_entries: list[tuple[str, bytes]]) -> StreamingResponse:
        raise NotImplementedError("This method should be implemented by the subclass")
