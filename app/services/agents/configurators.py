import logging
from typing import Any

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

class AgentConfigurator:
    def __init__(
        self,
        project_uuid: str,
        definition: dict[str, Any],
        toolkit_version: str,
        request_id: str,
        authorization: str,
    ):
        self.project_uuid = project_uuid
        self.definition = definition
        self.toolkit_version = toolkit_version
        self.request_id = request_id
        self.authorization = authorization

    def configure_agents(self, agent_resources_entries: list[tuple[str, bytes]]) -> StreamingResponse:
        raise NotImplementedError("This method should be implemented by the subclass")
