import json
import logging

import requests
from requests import Response

from app.core.config import settings

logger = logging.getLogger(__name__)


class NexusClient:
    base_url: str = ""
    headers: dict[str, str] = {}
    project_uuid: str

    def __init__(self, user_auth_token: str, project_uuid: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.NEXUS_BASE_URL
        self.project_uuid = project_uuid

    def push_agents(
        self,
        agents_definition: dict,
        tool_files: dict,
        apm_instrumentation: str | None = None,
    ) -> Response:
        url = f"{self.base_url}/api/agents/push"

        data = {
            "project_uuid": self.project_uuid,
            "agents": json.dumps(agents_definition),
        }

        if apm_instrumentation is not None:
            data["apm_instrumentation"] = apm_instrumentation

        logger.info(
            "Pushing agents to Nexus for project %s (apm_instrumentation=%s)",
            self.project_uuid,
            apm_instrumentation,
        )
        return requests.post(url, headers=self.headers, data=data, files=tool_files)

    def get_log_group(self, agent_key: str, tool_key: str) -> Response:
        url = f"{self.base_url}/api/agents/log-group"

        params = {
            "project": self.project_uuid,
            "agent_key": agent_key,
            "tool_key": tool_key,
        }

        return requests.get(url, headers=self.headers, params=params)
