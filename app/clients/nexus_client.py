import json

import requests
from requests import Response

from app.core.config import settings


class NexusClient:
    base_url: str = ""
    headers: dict[str, str] = {}

    def __init__(self, user_auth_token: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.NEXUS_BASE_URL

    def push_agents(self, project_uuid: str, agents_definition: dict, tool_files: dict) -> Response:
        url = f"{self.base_url}/api/agents/push"

        data = {
            "project_uuid": project_uuid,
            "agents": json.dumps(agents_definition),
        }

        return requests.post(url, headers=self.headers, data=data, files=tool_files)
