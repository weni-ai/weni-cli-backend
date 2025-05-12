import json

import requests
from requests import Response

from app.core.config import settings


class GalleryClient:
    base_url: str = ""
    headers: dict[str, str] = {}
    project_uuid: str = ""
    def __init__(self, project_uuid: str, user_auth_token: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.GALLERY_BASE_URL
        self.project_uuid = project_uuid

    def push_agents(self, agents_definition: dict, rules_files: dict) -> Response:
        url = f"{self.base_url}/api/v3/agents"

        data = {
            "project_uuid": self.project_uuid,
            "agents": json.dumps(agents_definition),
        }

        return requests.post(url, headers=self.headers, data=data, files=rules_files)
