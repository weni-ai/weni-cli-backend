import json

import requests
from requests import Response

from app.core.config import settings


class FlowsClient:
    base_url: str = ""
    headers: dict[str, str] = {}
    project_uuid: str = ""

    def __init__(self, user_auth_token: str, project_uuid: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.FLOWS_BASE_URL
        self.project_uuid = project_uuid

    def create_channel(self, channel_definition: dict) -> Response:
        url = f"{self.base_url}/api/v2/internals/channels/create"

        data = {
            "project_uuid": self.project_uuid,
            "channel_data": json.dumps(channel_definition),
        }

        return requests.post(url, headers=self.headers, json=data)


