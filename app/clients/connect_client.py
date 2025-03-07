import requests
from requests import Response

from app.core.config import settings


class ConnectClient:
    base_url: str = ""
    headers: dict[str, str] = {}

    def __init__(self, user_auth_token: str, project_uuid: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.WENI_API_URL
        self.project_uuid = project_uuid

    def check_authorization(self) -> Response:
        url = f"{self.base_url}/v2/projects/{self.project_uuid}/authorization"

        return requests.get(url, headers=self.headers)
