import logging

import requests
from requests import Response

from app.core.config import settings

# HTTP status code constant
HTTP_OK = 200

logger = logging.getLogger(__name__)


class FlowsClient:
    base_url: str = ""
    headers: dict[str, str] = {}
    project_uuid: str = ""
    user_email: str = ""

    def __init__(self, user_auth_token: str, project_uuid: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.FLOWS_BASE_URL
        self.project_uuid = project_uuid
        self.user_email = self._get_user_email()

    def _get_user_email(self) -> str:
        """Get user email from Connect API using the auth token."""
        try:
            url = f"{settings.WENI_API_URL}/v2/account/my-profile/"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == HTTP_OK:
                return response.json().get("email", "")
            return ""
        except Exception:
            return ""

    def create_channel(self, channel_definition: dict) -> Response:
        url = f"{self.base_url}/api/v2/internals/channels/create/"

        # Extract fields from channel_definition
        channel_type = channel_definition.get("channel_type", "")
        config = channel_definition.get("config", {})

        # Build the payload according to Flows API format
        # Format: {"user": "email", "org": "uuid", "channeltype_code": "E2", "data": {...config...}}
        data = {
            "user": self.user_email,
            "org": self.project_uuid,
            "channeltype_code": channel_type,
            "data": config,
        }

        # Debug logging - show complete request
        logger.debug("=" * 80)
        logger.debug("FLOWS API REQUEST")
        logger.debug("=" * 80)
        logger.debug(f"URL: {url}")
        logger.debug(f"Headers: {self.headers}")
        logger.debug("Payload:")
        logger.debug(f"  user: {data['user']}")
        logger.debug(f"  org: {data['org']}")
        logger.debug(f"  channeltype_code: {data['channeltype_code']}")
        logger.debug(f"  data: {data['data']}")
        logger.debug("=" * 80)

        response = requests.post(url, headers=self.headers, json=data)

        # Debug logging - show response
        logger.debug("FLOWS API RESPONSE")
        logger.debug("=" * 80)
        logger.debug(f"Status Code: {response.status_code}")
        logger.debug(f"Headers: {dict(response.headers)}")
        logger.debug(f"Body: {response.text}")
        logger.debug("=" * 80)

        return response
