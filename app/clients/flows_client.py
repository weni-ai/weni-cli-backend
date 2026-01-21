import base64
import json
import logging
from typing import Any

import requests
from requests import Response

from app.core.config import settings

logger = logging.getLogger(__name__)

# JWT token constants
JWT_PARTS_COUNT = 3  # header.payload.signature


class FlowsClient:
    base_url: str = ""
    headers: dict[str, str] = {}
    project_uuid: str = ""
    user_email: str = ""

    def __init__(self, user_auth_token: str, project_uuid: str):
        self.headers = {"Authorization": user_auth_token}
        self.base_url = settings.FLOWS_BASE_URL
        self.project_uuid = project_uuid
        self.user_email = self._extract_email_from_token(user_auth_token)




    def _extract_email_from_token(self, token: str) -> str:
        """Extract email from JWT token payload."""
        try:
            # Remove "Bearer " prefix if present
            if token.startswith("Bearer "):
                token = token[7:]

            # JWT format: header.payload.signature
            parts = token.split(".")
            if len(parts) != JWT_PARTS_COUNT:
                logger.warning("Invalid JWT token format")
                return ""

            # Decode payload (second part)
            payload = parts[1]
            # Add padding if needed
            padding = len(payload) % 4
            if padding:
                payload += "=" * (4 - padding)

            decoded = base64.urlsafe_b64decode(payload)
            payload_data: dict[str, Any] = json.loads(decoded)

            # Extract email from payload ensuring str type
            email = payload_data.get("email", "")
            return email if isinstance(email, str) else str(email)

        except Exception as e:
            logger.warning(f"Failed to extract email from token: {e}")
            return ""

    def create_channel(self, channel_definition: dict) -> Response:
        url = f"{self.base_url}/api/v2/internals/channel/"

        # Extract fields from channel_definition
        channel_type = channel_definition.get("channel_type", "")
        name = channel_definition.get("name", "")
        schemes = channel_definition.get("schemes", [])
        address = channel_definition.get("address", "")
        config = channel_definition.get("config", {})

        # The Flows internal endpoint forwards only `data` to the channel claim view.
        # Include name/address/schemes inside `data` so the channel saves them correctly (e.g. E2).
        data_payload: dict[str, Any] = dict(config) if isinstance(config, dict) else {}
        if isinstance(name, str) and name.strip() and "name" not in data_payload:
            data_payload["name"] = name.strip()
        if isinstance(address, str) and address.strip() and "address" not in data_payload:
            data_payload["address"] = address.strip()
        if isinstance(schemes, list) and schemes and "schemes" not in data_payload:
            data_payload["schemes"] = ",".join([s for s in schemes if isinstance(s, str) and s.strip()])

        data_str = json.dumps(data_payload)

        form_data = {
            "user": self.user_email,
            "org": self.project_uuid,
            "channeltype_code": channel_type,
            "data": data_str,
        }

        response = requests.post(url, headers=self.headers, data=form_data)

        return response
