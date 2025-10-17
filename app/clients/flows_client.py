import base64
import json
import logging
import time

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

        # Debug: Log token info
        self._log_token_info(user_auth_token)

    def _log_token_info(self, token: str) -> None:
        """Log JWT token information for debugging."""
        try:
            # Remove "Bearer " prefix if present
            clean_token = token[7:] if token.startswith("Bearer ") else token

            # JWT format: header.payload.signature
            parts = clean_token.split(".")
            if len(parts) != JWT_PARTS_COUNT:
                logger.debug("Token format: Invalid (not a JWT)")
                return

            # Decode payload
            payload = parts[1]
            padding = len(payload) % 4
            if padding:
                payload += "=" * (4 - padding)

            decoded = base64.urlsafe_b64decode(payload)
            payload_data = json.loads(decoded)

            # Log useful info
            logger.debug("=" * 80)
            logger.debug("JWT TOKEN INFO")
            logger.debug("=" * 80)
            logger.debug(f"Token completo: Bearer {clean_token}")
            logger.debug(f"Email: {payload_data.get('email', 'N/A')}")
            logger.debug(f"Username: {payload_data.get('preferred_username', 'N/A')}")
            logger.debug(f"Name: {payload_data.get('name', 'N/A')}")
            logger.debug(f"Subject (sub): {payload_data.get('sub', 'N/A')}")
            logger.debug(f"Client (azp): {payload_data.get('azp', 'N/A')}")
            logger.debug(f"Issuer: {payload_data.get('iss', 'N/A')}")

            # Check token expiration
            exp = payload_data.get("exp")
            if exp:
                is_expired = time.time() > exp
                logger.debug(f"Expires at: {exp} ({'EXPIRED' if is_expired else 'Valid'})")

            logger.debug("=" * 80)

        except Exception as e:
            logger.debug(f"Could not decode token info: {e}")

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
            payload_data = json.loads(decoded)

            # Extract email from payload
            email = payload_data.get("email", "")
            return email

        except Exception as e:
            logger.warning(f"Failed to extract email from token: {e}")
            return ""

    def create_channel(self, channel_definition: dict) -> Response:
        url = f"{self.base_url}/api/v2/internals/channel/"

        # Extract fields from channel_definition
        channel_type = channel_definition.get("channel_type", "")
        config = channel_definition.get("config", {})

        # Build the payload according to Flows API format
        # NOTE: Flows receives Keycloak token in headers and extracts user from there
        # The 'user' field is included for compatibility but Flows may extract from token
        data = {
            "user": self.user_email,  # Extracted from JWT token
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
