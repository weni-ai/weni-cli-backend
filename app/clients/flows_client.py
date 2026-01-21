import base64
import json
import logging
from typing import Any
from uuid import uuid4

import requests
from requests import Response

from app.core.config import settings

logger = logging.getLogger(__name__)

# JWT token constants
JWT_PARTS_COUNT = 3  # header.payload.signature


def _mask_secret(value: str, *, keep_start: int = 14, keep_end: int = 6) -> str:
    if not value:
        return ""
    if len(value) <= keep_start + keep_end + 3:
        return "***"
    return f"{value[:keep_start]}...{value[-keep_end:]}"


def _truncate(value: str, limit: int = 800) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


def _redact_config(value: Any) -> Any:
    """Redact secrets and truncate large strings for safe logging."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k).lower()
            if any(s in key for s in ("authorization", "token", "secret", "password", "api_key", "apikey")):
                redacted[str(k)] = "***REDACTED***"
                continue
            if key in {"send_template", "receive_template"} and isinstance(v, str):
                redacted[str(k)] = _truncate(v, 600)
                continue
            redacted[str(k)] = _redact_config(v)
        return redacted
    if isinstance(value, list):
        return [_redact_config(v) for v in value]
    if isinstance(value, str):
        return _truncate(value, 800)
    return value


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
        request_id = uuid4().hex[:10]

        # Extract fields from channel_definition
        channel_type = channel_definition.get("channel_type", "")
        name = channel_definition.get("name", "")
        schemes = channel_definition.get("schemes", [])
        address = channel_definition.get("address", "")
        config = channel_definition.get("config", {})

        # The Flows internal endpoint (rapidpro-apps) forwards only `data` into the claim view POST.
        # To make sure E2 (External API V2) saves the intended name/schemes/address, include them in `data`.
        if isinstance(schemes, list):
            schemes_str = ",".join([s for s in schemes if isinstance(s, str) and s.strip()])
        else:
            schemes_str = ""
        data_payload: dict[str, Any] = dict(config) if isinstance(config, dict) else {"config": config}
        # Only set if absent to avoid overriding user-provided config keys.
        if isinstance(name, str) and name.strip() and "name" not in data_payload:
            data_payload["name"] = name.strip()
        if isinstance(address, str) and address.strip() and "address" not in data_payload:
            data_payload["address"] = address.strip()
        if isinstance(schemes, list) and schemes and "schemes" not in data_payload:
            data_payload["schemes"] = schemes
        data_str = json.dumps(data_payload)

        form_data = {
            "user": self.user_email,
            "org": self.project_uuid,
            "channeltype_code": channel_type,
            "name": name,
            "address": address,
            "schemes": schemes_str,
            "data": data_str,
        }

        safe_headers = dict(self.headers)
        if isinstance(safe_headers.get("Authorization"), str):
            safe_headers["Authorization"] = _mask_secret(safe_headers["Authorization"])

        try:
            config_obj = json.loads(data_str) if isinstance(data_str, str) else config
        except Exception:
            config_obj = config

        logger.info(
            "Flows create_channel [%s] -> type=%s name=%r address=%r schemes=%r org=%s user=%s",
            request_id,
            channel_type,
            name,
            address,
            schemes_str,
            self.project_uuid,
            self.user_email,
        )
        logger.debug(
            "Flows create_channel [%s] REQUEST url=%s headers=%s form_data=%s",
            request_id,
            url,
            safe_headers,
            {
                "user": self.user_email,
                "org": self.project_uuid,
                "channeltype_code": channel_type,
                "name": name,
                "address": address,
                "schemes": schemes_str,
                "data": _redact_config(config_obj),
            },
        )

        response = requests.post(url, headers=self.headers, data=form_data)

        # Log response details for debugging mismatched names (e.g. defaulting to "External API V2")
        body_text = ""
        try:
            body_text = response.text
        except Exception:
            body_text = "<unable to read response body>"

        logger.info("Flows create_channel [%s] <- status=%s", request_id, response.status_code)
        logger.debug(
            "Flows create_channel [%s] RESPONSE headers=%s body=%s",
            request_id,
            dict(response.headers),
            _truncate(body_text, 2000),
        )

        return response
