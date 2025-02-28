import json
from datetime import UTC, datetime
from typing import Any, TypedDict
from uuid import uuid4


class CLIResponse(TypedDict, total=False):
    message: str
    success: bool
    timestamp: str
    data: dict[str, Any] | None
    request_id: str | None
    code: str | None
    progress: float | None


def send_response(response: CLIResponse, request_id: str | None = None, include_timestamp: bool = True) -> bytes:
    """
    Format and serialize a CLI response with additional metadata.

    Args:
        response: The response object to send
        request_id: Optional request identifier for correlation
        include_timestamp: Whether to include a timestamp in the response

    Returns:
        Serialized response as bytes with newline
    """
    # Add timestamp if requested
    if include_timestamp:
        response["timestamp"] = datetime.now(UTC).isoformat() + "Z"

    # Add request_id if provided or generate one
    if "request_id" not in response:
        response["request_id"] = request_id or str(uuid4())

    return json.dumps(response).encode() + b"\n"
