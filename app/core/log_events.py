import json
from collections.abc import Mapping


def format_log_event(event: str, fields: Mapping[str, str | None]) -> str:
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={json.dumps(value)}")
    return " ".join(parts)
