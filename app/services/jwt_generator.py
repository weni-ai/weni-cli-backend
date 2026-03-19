"""
JWT token generation for tool credentials.
"""

from datetime import UTC, datetime, timedelta

import jwt

DEFAULT_EXPIRATION_MINUTES = 2
JWT_PROJECT_KEY = "auth-token"


def generate_jwt_token(
    project_uuid: str,
    secret_key: str,
    expiration_minutes: int | None = None,
) -> str:
    """
    Generate JWT token for project UUID.

    Args:
        project_uuid: The project UUID to include in the token payload.
        secret_key: The RSA private key (PEM format) used to sign the token.
        expiration_minutes: Optional token expiration time in minutes.
                           If not provided, uses default (60 minutes).

    Returns:
        The encoded JWT token string.
    """
    exp_minutes = expiration_minutes or DEFAULT_EXPIRATION_MINUTES
    normalized_key = secret_key.replace("\\n", "\n")
    payload = {
        "project_uuid": project_uuid,
        "exp": datetime.now(UTC) + timedelta(minutes=exp_minutes),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, normalized_key, algorithm="RS256")
