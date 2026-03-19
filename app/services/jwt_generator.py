"""
JWT token generation for tool credentials.
"""

from datetime import UTC, datetime, timedelta

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

DEFAULT_EXPIRATION_MINUTES = 2
JWT_PROJECT_KEY = "auth-token"


def _load_private_key(secret_key: str) -> RSAPrivateKey:
    """Load an RSA private key from a PEM string, handling common env var formatting issues."""
    normalized = secret_key.replace("\\n", "\n").replace("\r\n", "\n").strip()
    key = load_pem_private_key(normalized.encode(), password=None)
    if not isinstance(key, RSAPrivateKey):
        raise ValueError(f"Expected RSA private key, got {type(key).__name__}")
    return key


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
    private_key = _load_private_key(secret_key)
    payload = {
        "project_uuid": project_uuid,
        "exp": datetime.now(UTC) + timedelta(minutes=exp_minutes),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, private_key, algorithm="RS256")
