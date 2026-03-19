"""Tests for JWT token generation."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.services.jwt_generator import (
    DEFAULT_EXPIRATION_MINUTES,
    JWT_PROJECT_KEY,
    generate_jwt_token,
)


@pytest.fixture(scope="module")
def rsa_key_pair() -> tuple[str, str]:
    """Generate an RSA key pair for testing.

    Returns:
        A tuple of (private_key_pem, public_key_pem) as strings.
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )

    return private_pem, public_pem


class TestGenerateJwtToken:
    """Tests for the generate_jwt_token function."""

    def test_generates_valid_token(self, rsa_key_pair: tuple[str, str]) -> None:
        """Test that generate_jwt_token produces a valid JWT decodable with the public key."""
        private_key, public_key = rsa_key_pair
        project_uuid = str(uuid4())

        token = generate_jwt_token(project_uuid, private_key)

        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        assert decoded["project_uuid"] == project_uuid
        assert "exp" in decoded
        assert "iat" in decoded

    def test_default_expiration(self, rsa_key_pair: tuple[str, str]) -> None:
        """Test that the default expiration is DEFAULT_EXPIRATION_MINUTES (2 minutes)."""
        private_key, public_key = rsa_key_pair
        project_uuid = str(uuid4())

        token = generate_jwt_token(project_uuid, private_key)

        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        exp = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        iat = datetime.fromtimestamp(decoded["iat"], tz=UTC)

        # The difference should be approximately DEFAULT_EXPIRATION_MINUTES
        expected_minutes = DEFAULT_EXPIRATION_MINUTES
        delta = exp - iat
        assert abs(delta - timedelta(minutes=expected_minutes)) < timedelta(seconds=5)

    def test_custom_expiration(self, rsa_key_pair: tuple[str, str]) -> None:
        """Test that a custom expiration is applied correctly."""
        private_key, public_key = rsa_key_pair
        project_uuid = str(uuid4())
        custom_minutes = 30

        token = generate_jwt_token(project_uuid, private_key, expiration_minutes=custom_minutes)

        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        exp = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        iat = datetime.fromtimestamp(decoded["iat"], tz=UTC)

        delta = exp - iat
        assert abs(delta - timedelta(minutes=custom_minutes)) < timedelta(seconds=5)

    def test_token_contains_project_uuid(self, rsa_key_pair: tuple[str, str]) -> None:
        """Test that the token payload contains the correct project_uuid."""
        private_key, public_key = rsa_key_pair
        project_uuid = str(uuid4())

        token = generate_jwt_token(project_uuid, private_key)

        decoded = jwt.decode(token, public_key, algorithms=["RS256"])
        assert decoded["project_uuid"] == project_uuid

    def test_jwt_project_key_constant(self) -> None:
        """Test that JWT_PROJECT_KEY is 'auth-token'."""
        assert JWT_PROJECT_KEY == "auth-token"
