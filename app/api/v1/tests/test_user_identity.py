import base64
from typing import Any

import jwt
import pytest

from app.api.v1.user_identity import read_user_email
from app.tests.utils import CLI_BEARER_TOKEN_TEST_SECRET, make_cli_bearer_token

_BEARER_PREFIX = "Bearer "
_INVALID_BASE64_PAYLOAD_SEGMENT = "abcde"
_NON_OBJECT_PAYLOAD = b"[]"


def _token_with_payload(payload: dict[str, Any]) -> str:
    token = jwt.encode(payload, CLI_BEARER_TOKEN_TEST_SECRET, algorithm="HS256")
    return _BEARER_PREFIX + token


def _token_with_payload_segment(payload_segment: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    signature = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()
    return f"{_BEARER_PREFIX}{header}.{payload_segment}.{signature}"


def test_valid_bearer_returns_email() -> None:
    authorization = make_cli_bearer_token("dev@example.com")

    assert read_user_email(authorization) == "dev@example.com"


@pytest.mark.parametrize(
    "authorization",
    [
        pytest.param(None, id="missing-header"),
        pytest.param("", id="empty-string"),
        pytest.param("Basic abc", id="non-bearer-scheme"),
        pytest.param("Bearer", id="bearer-with-no-token"),
        pytest.param(f"{_BEARER_PREFIX}not-a-jwt", id="not-a-jwt"),
        pytest.param(
            _token_with_payload_segment(_INVALID_BASE64_PAYLOAD_SEGMENT),
            id="payload-segment-is-not-base64-json",
        ),
        pytest.param(
            _token_with_payload_segment(base64.urlsafe_b64encode(_NON_OBJECT_PAYLOAD).rstrip(b"=").decode()),
            id="payload-is-json-but-not-an-object",
        ),
        pytest.param(make_cli_bearer_token(None), id="token-without-email"),
        pytest.param(make_cli_bearer_token(""), id="empty-email"),
        pytest.param(_token_with_payload({"email": 123}), id="email-is-not-a-string"),
        pytest.param(_token_with_payload({"email": ["dev@example.com"]}), id="email-is-a-list"),
    ],
)
def test_unreadable_identity_returns_none(authorization: str | None) -> None:
    assert read_user_email(authorization) is None
