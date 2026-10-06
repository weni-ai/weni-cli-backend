import json
import logging
import re
from uuid import UUID

import pytest
from fastapi import status
from starlette.requests import Request

from app.api.v1.project_binding import (
    PROJECT_MISMATCH_CODE,
    PROJECT_MISMATCH_MESSAGE,
    ProjectMismatchError,
    ensure_body_project_is_authorized,
)
from app.tests.utils import make_cli_bearer_token

_HEADER_PROJECT = "c67bc61e-c2b2-43f1-a409-88dec4bd4b9e"
_OTHER_PROJECT = "6f1c2c1e-8b7a-4d3e-9c2b-0a1b2c3d4e5f"
_RUN_PATH = "/api/v1/runs"
_USER_EMAIL = "dev@example.com"
_QUOTED_FIELD = re.compile(r'([a-z_]+)=("(?:\\.|[^"\\])*")')


def _request(authorization: str | None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": _RUN_PATH,
            "headers": headers,
            "query_string": b"",
        }
    )


def _quoted_fields(message: str) -> list[tuple[str, str]]:
    return [(key, json.loads(value)) for key, value in _QUOTED_FIELD.findall(message)]


def _assert_token_absent(caplog: pytest.LogCaptureFixture, authorization: str) -> None:
    assert authorization not in caplog.text
    token = authorization.removeprefix("Bearer ")
    assert token not in caplog.text


def test_identical_strings_return_none_without_a_log_record(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    authorization = make_cli_bearer_token(_USER_EMAIL)

    result = ensure_body_project_is_authorized(  # type: ignore[func-returns-value]
        _request(authorization), _HEADER_PROJECT, _HEADER_PROJECT
    )

    assert result is None
    assert caplog.records == []
    _assert_token_absent(caplog, authorization)


@pytest.mark.parametrize(
    "raw_body",
    [
        pytest.param(_OTHER_PROJECT, id="different-uuid"),
        pytest.param(_HEADER_PROJECT.upper(), id="same-uuid-upper-case"),
        pytest.param(_HEADER_PROJECT.replace("-", ""), id="same-uuid-without-hyphens"),
    ],
)
def test_mismatch_raises_and_logs_one_event(caplog: pytest.LogCaptureFixture, raw_body: str) -> None:
    caplog.set_level(logging.INFO)
    authorization = make_cli_bearer_token(_USER_EMAIL)

    with pytest.raises(ProjectMismatchError) as captured:
        ensure_body_project_is_authorized(_request(authorization), _HEADER_PROJECT, raw_body)

    error = captured.value
    assert error.http_status == status.HTTP_403_FORBIDDEN
    assert error.code == PROJECT_MISMATCH_CODE
    assert error.message == PROJECT_MISMATCH_MESSAGE
    assert str(UUID(error.request_id)) == error.request_id
    assert not hasattr(error, "status_code")
    assert _HEADER_PROJECT not in error.message
    assert raw_body not in error.message

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].message
    assert message.startswith("event=project_mismatch_rejected")
    assert [key for key, _value in _quoted_fields(message)] == [
        "header_project_uuid",
        "body_project_uuid",
        "endpoint",
        "request_id",
        "user_email",
    ]
    fields = dict(_quoted_fields(message))
    assert fields["header_project_uuid"] == _HEADER_PROJECT
    assert fields["body_project_uuid"] == raw_body
    assert fields["endpoint"] == _RUN_PATH
    assert fields["request_id"] == error.request_id
    assert fields["user_email"] == _USER_EMAIL
    _assert_token_absent(caplog, authorization)


@pytest.mark.parametrize(
    "authorization",
    [
        pytest.param(make_cli_bearer_token(None), id="token-without-email"),
        pytest.param("Bearer not-a-jwt", id="header-is-not-a-jwt"),
        pytest.param(None, id="no-authorization-header"),
    ],
)
def test_mismatch_omits_user_email_when_identity_cannot_be_read(
    caplog: pytest.LogCaptureFixture,
    authorization: str | None,
) -> None:
    caplog.set_level(logging.INFO)

    with pytest.raises(ProjectMismatchError):
        ensure_body_project_is_authorized(_request(authorization), _HEADER_PROJECT, _OTHER_PROJECT)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    fields = dict(_quoted_fields(warnings[0].message))
    assert "user_email" not in fields
    if authorization is not None:
        _assert_token_absent(caplog, authorization)


def test_raw_body_with_quote_and_newline_stays_on_one_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    authorization = make_cli_bearer_token(_USER_EMAIL)
    raw_body = 'say "hi"\nnext'

    with pytest.raises(ProjectMismatchError):
        ensure_body_project_is_authorized(_request(authorization), _HEADER_PROJECT, raw_body)

    message = caplog.records[0].message
    assert "\n" not in message
    assert "\r" not in message
    assert dict(_quoted_fields(message))["body_project_uuid"] == raw_body
    _assert_token_absent(caplog, authorization)
