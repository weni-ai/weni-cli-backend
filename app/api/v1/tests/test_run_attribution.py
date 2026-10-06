import json
import logging
import re
from uuid import UUID

import pytest
from fastapi import status
from starlette.requests import Request

from app.api.v1.models.requests import RunRequestModel
from app.api.v1.run_attribution import (
    RUN_NOT_ATTRIBUTABLE_CODE,
    RUN_NOT_ATTRIBUTABLE_MESSAGE,
    AttributedRun,
    RunNotAttributableError,
    attributed_run_request,
)
from app.tests.utils import make_cli_bearer_token

_HEADER_PROJECT = "c67bc61e-c2b2-43f1-a409-88dec4bd4b9e"
_RUN_PATH = "/api/v1/runs"
_USER_EMAIL = "dev@example.com"
_QUOTED_FIELD = re.compile(r'([a-z_]+)=("(?:\\.|[^"\\])*")')


def _request(authorization: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": _RUN_PATH,
            "headers": [(b"authorization", authorization.encode())],
            "query_string": b"",
        }
    )


def _run_request() -> RunRequestModel:
    return RunRequestModel(
        project_uuid=UUID(_HEADER_PROJECT),
        definition="{}",
        toolkit_version="1.0.0",
        test_definition="{}",
        agent_key="test_agent",
    )


def _quoted_fields(message: str) -> list[tuple[str, str]]:
    return [(key, json.loads(value)) for key, value in _QUOTED_FIELD.findall(message)]


def _assert_token_absent(caplog: pytest.LogCaptureFixture, authorization: str) -> None:
    assert authorization not in caplog.text
    token = authorization.removeprefix("Bearer ")
    assert token not in caplog.text


async def test_valid_bearer_returns_attributed_run(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    authorization = make_cli_bearer_token(_USER_EMAIL)
    data = _run_request()

    result = await attributed_run_request(_request(authorization), data, _HEADER_PROJECT)

    assert result == AttributedRun(
        request=data,
        authorized_project_uuid=str(UUID(_HEADER_PROJECT)),
        user_email=_USER_EMAIL,
    )
    _assert_token_absent(caplog, authorization)


async def test_upper_case_header_is_canonicalized(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    authorization = make_cli_bearer_token(_USER_EMAIL)
    header = _HEADER_PROJECT.upper()

    result = await attributed_run_request(_request(authorization), _run_request(), header)

    assert result.authorized_project_uuid == _HEADER_PROJECT
    assert result.authorized_project_uuid == str(UUID(header))
    _assert_token_absent(caplog, authorization)


async def test_missing_email_raises_and_logs_one_event(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    authorization = make_cli_bearer_token(None)
    header = _HEADER_PROJECT.upper()

    with pytest.raises(RunNotAttributableError) as captured:
        await attributed_run_request(_request(authorization), _run_request(), header)

    error = captured.value
    assert error.http_status == status.HTTP_403_FORBIDDEN
    assert error.code == RUN_NOT_ATTRIBUTABLE_CODE
    assert error.message == RUN_NOT_ATTRIBUTABLE_MESSAGE
    assert str(UUID(error.request_id)) == error.request_id
    assert not hasattr(error, "status_code")

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].message
    assert message.startswith("event=run_not_attributable")
    assert [key for key, _value in _quoted_fields(message)] == [
        "header_project_uuid",
        "endpoint",
        "request_id",
    ]
    fields = dict(_quoted_fields(message))
    assert fields["header_project_uuid"] == header
    assert fields["endpoint"] == _RUN_PATH
    assert fields["request_id"] == error.request_id
    assert "user_email" not in fields
    _assert_token_absent(caplog, authorization)
