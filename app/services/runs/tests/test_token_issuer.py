import json
import logging
import re

import jwt
import pytest
from pytest_mock import MockerFixture

from app.core.config import settings
from app.services.jwt_generator import DEFAULT_EXPIRATION_MINUTES
from app.services.runs.token_issuer import RunTokenIssuer, RunType
from app.tests.utils import generate_rsa_key_pair

_AUTHORIZED_PROJECT = "c67bc61e-c2b2-43f1-a409-88dec4bd4b9e"
_USER_EMAIL = "dev@example.com"
_AGENT_KEY = "test_agent"
_TOOL_KEY = "test_tool"
_REQUEST_ID = "11111111-1111-4111-8111-111111111111"
_QUOTED_FIELD = re.compile(r'([a-z_]+)=("(?:\\.|[^"\\])*")')


def _quoted_fields(message: str) -> list[tuple[str, str]]:
    return [(key, json.loads(value)) for key, value in _QUOTED_FIELD.findall(message)]


def _issuer(
    *,
    tool_key: str | None = _TOOL_KEY,
    run_type: RunType = "passive",
) -> RunTokenIssuer:
    return RunTokenIssuer(
        authorized_project_uuid=_AUTHORIZED_PROJECT,
        user_email=_USER_EMAIL,
        agent_key=_AGENT_KEY,
        tool_key=tool_key,
        run_type=run_type,
        request_id=_REQUEST_ID,
    )


@pytest.fixture
def public_pem(mocker: MockerFixture) -> str:
    private_pem, public_pem = generate_rsa_key_pair()
    mocker.patch.object(settings, "JWT_SECRET_KEY", private_pem)
    return public_pem


def _minted_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.message.startswith("event=run_token_minted")
    ]


def _assert_token_absent(caplog: pytest.LogCaptureFixture, token: str) -> None:
    assert token not in caplog.text
    for part in token.split("."):
        assert part not in caplog.text


def test_passive_issue_decodes_to_the_unchanged_contract(
    caplog: pytest.LogCaptureFixture,
    public_pem: str,
) -> None:
    caplog.set_level(logging.INFO)

    token = _issuer().issue()

    decoded = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert set(decoded) == {"project_uuid", "exp", "iat"}
    assert decoded["project_uuid"] == _AUTHORIZED_PROJECT
    assert decoded["exp"] - decoded["iat"] == DEFAULT_EXPIRATION_MINUTES * 60

    records = _minted_records(caplog)
    assert len(records) == 1
    message = records[0].message
    assert [key for key, _value in _quoted_fields(message)] == [
        "user_email",
        "project_uuid",
        "agent_key",
        "tool_key",
        "run_type",
        "request_id",
    ]
    fields = dict(_quoted_fields(message))
    assert fields["user_email"] == _USER_EMAIL
    assert fields["project_uuid"] == decoded["project_uuid"]
    assert fields["agent_key"] == _AGENT_KEY
    assert fields["tool_key"] == _TOOL_KEY
    assert fields["run_type"] == "passive"
    assert fields["request_id"] == _REQUEST_ID
    _assert_token_absent(caplog, token)


def test_active_issue_omits_tool_key(caplog: pytest.LogCaptureFixture, public_pem: str) -> None:
    caplog.set_level(logging.INFO)

    token = _issuer(tool_key=None, run_type="active").issue()

    records = _minted_records(caplog)
    assert len(records) == 1
    fields = dict(_quoted_fields(records[0].message))
    assert "tool_key" not in fields
    assert fields["run_type"] == "active"
    _assert_token_absent(caplog, token)


def test_two_calls_write_two_records(caplog: pytest.LogCaptureFixture, public_pem: str) -> None:
    caplog.set_level(logging.INFO)
    issuer = _issuer()

    issued = [issuer.issue(), issuer.issue()]

    assert len(_minted_records(caplog)) == len(issued)
    for token in issued:
        _assert_token_absent(caplog, token)
