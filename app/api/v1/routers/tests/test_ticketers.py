"""Tests for ticketers endpoints."""

import json
import logging
import re
import uuid
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from requests import Response

from app.api.v1.project_binding import PROJECT_MISMATCH_CODE, PROJECT_MISMATCH_MESSAGE
from app.core.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(scope="module")
def api_path() -> str:
    """Return an API path for ticketers endpoint."""
    return f"{settings.API_PREFIX}/v1/ticketers"


@pytest.fixture
def project_uuid() -> str:
    """Return a test project UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def valid_request_data(project_uuid: str) -> dict[str, Any]:
    """Return valid request data for ticketer creation."""
    ticketer_definition = {
        "name": "Generic Ticketer Integration",
        "ticketer_type": "generic",
        "config": {
            "base_url": "https://your-ticketer-host",
            "api_token": "your-api-token",
            "skip_webhook_hmac": "yes",
            "project_uuid": project_uuid,
            "project_name": "my org",
        },
    }
    return {"project_uuid": project_uuid, "ticketer_definition": ticketer_definition}


@pytest.fixture
def mock_flows_client(mocker: MockerFixture) -> Any:
    """Mock the FlowsClient."""
    mock = mocker.MagicMock()
    mocker.patch("app.api.v1.routers.ticketers.FlowsClient", return_value=mock)
    return mock


def test_create_ticketer_success(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test successful ticketer creation."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_201_CREATED
    mock_response._content = json.dumps(
        {"uuid": "ticketer-uuid-123", "name": "Generic Ticketer Integration"}
    ).encode()
    mock_flows_client.create_ticketer.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    result = response.json()
    assert result["uuid"] == "ticketer-uuid-123"
    assert result["name"] == "Generic Ticketer Integration"
    mock_flows_client.create_ticketer.assert_called_once()


def test_create_ticketer_bad_request(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test ticketer creation with bad request."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_400_BAD_REQUEST
    mock_response._content = json.dumps({"detail": "Invalid ticketer data"}).encode()
    mock_flows_client.create_ticketer.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    mock_flows_client.create_ticketer.assert_called_once()


def test_create_ticketer_unauthorized(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test ticketer creation with unauthorized request."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_401_UNAUTHORIZED
    mock_response._content = json.dumps({"detail": "Invalid authentication credentials"}).encode()
    mock_flows_client.create_ticketer.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer invalid_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    mock_flows_client.create_ticketer.assert_called_once()


def test_create_ticketer_forbidden(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test ticketer creation with forbidden access."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_403_FORBIDDEN
    mock_response._content = json.dumps({"detail": "Permission denied"}).encode()
    mock_flows_client.create_ticketer.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_flows_client.create_ticketer.assert_called_once()


def test_create_ticketer_internal_server_error(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test ticketer creation with internal server error."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_response._content = json.dumps({"detail": "Internal server error"}).encode()
    mock_flows_client.create_ticketer.return_value = mock_response

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_flows_client.create_ticketer.assert_called_once()


def test_create_ticketer_exception(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test exception during ticketer creation."""
    # Setup
    mock_flows_client.create_ticketer.side_effect = Exception("Unexpected error")

    # Execute
    response = client.post(
        api_path,
        json=valid_request_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Internal server error" in response.json()["detail"]
    mock_flows_client.create_ticketer.assert_called_once()


def test_create_ticketer_missing_project_uuid(client: TestClient, api_path: str, mock_auth_middleware: None) -> None:
    """Test ticketer creation with missing project_uuid."""
    # Setup
    invalid_data = {"ticketer_definition": {"name": "Test", "ticketer_type": "generic", "config": {}}}

    # Execute
    response = client.post(
        api_path,
        json=invalid_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": str(uuid.uuid4()),
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_ticketer_missing_ticketer_definition(
    client: TestClient, api_path: str, project_uuid: str, mock_auth_middleware: None
) -> None:
    """Test ticketer creation with missing ticketer_definition."""
    # Setup
    invalid_data = {"project_uuid": project_uuid}

    # Execute
    response = client.post(
        api_path,
        json=invalid_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": project_uuid,
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_ticketer_invalid_project_uuid_format(
    client: TestClient, api_path: str, mock_auth_middleware: None
) -> None:
    """Test ticketer creation with invalid project_uuid format."""
    # Setup
    invalid_data = {
        "project_uuid": "not-a-valid-uuid",
        "ticketer_definition": {"name": "Test", "ticketer_type": "generic", "config": {}},
    }

    # Execute
    response = client.post(
        api_path,
        json=invalid_data,
        headers={
            "Authorization": "Bearer test_token",
            "X-Project-Uuid": str(uuid.uuid4()),
            "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
        },
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


_OTHER_PROJECT_UUID = "6f1c2c1e-8b7a-4d3e-9c2b-0a1b2c3d4e5f"
_INVALID_BODY_HEADER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
_MISSING_BODY_HEADER = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
_AUTHORIZATION = "Bearer test_token"
_QUOTED_FIELD = re.compile(r'([a-z_]+)=("(?:\\.|[^"\\])*")')


def _quoted_fields(message: str) -> list[tuple[str, str]]:
    return [(key, json.loads(value)) for key, value in _QUOTED_FIELD.findall(message)]


def _mismatch_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.message.startswith("event=project_mismatch_rejected")
    ]


def _headers(project_uuid: str) -> dict[str, str]:
    return {
        "Authorization": _AUTHORIZATION,
        "X-Project-Uuid": project_uuid,
        "X-CLI-Version": settings.CLI_MINIMUM_VERSION,
    }


def _assert_bearer_absent(caplog: pytest.LogCaptureFixture) -> None:
    assert _AUTHORIZATION not in caplog.text
    assert _AUTHORIZATION.removeprefix("Bearer ") not in caplog.text


def _assert_mismatch(response: Any) -> None:
    assert response.status_code == status.HTTP_403_FORBIDDEN
    payload = response.json()
    assert payload["message"] == PROJECT_MISMATCH_MESSAGE
    assert payload["data"] is None
    assert payload["success"] is False
    assert payload["code"] == PROJECT_MISMATCH_CODE
    assert str(uuid.UUID(payload["request_id"])) == payload["request_id"]
    assert set(payload) == {"message", "data", "success", "code", "request_id"}


class TestTicketerProjectBinding:
    def test_mismatched_project_is_403(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        project_uuid: str,
        valid_request_data: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        flows_client = mocker.patch("app.api.v1.routers.ticketers.FlowsClient")
        body = {**valid_request_data, "project_uuid": _OTHER_PROJECT_UUID}

        response = client.post(api_path, json=body, headers=_headers(project_uuid))

        _assert_mismatch(response)
        flows_client.assert_not_called()
        flows_client.return_value.create_ticketer.assert_not_called()
        warnings = _mismatch_warnings(caplog)
        assert len(warnings) == 1
        assert dict(_quoted_fields(warnings[0].message))["endpoint"] == api_path
        _assert_bearer_absent(caplog)

    def test_upper_case_body_is_a_mismatch(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        project_uuid: str,
        valid_request_data: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        mocker.patch("app.api.v1.routers.ticketers.FlowsClient")
        body = {**valid_request_data, "project_uuid": project_uuid.upper()}

        response = client.post(api_path, json=body, headers=_headers(project_uuid))

        _assert_mismatch(response)
        _assert_bearer_absent(caplog)

    def test_nested_project_uuid_is_not_checked(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        project_uuid: str,
        valid_request_data: dict[str, Any],
        mocker: MockerFixture,
        mock_auth_middleware: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        flows_client = mocker.patch("app.api.v1.routers.ticketers.FlowsClient")
        mock_response = Response()
        mock_response.status_code = status.HTTP_201_CREATED
        mock_response._content = json.dumps({"uuid": "ticketer-uuid-123"}).encode()
        flows_client.return_value.create_ticketer.return_value = mock_response
        definition = {
            **valid_request_data["ticketer_definition"],
            "config": {
                **valid_request_data["ticketer_definition"]["config"],
                "project_uuid": _OTHER_PROJECT_UUID,
            },
        }
        body = {"project_uuid": project_uuid, "ticketer_definition": definition}

        response = client.post(api_path, json=body, headers=_headers(project_uuid))

        assert response.status_code == status.HTTP_201_CREATED
        flows_client.return_value.create_ticketer.assert_called_once_with(definition)
        assert definition["config"]["project_uuid"] == _OTHER_PROJECT_UUID
        assert _mismatch_warnings(caplog) == []
        _assert_bearer_absent(caplog)

    @pytest.mark.parametrize(
        ("body_project_uuid", "header_project_uuid"),
        [
            pytest.param("not-a-uuid", _INVALID_BODY_HEADER, id="not-a-uuid"),
            pytest.param(None, _MISSING_BODY_HEADER, id="missing"),
        ],
    )
    def test_invalid_or_missing_body_project_returns_422(  # noqa: PLR0913
        self,
        client: TestClient,
        api_path: str,
        valid_request_data: dict[str, Any],
        mock_auth_middleware: None,
        caplog: pytest.LogCaptureFixture,
        body_project_uuid: str | None,
        header_project_uuid: str,
    ) -> None:
        caplog.set_level(logging.INFO)
        body = {key: value for key, value in valid_request_data.items() if key != "project_uuid"}
        if body_project_uuid is not None:
            body["project_uuid"] = body_project_uuid

        response = client.post(api_path, json=body, headers=_headers(header_project_uuid))

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert _mismatch_warnings(caplog) == []
        _assert_bearer_absent(caplog)
