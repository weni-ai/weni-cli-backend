"""Tests for channels endpoints."""

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
    """Return an API path for channels endpoint."""
    return f"{settings.API_PREFIX}/v1/channels"


@pytest.fixture
def project_uuid() -> str:
    """Return a test project UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def valid_request_data(project_uuid: str) -> dict[str, Any]:
    """Return valid request data for channel creation."""
    channel_definition = {
        "channel_type": "WAC",
        "name": "Test Channel",
        "address": "+5511999999999",
        "config": {"wa_pin": "123456", "wa_verified_name": "Test Business"},
    }
    return {"project_uuid": project_uuid, "channel_definition": channel_definition}


@pytest.fixture
def mock_flows_client(mocker: MockerFixture) -> Any:
    """Mock the FlowsClient."""
    mock = mocker.MagicMock()
    mocker.patch("app.api.v1.routers.channels.FlowsClient", return_value=mock)
    return mock


def test_create_channel_success(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test successful channel creation."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_201_CREATED
    mock_response._content = json.dumps(
        {"uuid": "channel-uuid-123", "name": "Test Channel", "address": "+5511999999999", "channel_type": "WAC"}
    ).encode()
    mock_flows_client.create_channel.return_value = mock_response

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
    assert result["uuid"] == "channel-uuid-123"
    assert result["name"] == "Test Channel"
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_bad_request(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with bad request."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_400_BAD_REQUEST
    mock_response._content = json.dumps({"detail": "Invalid channel data"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

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
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_unauthorized(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with unauthorized request."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_401_UNAUTHORIZED
    mock_response._content = json.dumps({"detail": "Invalid authentication credentials"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

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
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_forbidden(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with forbidden access."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_403_FORBIDDEN
    mock_response._content = json.dumps({"detail": "Permission denied"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

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
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_internal_server_error(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test channel creation with internal server error."""
    # Setup
    mock_response = Response()
    mock_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    mock_response._content = json.dumps({"detail": "Internal server error"}).encode()
    mock_flows_client.create_channel.return_value = mock_response

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
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_exception(  # noqa: PLR0913
    client: TestClient,
    api_path: str,
    project_uuid: str,
    valid_request_data: dict[str, Any],
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    """Test exception during channel creation."""
    # Setup
    mock_flows_client.create_channel.side_effect = Exception("Unexpected error")

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
    mock_flows_client.create_channel.assert_called_once()


def test_create_channel_missing_project_uuid(client: TestClient, api_path: str, mock_auth_middleware: None) -> None:
    """Test channel creation with missing project_uuid."""
    # Setup
    invalid_data = {"channel_definition": {"channel_type": "WAC", "name": "Test Channel"}}

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


def test_create_channel_missing_channel_definition(
    client: TestClient, api_path: str, project_uuid: str, mock_auth_middleware: None
) -> None:
    """Test channel creation with missing channel_definition."""
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


def test_create_channel_invalid_project_uuid_format(
    client: TestClient, api_path: str, mock_auth_middleware: None
) -> None:
    """Test channel creation with invalid project_uuid format."""
    # Setup
    invalid_data = {
        "project_uuid": "not-a-valid-uuid",
        "channel_definition": {"channel_type": "WAC", "name": "Test Channel"},
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


class TestChannelProjectBinding:
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
        flows_client = mocker.patch("app.api.v1.routers.channels.FlowsClient")
        body = {**valid_request_data, "project_uuid": _OTHER_PROJECT_UUID}

        response = client.post(api_path, json=body, headers=_headers(project_uuid))

        _assert_mismatch(response)
        flows_client.assert_not_called()
        flows_client.return_value.create_channel.assert_not_called()
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
        mocker.patch("app.api.v1.routers.channels.FlowsClient")
        body = {**valid_request_data, "project_uuid": project_uuid.upper()}

        response = client.post(api_path, json=body, headers=_headers(project_uuid))

        _assert_mismatch(response)
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


_CHANNEL_UUID = "11111111-1111-4111-8111-111111111111"


def _flows_response(status_code: int, payload: Any) -> Response:
    mock_response = Response()
    mock_response.status_code = status_code
    mock_response._content = json.dumps(payload).encode()
    return mock_response


def _channel_payload(project_uuid: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "uuid": _CHANNEL_UUID,
        "name": "Test Channel",
        "address": "+5511999999999",
        "config": {"wa_pin": "123456"},
        "org": project_uuid,
        "is_active": True,
        "channel_type": "WAC",
    }
    payload.update(overrides)
    return payload


def test_list_channels_success(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    channels = [_channel_payload(project_uuid)]
    mock_flows_client.list_channels.return_value = _flows_response(status.HTTP_200_OK, channels)

    response = client.get(
        api_path,
        params={"channel_type": "WAC", "exclude_wpp_demo": True},
        headers=_headers(project_uuid),
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == channels
    mock_flows_client.list_channels.assert_called_once_with(channel_type="WAC", exclude_wpp_demo=True)


def test_list_channels_flows_error(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    mock_flows_client.list_channels.return_value = _flows_response(
        status.HTTP_403_FORBIDDEN, {"detail": "Permission denied"}
    )

    response = client.get(api_path, headers=_headers(project_uuid))

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_get_channel_success(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    channel = _channel_payload(project_uuid)
    mock_flows_client.get_channel.return_value = _flows_response(status.HTTP_200_OK, channel)

    response = client.get(f"{api_path}/{_CHANNEL_UUID}", headers=_headers(project_uuid))

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["uuid"] == _CHANNEL_UUID
    mock_flows_client.get_channel.assert_called_once_with(_CHANNEL_UUID)


def test_get_channel_from_another_project_is_404(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    mock_flows_client.get_channel.return_value = _flows_response(
        status.HTTP_200_OK, _channel_payload(_OTHER_PROJECT_UUID)
    )

    response = client.get(f"{api_path}/{_CHANNEL_UUID}", headers=_headers(project_uuid))

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_update_channel_success(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    mock_flows_client.get_channel.return_value = _flows_response(status.HTTP_200_OK, _channel_payload(project_uuid))
    updated = _channel_payload(project_uuid, name="Renamed", config={"wa_pin": "123456", "extra": "1"})
    mock_flows_client.update_channel.return_value = _flows_response(status.HTTP_200_OK, updated)

    response = client.patch(
        f"{api_path}/{_CHANNEL_UUID}",
        json={"name": "Renamed", "config": {"extra": "1"}},
        headers=_headers(project_uuid),
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["name"] == "Renamed"
    mock_flows_client.update_channel.assert_called_once_with(
        _CHANNEL_UUID,
        name="Renamed",
        address=None,
        config={"extra": "1"},
    )


def test_update_channel_requires_a_field(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_auth_middleware: None,
) -> None:
    response = client.patch(f"{api_path}/{_CHANNEL_UUID}", json={}, headers=_headers(project_uuid))

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_update_channel_does_not_write_another_project(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    mock_flows_client.get_channel.return_value = _flows_response(
        status.HTTP_200_OK, _channel_payload(_OTHER_PROJECT_UUID)
    )

    response = client.patch(
        f"{api_path}/{_CHANNEL_UUID}",
        json={"name": "Renamed"},
        headers=_headers(project_uuid),
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_flows_client.update_channel.assert_not_called()


def test_delete_channel_success(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    mock_flows_client.get_channel.return_value = _flows_response(status.HTTP_200_OK, _channel_payload(project_uuid))
    released = Response()
    released.status_code = status.HTTP_200_OK
    released._content = b""
    mock_flows_client.release_channel.return_value = released

    response = client.delete(f"{api_path}/{_CHANNEL_UUID}", headers=_headers(project_uuid))

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {}
    mock_flows_client.release_channel.assert_called_once_with(_CHANNEL_UUID)


def test_delete_channel_flows_error_skips_release(
    client: TestClient,
    api_path: str,
    project_uuid: str,
    mock_flows_client: Any,
    mock_auth_middleware: None,
) -> None:
    mock_flows_client.get_channel.return_value = _flows_response(
        status.HTTP_404_NOT_FOUND, {"detail": "Not found"}
    )

    response = client.delete(f"{api_path}/{_CHANNEL_UUID}", headers=_headers(project_uuid))

    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_flows_client.release_channel.assert_not_called()
