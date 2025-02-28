"""
Tests for the response module.
"""

import json
import re
from datetime import UTC, datetime

from pytest_mock import MockerFixture

from app.core.response import CLIResponse, send_response


def test_basic_response_serialization() -> None:
    """Test basic response serialization functionality."""
    # Create a sample response
    response: CLIResponse = {
        "message": "Operation completed",
        "success": True,
        "data": {"key": "value"},
    }

    # Serialize the response
    result = send_response(response)

    # Verify it's bytes
    assert isinstance(result, bytes)

    # Convert back to dict and verify content
    parsed = json.loads(result.decode().strip())
    assert parsed["message"] == "Operation completed"
    assert parsed["success"] is True
    assert parsed["data"] == {"key": "value"}

    # Verify it has a timestamp and request_id
    assert "timestamp" in parsed
    assert "request_id" in parsed

    # Verify it ends with a newline
    assert result.endswith(b"\n")


def test_timestamp_inclusion() -> None:
    """Test timestamp inclusion behavior."""
    # Test with timestamp (default)
    response1: CLIResponse = {"message": "Test", "success": True}
    result_with_timestamp = send_response(response1)
    parsed = json.loads(result_with_timestamp.decode().strip())
    assert "timestamp" in parsed

    # Verify timestamp format is ISO8601 with Z suffix
    # The format is like: 2023-01-15T12:30:45.123456+00:00Z
    iso8601_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+00:00Z$"
    assert re.match(iso8601_pattern, parsed["timestamp"])

    # Test without timestamp - use a fresh response object
    response2: CLIResponse = {"message": "Test", "success": True}
    result_without_timestamp = send_response(response2, include_timestamp=False)
    parsed = json.loads(result_without_timestamp.decode().strip())
    assert "timestamp" not in parsed


def test_request_id_scenarios() -> None:
    """Test various request ID scenarios."""
    # Scenario 1: No request_id provided, should generate one
    response1: CLIResponse = {"message": "Test", "success": True}
    result1 = send_response(response1, include_timestamp=False)
    parsed1 = json.loads(result1.decode().strip())
    assert "request_id" in parsed1
    assert parsed1["request_id"] is not None

    # Scenario 2: Providing a request_id
    custom_id = "custom-request-123"
    # Create a fresh response dictionary to avoid side effects
    response2: CLIResponse = {"message": "Test", "success": True}
    result2 = send_response(response2, request_id=custom_id, include_timestamp=False)
    parsed2 = json.loads(result2.decode().strip())
    assert parsed2["request_id"] == custom_id

    # Scenario 3: Response already has a request_id
    response_with_id: CLIResponse = {"message": "Test", "success": True, "request_id": "existing-id-456"}
    result3 = send_response(response_with_id, request_id="should-not-use-this", include_timestamp=False)
    parsed3 = json.loads(result3.decode().strip())
    assert parsed3["request_id"] == "existing-id-456"  # Should use existing, not the provided one


def test_uuid_generation(mocker: MockerFixture) -> None:
    """Test UUID generation when no request_id is provided."""
    # Mock uuid4 to return a known value
    mock_uuid = "fake-uuid-12345"
    mocker.patch("app.core.response.uuid4", return_value=mock_uuid)

    response: CLIResponse = {"message": "Test", "success": True}
    result = send_response(response, include_timestamp=False)
    parsed = json.loads(result.decode().strip())

    # The request_id should be the string representation of our mock UUID
    assert parsed["request_id"] == str(mock_uuid)


def test_current_time_usage(mocker: MockerFixture) -> None:
    """Test that current time is used for timestamp."""
    # Create a fixed datetime for testing
    fixed_dt = datetime(2023, 1, 15, 12, 30, 45, 123456, tzinfo=UTC)
    expected_timestamp = "2023-01-15T12:30:45.123456+00:00Z"

    # Create a mock for datetime that returns the fixed datetime from now()
    mock_datetime = mocker.MagicMock(wraps=datetime)
    mock_datetime.now.return_value = fixed_dt
    # Keep the UTC reference
    mock_datetime.UTC = UTC

    # Patch the datetime class
    mocker.patch("app.core.response.datetime", mock_datetime)

    response: CLIResponse = {"message": "Test", "success": True}
    result = send_response(response, include_timestamp=True)
    parsed = json.loads(result.decode().strip())

    assert parsed["timestamp"] == expected_timestamp


def test_edge_cases() -> None:
    """Test edge cases like empty response and different data types."""
    # Empty response
    empty_response: CLIResponse = {}
    result = send_response(empty_response, include_timestamp=False)
    parsed = json.loads(result.decode().strip())
    assert "request_id" in parsed
    assert len(parsed) == 1  # Only request_id should be present

    # Response with various data types
    complex_response: CLIResponse = {
        "message": "Complex response",
        "success": True,
        "data": {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "bool": False,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        },
        "code": "SUCCESS_200",
        "progress": 0.75,
    }

    result = send_response(complex_response, include_timestamp=False)
    parsed = json.loads(result.decode().strip())

    # Verify data types are preserved after serialization
    assert parsed["data"]["string"] == "text"
    assert parsed["data"]["number"] == 42  # noqa: PLR2004
    assert parsed["data"]["float"] == 3.14  # noqa: PLR2004
    assert parsed["data"]["bool"] is False
    assert parsed["data"]["null"] is None
    assert parsed["data"]["list"] == [1, 2, 3]
    assert parsed["data"]["nested"] == {"key": "value"}
    assert parsed["code"] == "SUCCESS_200"
    assert parsed["progress"] == 0.75  # noqa: PLR2004
