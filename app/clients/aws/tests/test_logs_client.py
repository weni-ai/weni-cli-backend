from typing import Any

import pytest
from pytest_mock import MockerFixture

from app.clients.aws.logs_client import AWSLogsClient
from app.core.config import settings

# Constants for status codes and expected response values
HTTP_STATUS_OK = 200
EXPECTED_CALL_COUNT_ZERO = 0
EXPECTED_CALL_COUNT_ONE = 1
EXPECTED_CALL_COUNT_TWO = 2

# Constants for test data
TEST_FUNCTION_NAME = "test-function"
TEST_LOG_GROUP_NAME = settings.AGENT_LOG_GROUP
TEST_START_TIME_MS_REDUCED = 1609459200000 - 60000  # 2020-12-31T23:59:00Z in milliseconds
TEST_START_TIME = 1609459200  # 2021-01-01T00:00:00Z in seconds
TEST_END_TIME = 1609459260  # 2021-01-01T00:01:00Z in seconds
TEST_END_TIME_MS_ADDED = 1609459260000 + 60000  # 2021-01-01T00:01:00Z in milliseconds
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
MOCK_REQUEST_ID = "1234567890"


class MockResourceNotFoundError(Exception):
    """Mock exception for ResourceNotFoundException from AWS boto3"""

    pass


@pytest.fixture
def mock_logs_client(mocker: MockerFixture) -> Any:
    """
    Create a mock for the logs client.
    """
    mock_client = mocker.Mock()
    # Configure the mock client to have the proper exceptions
    mock_client.exceptions.ResourceNotFoundException = MockResourceNotFoundError

    # Mock the boto3 module
    mocker.patch("boto3.client", return_value=mock_client)

    return mock_client


@pytest.fixture
def logs_client(mock_logs_client: Any) -> AWSLogsClient:
    """
    Return an AWSLogsClient instance with the mocked boto3 client.
    """
    return AWSLogsClient()


@pytest.fixture
def test_log_events() -> list[dict[str, Any]]:
    """Fixture for sample log events"""
    return [
        {"timestamp": 1609459300000, "message": "Log message 1", "logStreamName": "stream1"},
        {"timestamp": 1609459400000, "message": "Log message 2", "logStreamName": "stream1"},
        {"timestamp": 1609459200000, "message": "Log message 3", "logStreamName": "stream2"},
    ]


@pytest.fixture
def mock_sleep(mocker: MockerFixture) -> Any:
    """Mock asyncio.sleep to speed up tests"""
    return mocker.patch("asyncio.sleep", return_value=None)


class TestAWSLogsClient:
    def test_init(self, mocker: MockerFixture) -> None:
        """
        Test that the client is initialized with the correct parameters.
        """
        # Setup
        mock_boto3 = mocker.patch("boto3.client")

        # Execute
        logs_client = AWSLogsClient()

        # Assert
        mock_boto3.assert_called_once_with("logs", region_name=mocker.ANY)
        assert logs_client.client is mock_boto3.return_value

    @pytest.mark.asyncio
    async def test_get_function_logs_success(
        self, logs_client: AWSLogsClient, mock_logs_client: Any, test_log_events: list[dict[str, Any]]
    ) -> None:
        """
        Test retrieving logs successfully on the first attempt.
        """
        # Setup
        mock_logs_client.filter_log_events.return_value = {"events": test_log_events}

        # Execute
        result = await logs_client.get_function_logs(
            function_name=TEST_FUNCTION_NAME,
            request_id=MOCK_REQUEST_ID,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
        )

        # Assert
        mock_logs_client.filter_log_events.assert_called_once_with(
            logGroupName=TEST_LOG_GROUP_NAME,
            startTime=TEST_START_TIME_MS_REDUCED,
            endTime=TEST_END_TIME_MS_ADDED,
            filterPattern=f'"{MOCK_REQUEST_ID}"',
        )
        # The logs should be sorted by timestamp
        assert result == sorted(test_log_events, key=lambda x: x["timestamp"])

    @pytest.mark.asyncio
    async def test_get_function_logs_resource_not_found_then_success(
        self, logs_client: AWSLogsClient, mock_logs_client: Any, mock_sleep: Any, test_log_events: list[dict[str, Any]]
    ) -> None:
        """
        Test when logs are not found initially but then become available.
        """
        # Setup
        mock_logs_client.filter_log_events.side_effect = [
            MockResourceNotFoundError("Log group not found"),  # First call raises exception
            {"events": test_log_events},  # Second call succeeds
        ]

        # Execute
        result = await logs_client.get_function_logs(
            function_name=TEST_FUNCTION_NAME,
            request_id=MOCK_REQUEST_ID,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
        )

        # Assert
        assert mock_logs_client.filter_log_events.call_count == EXPECTED_CALL_COUNT_TWO
        assert mock_sleep.call_count == EXPECTED_CALL_COUNT_ONE
        assert mock_sleep.call_args[0][0] == RETRY_DELAY
        assert result == sorted(test_log_events, key=lambda x: x["timestamp"])

    @pytest.mark.asyncio
    async def test_get_function_logs_max_retries_exceeded(
        self, logs_client: AWSLogsClient, mock_logs_client: Any, mock_sleep: Any
    ) -> None:
        """
        Test when logs are not found after max retries.
        """
        # Setup
        # Create enough ResourceNotFoundError exceptions to exceed max_retries
        mock_logs_client.filter_log_events.side_effect = [
            MockResourceNotFoundError("Log group not found")
        ] * MAX_RETRIES

        # Execute and Assert
        # After max_retries, an empty list should be returned
        assert (
            await logs_client.get_function_logs(
                function_name=TEST_FUNCTION_NAME,
                request_id=MOCK_REQUEST_ID,
                start_time=TEST_START_TIME,
                end_time=TEST_END_TIME,
            )
            == []
        )

        assert mock_logs_client.filter_log_events.call_count == MAX_RETRIES
        assert mock_sleep.call_count == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_get_function_logs_other_exception(self, logs_client: AWSLogsClient, mock_logs_client: Any) -> None:
        """
        Test when an unexpected exception occurs.
        """
        # Setup
        test_exception = Exception("Test exception")
        mock_logs_client.filter_log_events.side_effect = test_exception

        # Execute and Assert
        with pytest.raises(Exception) as exc_info:
            await logs_client.get_function_logs(
                function_name=TEST_FUNCTION_NAME,
                request_id=MOCK_REQUEST_ID,
                start_time=TEST_START_TIME,
                end_time=TEST_END_TIME,
            )

        assert exc_info.value is test_exception
        mock_logs_client.filter_log_events.assert_called_once()
