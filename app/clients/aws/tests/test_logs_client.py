from datetime import UTC, datetime
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
TEST_START_TIME = 1609459200  # 2021-01-01T00:00:00Z in seconds
TEST_START_TIME_MS = TEST_START_TIME * 1000  # 2021-01-01T00:00:00Z in milliseconds
TEST_START_TIME_DATETIME = datetime.fromtimestamp(TEST_START_TIME, UTC)
TEST_END_TIME = 1609459260  # 2021-01-01T00:01:00Z in seconds
TEST_END_TIME_MS = TEST_END_TIME * 1000  # 2021-01-01T00:01:00Z in milliseconds
TEST_END_TIME_DATETIME = datetime.fromtimestamp(TEST_END_TIME, UTC) 
TEST_LOG_GROUP_ARN = "log-group-arn"
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
        result, _ = await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=f'"{MOCK_REQUEST_ID}"',
        )

        # Assert
        mock_logs_client.filter_log_events.assert_called_once_with(
            logGroupIdentifier=TEST_LOG_GROUP_ARN,
            startTime=TEST_START_TIME_MS,
            endTime=TEST_END_TIME_MS,
            limit=1000,
            filterPattern=f'""{MOCK_REQUEST_ID}"" -"START RequestId" -"END RequestId" -"REPORT RequestId" -"INIT_START Runtime Version"',  # noqa: E501
        )
        # The logs should be sorted by timestamp
        assert result == sorted(test_log_events, key=lambda x: x["timestamp"])


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
                log_group_arn=TEST_LOG_GROUP_ARN,
                start_time=TEST_START_TIME_DATETIME,
                end_time=TEST_END_TIME_DATETIME,
                filter_pattern=f'{MOCK_REQUEST_ID}',
            )

        assert exc_info.value is test_exception
        mock_logs_client.filter_log_events.assert_called_once()
