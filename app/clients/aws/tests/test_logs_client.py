import re
from datetime import UTC, datetime
from typing import Any

import pytest
from pytest_mock import MockerFixture

from app.clients.aws.logs_client import AWSLogsClient
from app.core.config import settings

# Constants for status codes and expected response values
HTTP_STATUS_OK = 200

# Constants for test data
TEST_FUNCTION_NAME = "test-function"
TEST_LAMBDA_NAME = "get_address-42"
TEST_LOG_GROUP_NAME = settings.AGENT_LOG_GROUP
TEST_START_TIME = 1609459200  # 2021-01-01T00:00:00Z in seconds
TEST_START_TIME_MS = TEST_START_TIME * 1000
TEST_START_TIME_DATETIME = datetime.fromtimestamp(TEST_START_TIME, UTC)
TEST_END_TIME = 1609459260  # 2021-01-01T00:01:00Z in seconds
TEST_END_TIME_MS = TEST_END_TIME * 1000
TEST_END_TIME_DATETIME = datetime.fromtimestamp(TEST_END_TIME, UTC)
TEST_LOG_GROUP_ARN = "arn:aws:logs:us-east-1:123456789012:log-group:/aws/lambda/agents-central:*"
TEST_QUERY_ID = "query-id-1234"
MOCK_REQUEST_ID = "1234567890"

# Page limit used to assert pagination behavior — kept tiny so tests remain fast.
TEST_PAGE_LIMIT = 2
# Number of polling iterations expected before the Insights query reaches Complete.
EXPECTED_POLL_CALLS_UNTIL_COMPLETE = 3
# Arbitrary fixed epoch-ms value for cursor round-trip assertions.
SAMPLE_CURSOR_TIMESTAMP_MS = 1700000000000


def _insights_row(timestamp: str, message: str, log_stream: str = "2021/01/01/get_address-42[$LATEST][abc]") -> list:
    """Build a Logs Insights result row in the boto3 shape."""
    return [
        {"field": "@timestamp", "value": timestamp},
        {"field": "@message", "value": message},
        {"field": "@logStream", "value": log_stream},
    ]


@pytest.fixture
def mock_logs_client(mocker: MockerFixture) -> Any:
    """Create a mock for the boto3 logs client."""
    mock_client = mocker.Mock()
    mock_client.start_query.return_value = {"queryId": TEST_QUERY_ID}
    mocker.patch("boto3.client", return_value=mock_client)
    return mock_client


@pytest.fixture
def logs_client(mock_logs_client: Any) -> AWSLogsClient:
    return AWSLogsClient()


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real sleeps during tests."""
    monkeypatch.setattr(settings, "AWS_LOGS_INSIGHTS_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(settings, "AWS_LOGS_INSIGHTS_MAX_WAIT_SECONDS", 5)


class TestAWSLogsClient:
    def test_init(self, mocker: MockerFixture) -> None:
        mock_boto3 = mocker.patch("boto3.client")
        client = AWSLogsClient()
        mock_boto3.assert_called_once_with("logs", region_name=mocker.ANY)
        assert client.client is mock_boto3.return_value

    @pytest.mark.asyncio
    async def test_get_function_logs_success_single_page(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        """Returns the events as `{timestamp, message, logStreamName}` and no next_token when below the page limit."""
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                _insights_row("2021-01-01 00:00:01.000", "first event"),
                _insights_row("2021-01-01 00:00:03.500", "second event"),
                _insights_row("2021-01-01 00:00:02.250", "third event"),
            ],
        }

        events, next_token = await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
        )

        assert next_token is None
        assert [e["message"] for e in events] == ["first event", "third event", "second event"]
        assert all(set(evt.keys()) == {"timestamp", "message", "logStreamName"} for evt in events)

    @pytest.mark.asyncio
    async def test_get_function_logs_uses_logstream_filter(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        """Critical multi-tenancy guard: the query MUST scope by @logStream."""
        mock_logs_client.get_query_results.return_value = {"status": "Complete", "results": []}

        await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
        )

        kwargs = mock_logs_client.start_query.call_args.kwargs
        assert kwargs["logGroupIdentifiers"] == [TEST_LOG_GROUP_ARN]
        assert "filter @logStream like" in kwargs["queryString"]
        # Ensure we anchor the lambda name with the `[` opener of the version segment
        # so `tool` does not match `tool_v2`. The lambda_name itself is regex-escaped.
        escaped_name = re.escape(TEST_LAMBDA_NAME)
        assert f"/\\/{escaped_name}\\[/" in kwargs["queryString"]
        # Boilerplate exclusions are present
        assert "filter @message not like /^START RequestId/" in kwargs["queryString"]
        assert "filter @message not like /^END RequestId/" in kwargs["queryString"]
        assert "filter @message not like /^REPORT RequestId/" in kwargs["queryString"]
        assert "filter @message not like /^INIT_START Runtime Version/" in kwargs["queryString"]

    @pytest.mark.asyncio
    async def test_get_function_logs_includes_pattern_when_provided(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.get_query_results.return_value = {"status": "Complete", "results": []}

        await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern="ERROR",
        )

        query_string = mock_logs_client.start_query.call_args.kwargs["queryString"]
        assert "filter @message like /ERROR/" in query_string

    @pytest.mark.asyncio
    async def test_get_function_logs_passes_seconds_to_insights(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        """Logs Insights expects start/end as epoch seconds, unlike FilterLogEvents (ms)."""
        mock_logs_client.get_query_results.return_value = {"status": "Complete", "results": []}

        await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
        )

        kwargs = mock_logs_client.start_query.call_args.kwargs
        assert kwargs["startTime"] == TEST_START_TIME
        assert kwargs["endTime"] == TEST_END_TIME

    @pytest.mark.asyncio
    async def test_get_function_logs_emits_cursor_when_page_full(
        self, logs_client: AWSLogsClient, mock_logs_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full page (events == page_limit) emits an opaque next_token anchored past the last timestamp."""
        monkeypatch.setattr(settings, "AWS_LOGS_INSIGHTS_PAGE_LIMIT", TEST_PAGE_LIMIT)
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                _insights_row("2021-01-01 00:00:01.000", "a"),
                _insights_row("2021-01-01 00:00:02.000", "b"),
            ],
        }

        events, next_token = await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
        )

        assert len(events) == TEST_PAGE_LIMIT
        assert next_token is not None
        decoded = AWSLogsClient._decode_cursor(next_token)
        assert decoded == events[-1]["timestamp"] + 1

    @pytest.mark.asyncio
    async def test_get_function_logs_resumes_from_cursor(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        """Calling with a previously emitted cursor advances the start of the window."""
        cursor_ms = TEST_START_TIME_MS + 5_000
        cursor = AWSLogsClient._encode_cursor(cursor_ms)
        mock_logs_client.get_query_results.return_value = {"status": "Complete", "results": []}

        await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
            next_token=cursor,
        )

        kwargs = mock_logs_client.start_query.call_args.kwargs
        assert kwargs["startTime"] == cursor_ms // 1000

    @pytest.mark.asyncio
    async def test_get_function_logs_ignores_malformed_cursor(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.get_query_results.return_value = {"status": "Complete", "results": []}

        await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
            next_token="not-a-real-cursor",
        )

        # Falls back to the original start_time
        kwargs = mock_logs_client.start_query.call_args.kwargs
        assert kwargs["startTime"] == TEST_START_TIME

    @pytest.mark.asyncio
    async def test_get_function_logs_short_circuits_when_window_invalid(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        """A cursor past the end_time should not query AWS at all."""
        cursor = AWSLogsClient._encode_cursor(TEST_END_TIME_MS + 1)

        events, next_token = await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
            next_token=cursor,
        )

        assert events == []
        assert next_token is None
        mock_logs_client.start_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_function_logs_polls_until_complete(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.get_query_results.side_effect = [
            {"status": "Running", "results": []},
            {"status": "Scheduled", "results": []},
            {
                "status": "Complete",
                "results": [_insights_row("2021-01-01 00:00:01.000", "ok")],
            },
        ]

        events, _ = await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
        )

        assert [e["message"] for e in events] == ["ok"]
        assert mock_logs_client.get_query_results.call_count == EXPECTED_POLL_CALLS_UNTIL_COMPLETE

    @pytest.mark.asyncio
    async def test_get_function_logs_raises_on_non_complete_terminal_status(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.get_query_results.return_value = {"status": "Failed", "results": []}

        with pytest.raises(RuntimeError) as exc_info:
            await logs_client.get_function_logs(
                log_group_arn=TEST_LOG_GROUP_ARN,
                lambda_name=TEST_LAMBDA_NAME,
                start_time=TEST_START_TIME_DATETIME,
                end_time=TEST_END_TIME_DATETIME,
                filter_pattern=None,
            )

        assert "Failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_function_logs_times_out_and_stops_query(
        self,
        logs_client: AWSLogsClient,
        mock_logs_client: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "AWS_LOGS_INSIGHTS_MAX_WAIT_SECONDS", 0)
        mock_logs_client.get_query_results.return_value = {"status": "Running", "results": []}

        with pytest.raises(TimeoutError):
            await logs_client.get_function_logs(
                log_group_arn=TEST_LOG_GROUP_ARN,
                lambda_name=TEST_LAMBDA_NAME,
                start_time=TEST_START_TIME_DATETIME,
                end_time=TEST_END_TIME_DATETIME,
                filter_pattern=None,
            )

        mock_logs_client.stop_query.assert_called_once_with(queryId=TEST_QUERY_ID)

    @pytest.mark.asyncio
    async def test_get_function_logs_propagates_start_query_errors(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.start_query.side_effect = RuntimeError("AWS Client Error")

        with pytest.raises(RuntimeError) as exc_info:
            await logs_client.get_function_logs(
                log_group_arn=TEST_LOG_GROUP_ARN,
                lambda_name=TEST_LAMBDA_NAME,
                start_time=TEST_START_TIME_DATETIME,
                end_time=TEST_END_TIME_DATETIME,
                filter_pattern=None,
            )

        assert "AWS Client Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_function_logs_requires_lambda_name(self, logs_client: AWSLogsClient) -> None:
        with pytest.raises(ValueError):
            await logs_client.get_function_logs(
                log_group_arn=TEST_LOG_GROUP_ARN,
                lambda_name="",
                start_time=TEST_START_TIME_DATETIME,
                end_time=TEST_END_TIME_DATETIME,
                filter_pattern=None,
            )

    @pytest.mark.asyncio
    async def test_get_function_logs_skips_rows_with_invalid_timestamp(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.get_query_results.return_value = {
            "status": "Complete",
            "results": [
                _insights_row("not-a-timestamp", "should be skipped"),
                _insights_row("2021-01-01 00:00:01.000", "ok"),
            ],
        }

        events, _ = await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=TEST_END_TIME_DATETIME,
            filter_pattern=None,
        )

        assert [e["message"] for e in events] == ["ok"]

    @pytest.mark.asyncio
    async def test_get_function_logs_defaults_end_time_to_now(
        self, logs_client: AWSLogsClient, mock_logs_client: Any
    ) -> None:
        mock_logs_client.get_query_results.return_value = {"status": "Complete", "results": []}

        await logs_client.get_function_logs(
            log_group_arn=TEST_LOG_GROUP_ARN,
            lambda_name=TEST_LAMBDA_NAME,
            start_time=TEST_START_TIME_DATETIME,
            end_time=None,
            filter_pattern=None,
        )

        kwargs = mock_logs_client.start_query.call_args.kwargs
        assert kwargs["endTime"] >= TEST_START_TIME

    def test_cursor_round_trip(self) -> None:
        token = AWSLogsClient._encode_cursor(SAMPLE_CURSOR_TIMESTAMP_MS)
        assert AWSLogsClient._decode_cursor(token) == SAMPLE_CURSOR_TIMESTAMP_MS

    def test_decode_cursor_invalid_returns_none(self) -> None:
        assert AWSLogsClient._decode_cursor("abc!!!") is None
