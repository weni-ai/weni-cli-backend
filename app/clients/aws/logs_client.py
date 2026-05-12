import asyncio
import base64
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


# Insights query statuses that indicate the query has finished (with or without success)
_INSIGHTS_TERMINAL_STATUSES: frozenset[str] = frozenset({"Complete", "Failed", "Cancelled", "Timeout"})

# Boilerplate emitted by the Lambda runtime that should never reach the user.
# Each item is matched with `filter @message not like /^.../` in Logs Insights QL.
_BOILERPLATE_PREFIXES: tuple[str, ...] = (
    "START RequestId",
    "END RequestId",
    "REPORT RequestId",
    "INIT_START Runtime Version",
)


class AWSLogsClient:
    """Client for reading Lambda logs through CloudWatch Logs Insights.

    The agents Lambdas now write to a single centralized log group (referenced
    by `AWS_LAMBDA_LOG_GROUP` on nexus-ai). To return only the events of a
    specific tool, queries filter on `@logStream`, which the AWS Lambda runtime
    automatically names as `YYYY/MM/DD/<lambda_name>[<version>][<exec_env_id>]`
    when a custom log group is used. This client is also compatible with legacy
    per-tool log groups (Standard class) that haven't been migrated yet, since
    the Insights query works on both classes.
    """

    def __init__(self) -> None:
        self.client = boto3.client("logs", region_name=settings.AWS_REGION_NAME)

    @staticmethod
    def _format_time_ms(time_value: datetime) -> int:
        """Convert a datetime to epoch milliseconds (UTC)."""
        if time_value.tzinfo:
            return int(time_value.astimezone(UTC).timestamp() * 1000)
        return int(time_value.replace(tzinfo=UTC).timestamp() * 1000)

    @staticmethod
    def _encode_cursor(start_time_ms: int) -> str:
        """Pack the next page anchor (epoch ms) into an opaque, URL-safe token."""
        payload = json.dumps({"start_ms": start_time_ms}, separators=(",", ":"))
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")

    @staticmethod
    def _decode_cursor(cursor: str) -> int | None:
        """Unpack a previously emitted cursor. Returns None if invalid."""
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
            payload = json.loads(decoded)
            value = payload.get("start_ms")
            return int(value) if value is not None else None
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Discarding malformed Logs Insights cursor")
            return None

    @staticmethod
    def _build_query(lambda_name: str, filter_pattern: str | None, limit: int) -> str:
        """Compose the Logs Insights QL string for a single page.

        The mandatory `filter @logStream like /.../` ensures multi-tenant
        isolation when the log group is shared across many Lambdas. It
        matches the `<lambda_name>[` segment so `tool` does not accidentally
        match `tool_v2`.
        """
        escaped_lambda = re.escape(lambda_name)
        parts: list[str] = [
            "fields @timestamp, @message, @logStream",
            f"filter @logStream like /\\/{escaped_lambda}\\[/",
        ]
        # The known runtime prefixes contain only letters, digits, spaces and
        # underscores, so they're safe to embed verbatim in the regex.
        for prefix in _BOILERPLATE_PREFIXES:
            parts.append(f"filter @message not like /^{prefix}/")
        if filter_pattern:
            parts.append(f"filter @message like /{re.escape(filter_pattern)}/")
        parts.append("sort @timestamp asc")
        parts.append(f"limit {limit}")
        return " | ".join(parts)

    @staticmethod
    def _row_to_field_map(row: list[dict[str, str]]) -> dict[str, str]:
        return {field.get("field", ""): field.get("value", "") for field in row}

    @classmethod
    def _row_to_event(cls, row: list[dict[str, str]]) -> dict[str, Any] | None:
        """Convert one Logs Insights row to the legacy event shape.

        Mirrors what `filter_log_events` used to return so the API contract
        with the CLI (`{timestamp: int_ms, message: str, logStreamName: str}`)
        is preserved.
        """
        fields = cls._row_to_field_map(row)
        timestamp_str = fields.get("@timestamp")
        message = fields.get("@message")
        if timestamp_str is None or message is None:
            return None

        # Insights returns timestamps as `YYYY-MM-DD HH:MM:SS.mmm` (UTC, naive)
        try:
            parsed = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                parsed = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.warning("Skipping Insights row with unexpected timestamp format")
                return None
        timestamp_ms = int(parsed.replace(tzinfo=UTC).timestamp() * 1000)

        return {
            "timestamp": timestamp_ms,
            "message": message,
            "logStreamName": fields.get("@logStream", ""),
        }

    async def _wait_for_query(self, query_id: str) -> dict[str, Any]:
        """Poll Logs Insights until the query reaches a terminal status."""
        deadline = time.monotonic() + settings.AWS_LOGS_INSIGHTS_MAX_WAIT_SECONDS
        while True:
            await asyncio.sleep(settings.AWS_LOGS_INSIGHTS_POLL_INTERVAL_SECONDS)
            response: dict[str, Any] = self.client.get_query_results(queryId=query_id)
            status = response.get("status", "")
            if status in _INSIGHTS_TERMINAL_STATUSES:
                return response
            if time.monotonic() > deadline:
                try:
                    self.client.stop_query(queryId=query_id)
                except Exception:  # noqa: BLE001 — best effort
                    logger.warning("Failed to stop Logs Insights query after timeout", exc_info=True)
                raise TimeoutError(
                    f"Logs Insights query {query_id} timed out after "
                    f"{settings.AWS_LOGS_INSIGHTS_MAX_WAIT_SECONDS}s"
                )

    async def get_function_logs(  # noqa: PLR0913
        self,
        log_group_arn: str,
        lambda_name: str,
        start_time: datetime | None,
        end_time: datetime | None,
        filter_pattern: str | None,
        next_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch a page of logs for a single Lambda from a (possibly shared) log group.

        Args:
            log_group_arn: ARN of the log group to query (centralized or legacy).
            lambda_name: Function name used to filter `@logStream` and isolate
                events of a single Lambda inside the shared log group.
            start_time: Inclusive lower bound for the log window.
            end_time: Inclusive upper bound for the log window. Defaults to now.
            filter_pattern: Optional case-sensitive substring to match in `@message`.
            next_token: Opaque cursor returned by a previous call.

        Returns:
            Tuple of (events sorted ascending by timestamp, next_token or None).
        """
        if not lambda_name:
            raise ValueError("lambda_name is required for log isolation")

        end_time_ms = self._format_time_ms(end_time) if end_time else self._format_time_ms(datetime.now(UTC))
        if next_token:
            cursor_ms = self._decode_cursor(next_token)
            if cursor_ms is not None:
                start_time_ms = cursor_ms
            else:
                start_time_ms = self._format_time_ms(start_time) if start_time else 0
        else:
            start_time_ms = self._format_time_ms(start_time) if start_time else 0

        if start_time_ms >= end_time_ms:
            return [], None

        page_limit = settings.AWS_LOGS_INSIGHTS_PAGE_LIMIT
        query_string = self._build_query(lambda_name, filter_pattern, page_limit)

        try:
            start_response = self.client.start_query(
                logGroupIdentifiers=[log_group_arn],
                startTime=start_time_ms // 1000,
                endTime=max(end_time_ms // 1000, 1),
                queryString=query_string,
                limit=page_limit,
            )
        except Exception as e:
            logger.error(f"Error starting Logs Insights query for {log_group_arn}: {str(e)}")
            raise

        query_id = start_response["queryId"]

        try:
            result = await self._wait_for_query(query_id)
        except Exception as e:
            logger.error(f"Error reading logs from {log_group_arn}: {str(e)}")
            raise

        status = result.get("status", "")
        if status != "Complete":
            raise RuntimeError(f"Logs Insights query {query_id} ended with status {status}")

        rows = result.get("results", []) or []
        events: list[dict[str, Any]] = []
        for row in rows:
            event = self._row_to_event(row)
            if event is not None:
                events.append(event)

        events.sort(key=lambda evt: evt["timestamp"])

        # Pagination: if we filled the page, advance the start cursor past the
        # last event returned. We use `+1ms` to avoid re-emitting the same
        # event; this can drop sibling events sharing the exact same
        # millisecond, which is acceptable for Lambda log volumes.
        next_cursor: str | None = None
        if len(events) >= page_limit:
            next_cursor = self._encode_cursor(events[-1]["timestamp"] + 1)

        return events, next_cursor
