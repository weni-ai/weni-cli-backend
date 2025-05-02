import logging
from datetime import UTC, datetime
from typing import Any

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSLogsClient:
    def __init__(self) -> None:
        self.client = boto3.client("logs", region_name=settings.AWS_REGION)

    # Convert a datetime to milliseconds
    # preserving the timezone or using UTC+0 if no timezone is present
    def __format_time(self, time: datetime) -> int:
        if time.tzinfo:
            return int(time.astimezone(UTC).timestamp() * 1000)
        else:
            return int(time.replace(tzinfo=UTC).timestamp() * 1000)

    async def get_function_logs(
        self,
        log_group_arn: str,
        start_time: datetime | None,
        end_time: datetime | None,
        filter_pattern: str | None,
        next_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Get the logs for a lambda function

        Args:
            log_group_name: The name of the log group to get the logs for
            start_time: The start time of the logs to get
            end_time: The end time of the logs to get

        Returns:
            A list of logs sorted by timestamp
            A next token if there are more logs to get
        """

        # Convert timestamps to milliseconds
        start_time_ms = self.__format_time(start_time) if start_time else 0
        end_time_ms = (
            self.__format_time(end_time)
            if end_time
            else self.__format_time(datetime.now(UTC))
        )

        try:
            current_token = next_token  # Use a different variable for the loop
            while True:
                kwargs = {
                    "logGroupIdentifier": log_group_arn,
                    "startTime": start_time_ms,
                    "endTime": end_time_ms,
                    "limit": 1000,
                    "filterPattern": f'"{filter_pattern}" -"START RequestId" -"END RequestId" -"REPORT RequestId" -"INIT_START Runtime Version"',  # noqa: E501
                }
                if current_token:
                    kwargs["nextToken"] = current_token

                response = self.client.filter_log_events(**kwargs)
                response_token = response.get("nextToken")
                events = response.get("events", [])

                # If we found events after filtering, return them and the next token
                if events:
                    return sorted(events, key=lambda x: x["timestamp"]), response_token

                # If no events after filtering, but there are more pages, continue to the next page
                if not events and response_token:
                    current_token = response_token
                    continue

                # If no events after filtering and no more pages, return empty
                if not events and not response_token:
                    return [], None

        except Exception as e:
            logger.error(f"Error getting logs for log group {log_group_arn}: {str(e)}")
            raise e
