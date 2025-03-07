import asyncio
import logging
from typing import Any

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSLogsClient:
    def __init__(self) -> None:
        self.client = boto3.client("logs", region_name=settings.AWS_REGION)

    async def get_function_logs(
        self, function_name: str, request_id: str, start_time: float, end_time: float
    ) -> list[dict[str, Any]]:
        """
        Get the logs for a lambda function

        Args:
            function_name: The name of the lambda function to get the logs for
            request_id: The request id of the lambda function to get the logs for
            start_time: The start time of the logs to get
            end_time: The end time of the logs to get

        Returns:
            A list of logs sorted by timestamp
        """

        # Add retries for log group creation
        max_retries = 5
        retry_delay = 5  # seconds

        log_group_name = settings.AGENT_LOG_GROUP

        # Convert start_time to milliseconds reducing 1 minute
        start_time_ms = int(start_time) * 1000 - 60000
        # Convert end_time to milliseconds adding 1 minute
        end_time_ms = int(end_time) * 1000 + 60000

        for _ in range(max_retries):
            try:
                response = self.client.filter_log_events(
                    logGroupName=log_group_name,
                    startTime=start_time_ms,
                    endTime=end_time_ms,
                    filterPattern=f'"{request_id}"',
                )

                if not response["events"]:
                    logger.info(
                        f"Log group {log_group_name} empty, for request {request_id}, waiting {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                return sorted(response["events"], key=lambda x: x["timestamp"])

            except self.client.exceptions.ResourceNotFoundException:
                logger.info(
                    f"Log group {log_group_name} not found, for request {request_id}, waiting {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)
                continue

            except Exception as e:
                logger.error(f"Error getting logs for function {function_name}: {str(e)}")
                raise e

        return []
