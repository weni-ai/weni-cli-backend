import asyncio
import logging
from typing import Any

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSLogsClient:
    def __init__(self) -> None:
        self.client = boto3.client("logs", region_name=settings.AWS_REGION)

    async def get_function_logs(self, function_name: str, start_time: float, end_time: float) -> list[dict[str, Any]]:
        """
        Get the logs for a lambda function

        Args:
            function_arn: The ARN of the lambda function to get the logs for
            start_time: The start time of the logs to get
            end_time: The end time of the logs to get

        Returns:
            A list of logs sorted by timestamp
        """

        # Add retries for log group creation
        max_retries = 5
        retry_delay = 5  # seconds

        log_group_name = f"/aws/lambda/{function_name}"

        for _ in range(max_retries):
            try:
                response = self.client.filter_log_events(
                    logGroupName=log_group_name, startTime=int(start_time), endTime=int(end_time), limit=50
                )

                return sorted(response["events"], key=lambda x: x["timestamp"])

            except self.client.exceptions.ResourceNotFoundException:
                logger.info(f"Log group {log_group_name} not found, waiting {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                continue

            except Exception as e:
                logger.error(f"Error getting logs for function {function_name}: {str(e)}")
                raise e

        raise Exception(f"Failed to get logs for function {function_name} after {max_retries} attempts")
