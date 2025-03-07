import asyncio
import json
import logging
import time
from io import BytesIO
from typing import Any

import boto3
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class LambdaFunction(BaseModel):
    function_arn: str | None
    function_name: str | None


class AWSLambdaClient:
    def __init__(self) -> None:
        self.client = boto3.client("lambda", region_name=settings.AWS_REGION)

    def create_function(self, function_name: str, handler: str, code: BytesIO, description: str) -> LambdaFunction:
        """
        Creates a lambda function

        Args:
            function_name: The name of the lambda function to create
            handler: The handler of the lambda function
            code: The code of the lambda function
            description: The description of the lambda function

        Returns:
            The ARN of the created lambda function
        """
        lambda_role = settings.AGENT_RESOURCE_ROLE_ARN
        logger.info(f"Creating lambda function {function_name}.")
        logger.debug(f"Function: {function_name}, Description: {description}")

        lambda_function = self.client.create_function(
            FunctionName=function_name,
            Runtime="python3.12",
            Timeout=180,
            Role=lambda_role,
            Code={"ZipFile": code.getvalue()},
            Handler=handler,
            Description=description,
        )

        response = LambdaFunction(function_arn=lambda_function.get("FunctionArn"), function_name=function_name)
        logger.info(f"Lambda function {function_name} created. ARN: {response.function_arn}")
        return response

    def delete_function(self, function_name: str) -> None:
        """
        Deletes a lambda function

        Args:
            function_name: The name of the lambda function to delete
        """

        logger.info(f"Deleting lambda function {function_name}.")
        self.client.delete_function(FunctionName=function_name)
        logger.info(f"Lambda function {function_name} deleted.")

    def invoke_function(self, function_arn: str, event: dict[str, Any]) -> tuple[Any, float, float]:
        """
        Invokes a lambda function and returns the response and the start time

        Args:
            function_arn: The ARN of the lambda function to invoke
            event: The event to pass to the lambda function

        Returns:
            A tuple containing the response and the start time
        """

        logger.info(f"Invoking lambda function {function_arn}.")
        start_time = time.time()

        invoke_response = self.client.invoke(
            FunctionName=function_arn, InvocationType="RequestResponse", Payload=json.dumps(event)
        )

        end_time = time.time()
        logger.info(f"Lambda function {function_arn} invoked in {end_time - start_time} seconds.")

        result = {
            "status_code": invoke_response["StatusCode"],
            "response": json.loads(invoke_response["Payload"].read()),
        }
        return result, start_time, end_time

    async def wait_for_function_active(self, function_arn: str, max_attempts: int = 10) -> bool:
        """
        Wait for Lambda function to become active

        Args:
            lambda_client: Boto3 Lambda client
            function_name: Name of the Lambda function
            max_attempts: Maximum number of attempts to check status

        Returns:
            True if function is active

        Raises:
            Exception: If function deployment failed or did not become active
        """
        max_attempts = 10
        retry_delay = 3

        for attempt in range(max_attempts):
            logger.info(f"Waiting for function {function_arn} to become active - attempt {attempt + 1}")
            try:
                response = self.client.get_function(FunctionName=function_arn)
                state = response["Configuration"]["State"]

                if state == "Active":
                    return True

                if state == "Failed":
                    raise Exception(
                        f"Function deployment failed: {response['Configuration'].get('StateReason', 'Unknown reason')}"
                    )

                await asyncio.sleep(retry_delay)

            except self.client.exceptions.ResourceNotFoundException:
                await asyncio.sleep(retry_delay)
                continue

        return False
