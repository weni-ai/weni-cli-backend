import asyncio
import base64
import datetime
import json
import logging
from io import BytesIO
from typing import Any

import boto3
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class LambdaFunction(BaseModel):
    arn: str | None
    name: str | None
    log_group: str | None


class AWSLambdaClient:
    def __init__(self) -> None:
        self.client = boto3.client("lambda", region_name=settings.AWS_REGION_NAME)

    @staticmethod
    def build_default_environment_variables(
        project_uuid: str,
        agent_key: str | None = None,
        tool_key: str | None = None,
        custom_vars: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Build default environment variables for Lambda functions.
        
        Args:
            project_uuid: The project UUID
            agent_key: The agent key (optional)
            tool_key: The tool key (optional)
            custom_vars: Custom environment variables to add
            
        Returns:
            Dictionary of environment variables
        """
        env_vars = {
            "PROJECT_UUID": project_uuid,
            "WENI_ENVIRONMENT": settings.ENVIRONMENT,
            "LOG_LEVEL": "INFO",
        }
        
        if agent_key:
            env_vars["AGENT_KEY"] = agent_key
            
        if tool_key:
            env_vars["TOOL_KEY"] = tool_key
            
        if custom_vars:
            env_vars.update(custom_vars)
            
        return env_vars

    def get_function(self, function_arn: str) -> LambdaFunction:
        """
        Get a lambda function
        """
        response = self.client.get_function(FunctionName=function_arn)

        config = response.get("Configuration", {})

        return LambdaFunction(
            arn=config.get("FunctionArn"),
            name=config.get("FunctionName"),
            log_group=config.get("LoggingConfig", {}).get("LogGroup", ""),
        )

    def create_function(
        self, function_name: str, handler: str, code: BytesIO, description: str, environment: dict[str, str] | None = None
    ) -> LambdaFunction:
        """
        Creates a lambda function

        Args:
            function_name: The name of the lambda function to create
            handler: The handler of the lambda function
            code: The code of the lambda function
            description: The description of the lambda function
            environment: Environment variables to set in the function

        Returns:
            The ARN of the created lambda function
        """
        lambda_role = settings.AGENT_RESOURCE_ROLE_ARN
        log_group = settings.AGENT_LOG_GROUP
        logger.info(f"Creating lambda function {function_name}.")
        logger.debug(f"Function: {function_name}, Description: {description}")

        create_params = {
            "FunctionName": function_name,
            "Runtime": "python3.12",
            "Timeout": 180,
            "Role": lambda_role,
            "Code": {"ZipFile": code.getvalue()},
            "Handler": handler,
            "Description": description,
            "LoggingConfig": {
                "LogGroup": log_group,
            },
        }

        # Add environment variables if provided
        if environment:
            create_params["Environment"] = {"Variables": environment}
            logger.debug(f"Setting environment variables: {list(environment.keys())}")

        lambda_function = self.client.create_function(**create_params)

        response = LambdaFunction(
            arn=lambda_function.get("FunctionArn"),
            name=function_name,
            log_group=log_group,
        )
        logger.info(f"Lambda function {function_name} created. ARN: {response.arn}")
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

    def invoke_function(
        self, function_arn: str, event: dict[str, Any]
    ) -> tuple[Any, float, float]:
        """
        Invokes a lambda function and returns the response and the start time

        Args:
            function_arn: The ARN of the lambda function to invoke
            event: The event to pass to the lambda function

        Returns:
            A tuple containing the response and the start time
        """

        logger.info(f"Invoking lambda function {function_arn}.")
        start_time = datetime.datetime.now().timestamp()

        invoke_response = self.client.invoke(
            FunctionName=function_arn,
            InvocationType="RequestResponse",
            Payload=json.dumps(event),
            LogType="Tail",
        )

        end_time = datetime.datetime.now().timestamp()
        logger.info(
            f"Lambda function {function_arn} invoked in {end_time - start_time} seconds."
        )

        # Decode base64 encoded logs
        logs = base64.b64decode(invoke_response["LogResult"]).decode("utf-8")

        request_id = invoke_response["ResponseMetadata"]["RequestId"]

        # Remove lines that contain RequestId
        logs = "\n".join([line for line in logs.split("\n") if request_id not in line])

        result = {
            "status_code": invoke_response["StatusCode"],
            "response": json.loads(invoke_response["Payload"].read()),
            "request_id": request_id,
            "logs": logs,
        }
        return result, start_time, end_time

    async def wait_for_function_active(
        self, function_arn: str, max_attempts: int = 10
    ) -> bool:
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
            logger.info(
                f"Waiting for function {function_arn} to become active - attempt {attempt + 1}"
            )
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

    def update_function_configuration(
        self, function_name: str, environment: dict[str, str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Updates a lambda function configuration

        Args:
            function_name: The name of the lambda function to update
            environment: Environment variables to set in the function
            **kwargs: Additional configuration parameters

        Returns:
            The response from the update operation
        """
        logger.info(f"Updating lambda function configuration for {function_name}.")
        
        update_params = {"FunctionName": function_name}
        
        # Add environment variables if provided
        if environment is not None:
            update_params["Environment"] = {"Variables": environment}
            logger.debug(f"Updating environment variables: {list(environment.keys())}")
        
        # Add any additional parameters
        update_params.update(kwargs)

        response = self.client.update_function_configuration(**update_params)
        logger.info(f"Lambda function {function_name} configuration updated.")
        return response

    def get_function_configuration(self, function_name: str) -> dict[str, Any]:
        """
        Get the configuration of a lambda function

        Args:
            function_name: The name of the lambda function

        Returns:
            The function configuration
        """
        logger.info(f"Getting lambda function configuration for {function_name}.")
        response = self.client.get_function_configuration(FunctionName=function_name)
        return response

    def update_environment_variables(
        self,
        function_name: str,
        new_environment_vars: dict[str, str],
        merge_with_existing: bool = True,
    ) -> dict[str, Any]:
        """
        Update environment variables for an existing Lambda function.
        
        Args:
            function_name: The name of the lambda function to update
            new_environment_vars: New environment variables to set
            merge_with_existing: Whether to merge with existing variables or replace them
            
        Returns:
            The response from the update operation
        """
        if merge_with_existing:
            # Get current configuration to merge with existing environment variables
            current_config = self.get_function_configuration(function_name)
            existing_env = current_config.get("Environment", {}).get("Variables", {})
            
            # Merge new variables with existing ones
            merged_env = {**existing_env, **new_environment_vars}
            logger.info(f"Merging environment variables for {function_name}. New keys: {list(new_environment_vars.keys())}")
            
            return self.update_function_configuration(
                function_name=function_name,
                environment=merged_env,
            )
        else:
            # Replace all environment variables
            logger.info(f"Replacing environment variables for {function_name}. New keys: {list(new_environment_vars.keys())}")
            
            return self.update_function_configuration(
                function_name=function_name,
                environment=new_environment_vars,
            )
