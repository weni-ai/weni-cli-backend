import base64
import datetime
import json
from io import BytesIO
from typing import Any

import pytest
from botocore.exceptions import ClientError
from pytest_mock import MockerFixture

from app.clients.aws.lambda_client import AWSLambdaClient, LambdaFunction
from app.core.config import settings

# Constants
# HTTP status codes
HTTP_STATUS_OK = 200

# Mock time values as valid timestamps in datetime format
MOCK_START_TIME = datetime.datetime(2021, 1, 1, 0, 0, 0)
MOCK_END_TIME = datetime.datetime(2021, 1, 1, 0, 0, 1)

# Call count expectations
EXPECTED_CALL_COUNT_TWO = 2
EXPECTED_CALL_COUNT_ONE = 1
MAX_ATTEMPTS = 10

MOCK_REQUEST_ID = "1234567890"

# AWS resource values
TEST_FUNCTION_NAME = "test-function"
TEST_FUNCTION_ARN = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
TEST_HANDLER = "lambda_handler.handler"


# Create a mock exception class for testing
class MockResourceNotFoundError(Exception):
    pass


@pytest.fixture
def mock_lambda_client(mocker: MockerFixture) -> Any:
    """
    Fixture that mocks the AWS Lambda client directly
    """
    # Create a mock for the boto3 lambda client
    mock_client = mocker.MagicMock()

    # Setup the exceptions attribute with proper exception classes
    mock_exceptions = mocker.MagicMock()
    mock_exceptions.ResourceNotFoundException = MockResourceNotFoundError
    mock_client.exceptions = mock_exceptions

    # Patch the boto3.client call to return our mock
    mocker.patch("boto3.client", return_value=mock_client)

    return mock_client


@pytest.fixture
def lambda_client(mock_lambda_client: Any) -> AWSLambdaClient:
    """
    Fixture that returns an instance of the AWSLambdaClient with mocked boto3
    """
    return AWSLambdaClient()


@pytest.fixture
def test_event() -> dict[str, str]:
    """
    Fixture that returns a sample test event for Lambda invocation
    """
    return {"key": "value"}


@pytest.fixture
def test_code() -> BytesIO:
    """
    Fixture that returns sample Lambda function code
    """
    return BytesIO(b"def handler(event, context): return {'statusCode': 200}")


@pytest.fixture
def mock_sleep(mocker: MockerFixture) -> Any:
    """
    Fixture that mocks asyncio.sleep to avoid waiting in tests
    """
    return mocker.patch("asyncio.sleep")


class TestAWSLambdaClient:
    def test_init(self, mocker: MockerFixture) -> None:
        """Test initialization of the AWS Lambda client"""
        # Setup
        mock_client = mocker.MagicMock()
        mocker.patch("boto3.client", return_value=mock_client)

        # Execute
        client = AWSLambdaClient()

        # Assert
        assert client.client == mock_client

    def test_create_function(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, test_code: BytesIO
    ) -> None:
        """Test create_function method"""
        # Setup
        mock_lambda_client.create_function.return_value = {"FunctionArn": TEST_FUNCTION_ARN}

        # Execute
        result = lambda_client.create_function(
            function_name=TEST_FUNCTION_NAME,
            handler=TEST_HANDLER,
            code=test_code,
            description="Test function description",
        )

        # Assert
        mock_lambda_client.create_function.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME,
            Runtime="python3.12",
            Timeout=180,
            Role=settings.AGENT_RESOURCE_ROLE_ARN,
            Code={"ZipFile": test_code.getvalue()},
            Handler=TEST_HANDLER,
            Description="Test function description",
            LoggingConfig={"LogGroup": settings.AGENT_LOG_GROUP},
        )

        assert isinstance(result, LambdaFunction)
        assert result.arn == TEST_FUNCTION_ARN
        assert result.name == TEST_FUNCTION_NAME

    def test_create_function_with_environment_variables(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, test_code: BytesIO
    ) -> None:
        """Test create_function method with environment variables"""
        # Setup
        mock_lambda_client.create_function.return_value = {"FunctionArn": TEST_FUNCTION_ARN}
        environment = {"ENV_VAR1": "value1", "ENV_VAR2": "value2"}

        # Execute
        result = lambda_client.create_function(
            function_name=TEST_FUNCTION_NAME,
            handler=TEST_HANDLER,
            code=test_code,
            description="Test function description",
            environment=environment,
        )

        # Assert
        mock_lambda_client.create_function.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME,
            Runtime="python3.12",
            Timeout=180,
            Role=settings.AGENT_RESOURCE_ROLE_ARN,
            Code={"ZipFile": test_code.getvalue()},
            Handler=TEST_HANDLER,
            Description="Test function description",
            LoggingConfig={"LogGroup": settings.AGENT_LOG_GROUP},
            Environment={"Variables": environment},
        )

        assert isinstance(result, LambdaFunction)
        assert result.arn == TEST_FUNCTION_ARN
        assert result.name == TEST_FUNCTION_NAME

    def test_update_function_configuration(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any
    ) -> None:
        """Test update_function_configuration method"""
        # Setup
        environment = {"ENV_VAR1": "new_value1", "ENV_VAR2": "new_value2"}
        mock_response = {
            "FunctionName": TEST_FUNCTION_NAME,
            "FunctionArn": TEST_FUNCTION_ARN,
            "Environment": {"Variables": environment},
        }
        mock_lambda_client.update_function_configuration.return_value = mock_response

        # Execute
        result = lambda_client.update_function_configuration(
            function_name=TEST_FUNCTION_NAME,
            environment=environment,
        )

        # Assert
        mock_lambda_client.update_function_configuration.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME,
            Environment={"Variables": environment},
        )
        assert result == mock_response

    def test_update_function_configuration_with_additional_params(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any
    ) -> None:
        """Test update_function_configuration method with additional parameters"""
        # Setup
        environment = {"ENV_VAR1": "new_value1"}
        mock_response = {
            "FunctionName": TEST_FUNCTION_NAME,
            "FunctionArn": TEST_FUNCTION_ARN,
            "Environment": {"Variables": environment},
            "Timeout": 300,
        }
        mock_lambda_client.update_function_configuration.return_value = mock_response

        # Execute
        result = lambda_client.update_function_configuration(
            function_name=TEST_FUNCTION_NAME,
            environment=environment,
            Timeout=300,
        )

        # Assert
        mock_lambda_client.update_function_configuration.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME,
            Environment={"Variables": environment},
            Timeout=300,
        )
        assert result == mock_response

    def test_get_function_configuration(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any
    ) -> None:
        """Test get_function_configuration method"""
        # Setup
        mock_response = {
            "FunctionName": TEST_FUNCTION_NAME,
            "FunctionArn": TEST_FUNCTION_ARN,
            "Environment": {"Variables": {"ENV_VAR1": "value1"}},
        }
        mock_lambda_client.get_function_configuration.return_value = mock_response

        # Execute
        result = lambda_client.get_function_configuration(function_name=TEST_FUNCTION_NAME)

        # Assert
        mock_lambda_client.get_function_configuration.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME
        )
        assert result == mock_response

    def test_delete_function(self, lambda_client: AWSLambdaClient, mock_lambda_client: Any) -> None:
        """Test delete_function method"""
        # Execute
        lambda_client.delete_function(function_name=TEST_FUNCTION_NAME)

        # Assert
        mock_lambda_client.delete_function.assert_called_once_with(FunctionName=TEST_FUNCTION_NAME)

    def test_delete_function_error(self, lambda_client: AWSLambdaClient, mock_lambda_client: Any) -> None:
        """Test delete_function method when an error occurs"""
        # Setup
        mock_lambda_client.delete_function.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Function not found"}}, "DeleteFunction"
        )

        # Execute and Assert
        with pytest.raises(ClientError):
            lambda_client.delete_function(function_name=TEST_FUNCTION_NAME)

    def test_invoke_function(
        self,
        lambda_client: AWSLambdaClient,
        mock_lambda_client: Any,
        mocker: MockerFixture,
        test_event: dict[str, str],
    ) -> None:
        """Test invoke_function method"""
        # Setup
        mock_payload = mocker.MagicMock()
        mock_payload.read.return_value = json.dumps({"result": "success"})

        mock_lambda_client.invoke.return_value = {
            "StatusCode": HTTP_STATUS_OK,
            "Payload": mock_payload,
            "ResponseMetadata": {"RequestId": MOCK_REQUEST_ID},
            "LogResult": base64.b64encode(b"test log"),
        }

        # Directly patch datetime.now to return our fixed times
        datetime_mock = mocker.patch("app.clients.aws.lambda_client.datetime")
        # First call for start_time
        datetime_mock.datetime.now.side_effect = [MOCK_START_TIME, MOCK_END_TIME]

        # Execute
        result, start_time, end_time = lambda_client.invoke_function(function_arn=TEST_FUNCTION_ARN, event=test_event)

        # Assert
        mock_lambda_client.invoke.assert_called_once_with(
            FunctionName=TEST_FUNCTION_ARN,
            InvocationType="RequestResponse",
            Payload=json.dumps(test_event),
            LogType="Tail",
        )

        assert result["status_code"] == HTTP_STATUS_OK
        assert result["response"] == {"result": "success"}
        assert result["logs"] == "test log"
        assert start_time == MOCK_START_TIME.timestamp()
        assert end_time == MOCK_END_TIME.timestamp()

    # Group related tests under a class for better organization
    class TestWaitForFunctionActive:
        @pytest.mark.asyncio
        async def test_immediate_success(
            self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, mock_sleep: Any
        ) -> None:
            """Test when function becomes active immediately"""
            # Setup
            mock_lambda_client.get_function.return_value = {"Configuration": {"State": "Active"}}

            # Execute
            result = await lambda_client.wait_for_function_active(function_arn=TEST_FUNCTION_ARN)

            # Assert
            mock_lambda_client.get_function.assert_called_once_with(FunctionName=TEST_FUNCTION_ARN)
            mock_sleep.assert_not_called()
            assert result is True

        @pytest.mark.asyncio
        async def test_pending_then_active(
            self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, mock_sleep: Any
        ) -> None:
            """Test when function transitions from pending to active"""
            # Setup
            mock_lambda_client.get_function.side_effect = [
                {"Configuration": {"State": "Pending"}},
                {"Configuration": {"State": "Active"}},
            ]

            # Execute
            result = await lambda_client.wait_for_function_active(function_arn=TEST_FUNCTION_ARN)

            # Assert
            assert mock_lambda_client.get_function.call_count == EXPECTED_CALL_COUNT_TWO
            assert mock_sleep.call_count == EXPECTED_CALL_COUNT_ONE
            assert result is True

        @pytest.mark.asyncio
        async def test_deployment_failed(
            self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, mock_sleep: Any
        ) -> None:
            """Test when function deployment fails"""
            # Setup
            error_reason = "Error deploying function"
            mock_lambda_client.get_function.return_value = {
                "Configuration": {"State": "Failed", "StateReason": error_reason}
            }

            # Execute and Assert
            with pytest.raises(Exception) as excinfo:
                await lambda_client.wait_for_function_active(function_arn=TEST_FUNCTION_ARN)

            # Verify the exception message matches expected
            assert f"Function deployment failed: {error_reason}" in str(excinfo.value)

        @pytest.mark.asyncio
        async def test_resource_not_found_then_active(
            self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, mock_sleep: Any
        ) -> None:
            """Test when resource is not found initially but then found"""
            # Setup
            mock_lambda_client.get_function.side_effect = [
                MockResourceNotFoundError("Function not found"),  # First call raises the exception
                {"Configuration": {"State": "Active"}},  # Second call returns success
            ]

            # Execute
            result = await lambda_client.wait_for_function_active(function_arn=TEST_FUNCTION_ARN)

            # Assert
            assert mock_lambda_client.get_function.call_count == EXPECTED_CALL_COUNT_TWO
            assert mock_sleep.call_count == EXPECTED_CALL_COUNT_ONE
            assert result is True

        @pytest.mark.asyncio
        async def test_timeout(self, lambda_client: AWSLambdaClient, mock_lambda_client: Any, mock_sleep: Any) -> None:
            """Test when max attempts are reached"""
            # Setup
            mock_responses = [{"Configuration": {"State": "Pending"}}] * MAX_ATTEMPTS
            mock_lambda_client.get_function.side_effect = mock_responses

            # Execute
            result = await lambda_client.wait_for_function_active(
                function_arn=TEST_FUNCTION_ARN, max_attempts=MAX_ATTEMPTS
            )

            # Assert
            assert mock_lambda_client.get_function.call_count == MAX_ATTEMPTS
            assert mock_sleep.call_count == MAX_ATTEMPTS
            assert result is False

    def test_build_default_environment_variables(self) -> None:
        """Test build_default_environment_variables static method"""
        # Execute
        result = AWSLambdaClient.build_default_environment_variables(
            project_uuid="test-project-uuid",
            agent_key="test-agent",
            tool_key="test-tool",
            custom_vars={"CUSTOM_VAR": "custom_value"},
        )

        # Assert
        expected = {
            "PROJECT_UUID": "test-project-uuid",
            "WENI_ENVIRONMENT": settings.ENVIRONMENT,
            "LOG_LEVEL": "INFO",
            "AGENT_KEY": "test-agent",
            "TOOL_KEY": "test-tool",
            "CUSTOM_VAR": "custom_value",
        }
        assert result == expected

    def test_build_default_environment_variables_minimal(self) -> None:
        """Test build_default_environment_variables with minimal parameters"""
        # Execute
        result = AWSLambdaClient.build_default_environment_variables(
            project_uuid="test-project-uuid"
        )

        # Assert
        expected = {
            "PROJECT_UUID": "test-project-uuid",
            "WENI_ENVIRONMENT": settings.ENVIRONMENT,
            "LOG_LEVEL": "INFO",
        }
        assert result == expected

    def test_update_environment_variables_merge(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any
    ) -> None:
        """Test update_environment_variables with merge option"""
        # Setup
        existing_env = {"EXISTING_VAR": "existing_value", "OVERRIDE_VAR": "old_value"}
        new_env = {"NEW_VAR": "new_value", "OVERRIDE_VAR": "new_value"}
        
        mock_lambda_client.get_function_configuration.return_value = {
            "Environment": {"Variables": existing_env}
        }
        mock_lambda_client.update_function_configuration.return_value = {"Updated": True}

        # Execute
        result = lambda_client.update_environment_variables(
            function_name=TEST_FUNCTION_NAME,
            new_environment_vars=new_env,
            merge_with_existing=True,
        )

        # Assert
        expected_merged = {
            "EXISTING_VAR": "existing_value",
            "NEW_VAR": "new_value",
            "OVERRIDE_VAR": "new_value",  # Should be overridden
        }
        mock_lambda_client.update_function_configuration.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME,
            Environment={"Variables": expected_merged},
        )
        assert result == {"Updated": True}

    def test_update_environment_variables_replace(
        self, lambda_client: AWSLambdaClient, mock_lambda_client: Any
    ) -> None:
        """Test update_environment_variables with replace option"""
        # Setup
        new_env = {"NEW_VAR": "new_value"}
        mock_lambda_client.update_function_configuration.return_value = {"Updated": True}

        # Execute
        result = lambda_client.update_environment_variables(
            function_name=TEST_FUNCTION_NAME,
            new_environment_vars=new_env,
            merge_with_existing=False,
        )

        # Assert
        mock_lambda_client.update_function_configuration.assert_called_once_with(
            FunctionName=TEST_FUNCTION_NAME,
            Environment={"Variables": new_env},
        )
        assert result == {"Updated": True}
        # Should not call get_function_configuration when not merging
        mock_lambda_client.get_function_configuration.assert_not_called()
