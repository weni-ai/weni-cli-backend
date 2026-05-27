"""Run strategy for active agents (PreProcessor + Rules)."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.api.v1.models.requests import RunRequestModel
from app.clients.aws import AWSLambdaClient
from app.clients.aws.lambda_client import LambdaFunction
from app.core.config import settings
from app.core.response import CLIResponse, send_response
from app.services.agents.active.models import ActiveAgentResourceModel, Resource, RuleResource
from app.services.agents.active.processor import ActiveAgentProcessor
from app.services.jwt_generator import JWT_PROJECT_KEY, generate_jwt_token

logger = logging.getLogger(__name__)

PREPROCESSOR_RESOURCE_KEY = "preprocessor_folder"
PREPROCESSOR_OUTPUT_EXAMPLE_KEY = "preprocessor_example"


def build_agent_resource(
    agent_key: str,
    definition: dict[str, Any],
    resources: list[tuple[str, bytes]],
) -> ActiveAgentResourceModel:
    """Build an ActiveAgentResourceModel from the multipart resource entries.

    Resource keys follow the convention ``<agent_key>:<resource_key>``:
      - ``<agent_key>:preprocessor_folder``  → preprocessor zip
      - ``<agent_key>:preprocessor_example`` → preprocessor result example JSON
      - ``<agent_key>:<rule_key>``           → rule zip
    """
    agent_definition = definition.get("agents", {}).get(agent_key)
    if not agent_definition:
        raise ValueError(f"Could not find agent {agent_key} in definition")

    resource_model = ActiveAgentResourceModel(preprocessor=None, rules=[], preprocessor_example=None)

    pre_processing = agent_definition.get("pre_processing", {})
    rules_definition = agent_definition.get("rules", {})

    for key, content in resources:
        if ":" not in key:
            continue

        resource_agent_key, resource_key = key.split(":", 1)
        if resource_agent_key != agent_key:
            continue

        if resource_key == PREPROCESSOR_RESOURCE_KEY:
            entrypoint = pre_processing.get("source", {}).get("entrypoint", "")
            module, _, class_name = entrypoint.partition(".")
            if not module or not class_name:
                raise ValueError(
                    f"Invalid preprocessor entrypoint for agent {agent_key}: '{entrypoint}'"
                )
            resource_model.preprocessor = Resource(
                content=content,
                module_name=module,
                class_name=class_name,
            )
        elif resource_key == PREPROCESSOR_OUTPUT_EXAMPLE_KEY:
            resource_model.preprocessor_example = content
        else:
            rule_definition = rules_definition.get(resource_key)
            if not rule_definition:
                raise ValueError(
                    f"Could not find rule {resource_key} for agent {agent_key} in definition"
                )

            entrypoint = rule_definition.get("source", {}).get("entrypoint", "")
            module, _, class_name = entrypoint.partition(".")
            if not module or not class_name:
                raise ValueError(
                    f"Invalid rule entrypoint for agent {agent_key} rule {resource_key}: '{entrypoint}'"
                )

            resource_model.rules.append(
                RuleResource(
                    key=resource_key,
                    content=content,
                    module_name=module,
                    class_name=class_name,
                    template=rule_definition.get("template", ""),
                )
            )

    if resource_model.preprocessor is None:
        raise ValueError(f"Preprocessor folder is required for active agent {agent_key}")

    return resource_model


def build_active_test_event(
    project_uuid: str,
    test_data: dict[str, Any],
    fallback_credentials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mount the lambda event for the active agent template.

    Mirrors exactly the keys that the active lambda template reads from the event.
    Injects the JWT token into ``project.auth_token`` (paridade com fluxo passivo).
    """
    project = test_data.get("project", {})
    if isinstance(project, str):
        project = json.loads(project)
    project = dict(project)

    if JWT_PROJECT_KEY not in project:
        token = generate_jwt_token(project_uuid, settings.JWT_SECRET_KEY)
        project[JWT_PROJECT_KEY] = token

    credentials = test_data.get("credentials")
    if credentials is None:
        credentials = fallback_credentials or {}
    if isinstance(credentials, str):
        credentials = json.loads(credentials)

    return {
        "payload": test_data.get("payload", {}),
        "params": test_data.get("params", {}),
        "credentials": credentials,
        "project": project,
        "project_rules": test_data.get("project_rules", []),
        "ignored_official_rules": test_data.get("ignored_official_rules", []),
        "global_rule": test_data.get("global_rule"),
    }


async def run(  # noqa: PLR0915
    data: RunRequestModel,
    resources: list[tuple[str, bytes]],
    function_name: str,
    request_id: str,
    lambda_client: AWSLambdaClient,
) -> AsyncIterator[bytes]:
    """Stream NDJSON responses while running test cases for an active agent."""

    initial_data: CLIResponse = {
        "message": "Processing test run...",
        "data": {
            "project_uuid": str(data.project_uuid),
        },
        "success": True,
        "code": "PROCESSING_STARTED",
    }
    yield send_response(initial_data, request_id=request_id)
    logger.info(f"Starting active agent test run for {data.agent_key}")

    agent_resource = build_agent_resource(data.agent_key, data.definition, resources)

    processor = ActiveAgentProcessor(str(data.project_uuid), data.toolkit_version, agent_resource)
    agent_zip = processor.process(data.agent_key)

    processed_response: CLIResponse = {
        "message": f"Active agent {data.agent_key} processed successfully",
        "data": {
            "agent_key": data.agent_key,
            "size_kb": round(len(agent_zip.getvalue()) / 1024, 2),
            "rules": [rule.key for rule in agent_resource.rules],
        },
        "success": True,
        "code": "ACTIVE_AGENT_PROCESSED",
    }
    yield send_response(processed_response, request_id=request_id)

    creating_response: CLIResponse = {
        "message": "Creating active agent",
        "data": {
            "project_uuid": str(data.project_uuid),
            "function_name": function_name,
        },
        "success": True,
        "code": "LAMBDA_FUNCTION_CREATING",
    }
    yield send_response(creating_response, request_id=request_id)

    lambda_function: LambdaFunction = lambda_client.create_function(
        function_name=function_name,
        handler="lambda_function.lambda_handler",
        code=agent_zip,
        description=f"CLI run for active agent {data.agent_key}. Project: {data.project_uuid}",
    )

    if not lambda_function.arn or not lambda_function.name:
        raise ValueError("Failed to create active agent")

    if not await lambda_client.wait_for_function_active(lambda_function.arn):
        raise ValueError("Active agent did not became active")

    logger.info(f"Active agent {lambda_function.arn} active, invoking...")

    starting_response: CLIResponse = {
        "message": "Running test cases",
        "data": {
            "project_uuid": str(data.project_uuid),
            "function_name": function_name,
        },
        "success": True,
        "code": "STARTING_TEST_CASES",
    }
    yield send_response(starting_response, request_id=request_id)

    fallback_credentials = data.tool_credentials or {}

    for test_case, test_data in data.test_definition.get("tests", {}).items():
        logger.info(f"Running test case {test_case} for active agent {data.agent_key}")

        running_response: CLIResponse = {
            "message": f"Running test case {test_case}",
            "data": {
                "project_uuid": str(data.project_uuid),
                "function_name": function_name,
                "test_case": test_case,
            },
            "success": True,
            "code": "TEST_CASE_RUNNING",
        }
        yield send_response(running_response, request_id=request_id)

        test_event = build_active_test_event(
            project_uuid=str(data.project_uuid),
            test_data=test_data,
            fallback_credentials=fallback_credentials,
        )

        invoke_result, invoke_start_time, invoke_end_time = lambda_client.invoke_function(
            lambda_function.arn, test_event
        )

        response_payload = invoke_result.get("response", {}) or {}
        has_error = (
            response_payload.get("errorMessage") is not None
            or response_payload.get("errorType") is not None
            or (
                isinstance(response_payload.get("error"), dict)
                and bool(response_payload.get("error"))
                and response_payload.get("status") not in (0, 1)
            )
            or (
                isinstance(response_payload.get("error"), str)
                and bool(response_payload.get("error"))
            )
        )

        test_response: CLIResponse = {
            "message": "Test case completed",
            "data": {
                "test_case": test_case,
                "logs": invoke_result.get("logs"),
                "duration": invoke_end_time - invoke_start_time,
                "test_status_code": 400 if has_error else invoke_result.get("status_code", 200),
                "test_response": response_payload,
            },
            "success": True,
            "code": "TEST_CASE_COMPLETED",
        }

        yield send_response(test_response, request_id=request_id)
