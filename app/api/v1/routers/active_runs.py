"""
Active agent runs endpoints for handling file uploads and processing.
"""

import json
import logging
import zipfile
from collections.abc import AsyncIterator
from io import BytesIO
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from app.api.v1.models.requests import RunActiveAgentRequestModel
from app.clients.aws import AWSLambdaClient
from app.clients.aws.lambda_client import LambdaFunction
from app.core.response import CLIResponse, send_response

router = APIRouter()
logger = logging.getLogger(__name__)


async def extract_rule_source_from_zip(folder_zip: bytes, rule_key: str) -> str:
    """
    Extract rule source code from the zip file.
    
    Args:
        folder_zip: The zip file bytes
        rule_key: The key of the rule to extract
        
    Returns:
        str: The source code of the rule, or empty string if not found
    """
    logger.info(f"DEBUG - extract_rule_source_from_zip called with rule_key={rule_key}, zip_size={len(folder_zip) if folder_zip else 0}")
    try:
        with zipfile.ZipFile(BytesIO(folder_zip), 'r') as zip_ref:
            # List all files in the zip
            file_list = zip_ref.namelist()
            logger.info(f"DEBUG - Files in zip: {file_list}")
            
            # Look for the rule file - it might be in different locations
            possible_rule_files = [
                f"{rule_key}.py",
                f"rules/{rule_key}.py",
                f"rules/{rule_key}/main.py",
                f"{rule_key}/main.py",
                "main.py",  # Default main file
            ]
            
            for rule_file in possible_rule_files:
                if rule_file in file_list:
                    logger.info(f"DEBUG - Found rule file: {rule_file}")
                    with zip_ref.open(rule_file) as f:
                        rule_source = f.read().decode('utf-8')
                        logger.info(f"DEBUG - Rule source code preview: {rule_source[:200]}...")
                        return rule_source
            
            # If no specific rule file found, try to find any Python file
            for file_name in file_list:
                if file_name.endswith('.py'):
                    logger.info(f"DEBUG - Trying Python file: {file_name}")
                    with zip_ref.open(file_name) as f:
                        rule_source = f.read().decode('utf-8')
                        # Check if it contains a rule class
                        if 'class' in rule_source and 'Rule' in rule_source:
                            logger.info(f"DEBUG - Found rule class in {file_name}")
                            return rule_source
                            
            logger.warning(f"DEBUG - No rule source found for {rule_key}")
            return ""
            
    except Exception as e:
        logger.error(f"DEBUG - Error extracting rule source: {str(e)}")
        return ""


@router.post("")
async def run_active_agent_test(  # noqa: PLR0915
    request: Request,
    data: Annotated[RunActiveAgentRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    """
    Run tests for active agents.
    
    This endpoint processes active agent tests by:
    1. Creating a Lambda function with the agent's rules
    2. Loading webhook data from payload_path (if provided)
    3. Running test cases with the webhook data
    4. Returning streaming results
    
    **Important:** For rules to trigger correctly, the webhook data must be loaded
    from the payload_path JSON file. The rules use webhook_data.get("status") to
    determine if they should trigger:
    - PaymentApproved: expects webhook_data.get("status") == "payment-approved"
    - StatusInvoiced: expects webhook_data.get("status") == "invoiced"
    
    Args:
        request: FastAPI request object to access form data and files
        data: Agent run request data including payload_path for webhook JSON
        authorization: Authorization header
        
    Returns:
        StreamingResponse: Streaming response with test results
        
    Test Event Structure:
        {
            "payload": {...},       # Data loaded from payload_path JSON file (webhook data)
            "params": {...},        # test_definition params (contact info)
            "credentials": {...},   # Rule credentials
            "ignored_official_rules": [...],
            "project_rules": [...],
            "project": {...}
        }
    """
    request_id = str(uuid4())
    logger.info(f"Processing active agent test run for project {data.project_uuid} - request_id: {request_id}")
    logger.debug(f"Agent definition: {data.definition}")

    # Access the form data with files
    form = await request.form()
    
    form_keys = list(form.keys())
    for key in form_keys:
        value = form.get(key)
        if hasattr(value, 'filename'):
            logger.info(f"  - {key}: FILE({value.filename})")
        else:
            logger.info(f"  - {key}: {repr(value)}")

    # Extract and process agent files (rules, preprocessor, etc.)
    agent_folder_zip = form.get("rule")

    if not agent_folder_zip or not isinstance(agent_folder_zip, UploadFile):
        raise HTTPException(status_code=400, detail="Rule folder zip is required")

    folder_zip = await agent_folder_zip.read()

    logger.info(f"Found agent folder to process for project {data.project_uuid}")
    logger.debug(f"Agent folder zip: {agent_folder_zip}")

    function_name = f"cli-active-{str(uuid4())}"
    lambda_client = AWSLambdaClient()

    async def response_stream() -> AsyncIterator[bytes]:
        try:
            initial_data: CLIResponse = {
                "message": "Processing active agent test run...",
                "data": {
                    "project_uuid": str(data.project_uuid),
                },
                "success": True,
                "code": "PROCESSING_STARTED",
            }
            yield send_response(initial_data, request_id=request_id)
            logger.info(f"Starting active agent test run for agent {data.agent_key}")

            # Get agent information
            agent_info = data.definition["agents"].get(data.agent_key)

            if not agent_info:
                raise ValueError(f"Could not find agent {data.agent_key} in definition")

            # Active agents have rules instead of tools
            if "rules" not in agent_info:
                raise ValueError(f"Agent {data.agent_key} does not have rules - not an active agent")

            rules_info = agent_info.get("rules", {})
            preprocessing_info = agent_info.get("pre-processing", {})

            # TODO: Process the agent folder zip for active agents
            # This will be different from passive tools processing
            # Active agents need rule processing and lambda creation
            
            response = {
                "message": "Processing active agent",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "agent_key": data.agent_key,
                    "rules_count": len(rules_info),
                    "has_preprocessing": bool(preprocessing_info),
                },
                "success": True,
                "code": "AGENT_PROCESSING",
            }
            yield send_response(response, request_id=request_id)

            # For now, we'll create a placeholder lambda function
            # In the future, this should process rules and create the actual active agent lambda
            response = {
                "message": "Creating active agent lambda",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "function_name": function_name,
                },
                "success": True,
                "code": "LAMBDA_FUNCTION_CREATING",
            }
            yield send_response(response, request_id=request_id)

            # Build environment variables for the lambda function
            environment_vars = lambda_client.build_default_environment_variables(
                project_uuid=str(data.project_uuid),
                agent_key=data.agent_key,
                custom_vars={
                    "RULE_CREDENTIALS": json.dumps(data.rule_credentials),
                    "RULE_GLOBALS": json.dumps(data.rule_globals),
                    "RULE_KEY": data.rule_key,
                    "AGENT_TYPE": data.type,
                    "FUNCTION_TYPE": "active_agent",
                }
            )

            lambda_function: LambdaFunction = lambda_client.create_function(
                function_name=function_name,
                handler="lambda_function.lambda_handler",
                code=BytesIO(folder_zip),  # Convert bytes to BytesIO
                description=f"CLI run for active agent {data.agent_key}. Project: {data.project_uuid}",
                environment=environment_vars,
            )

            if not lambda_function.arn or not lambda_function.name:
                raise ValueError("Failed to create active agent lambda")

            if not await lambda_client.wait_for_function_active(lambda_function.arn):
                raise ValueError("Active agent lambda did not become active")

            logger.info(f"Active agent {lambda_function.arn} active, invoking...")

            response = {
                "message": "Running test cases",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "function_name": function_name,
                },
                "success": True,
                "code": "STARTING_TEST_CASES",
            }
            yield send_response(response, request_id=request_id)

            for test_case, test_data in data.test_definition.get("tests", {}).items():
                logger.info(f"Running test case {test_case} for active agent {data.agent_key}")

                response = {
                    "message": f"Running test case {test_case}",
                    "data": {
                        "project_uuid": str(data.project_uuid),
                        "function_name": function_name,
                        "test_case": test_case,
                    },
                    "success": True,
                    "code": "TEST_CASE_RUNNING",
                }
                yield send_response(response, request_id=request_id)

                # 🔧 FIX: Use webhook_data that already vem do CLI!
                webhook_data = test_data.get("webhook_data")
                #if webhook_data:
                #    logger.info(f"✅ USANDO webhook_data DO CLI: {webhook_data}")
                #else:
                if not webhook_data:
                    # Fallback: tenta carregar do arquivo se webhook_data não veio
                    webhook_data = {}
                    if data.payload_path:
                        try:
                            import os
                            if os.path.exists(data.payload_path):
                                with open(data.payload_path, 'r') as f:
                                    webhook_data = json.load(f)
                            else:
                                logger.warning(f"❌ PAYLOAD PATH DOES NOT EXIST: {data.payload_path}")
                        except Exception as e:
                            logger.error(f"❌ ERROR LOADING WEBHOOK DATA do {data.payload_path}: {str(e)}")

                # Build project_rules from the agent definition
                project_rules = []
                if "rules" in agent_info:
                    logger.info(f"DEBUG - Building project_rules from agent_info rules: {agent_info['rules']}")
                    for rule_key, rule_info in agent_info["rules"].items():
                        if rule_key == data.rule_key:  # Only include the rule being tested
                            logger.info(f"DEBUG - Including rule {rule_key} in project_rules")
                            
                            # Extract rule source code from zip file
                            rule_source_code = await extract_rule_source_from_zip(folder_zip, rule_key)
                            logger.info(f"DEBUG - Extracted rule source code length: {len(rule_source_code) if rule_source_code else 0}")
                            
                            project_rules.append({
                                "key": rule_key,
                                "template": rule_info.get("template", ""),
                                "source": rule_source_code,  # Include the actual source code
                            })
                            
                            logger.info(f"DEBUG - Built project_rule for {rule_key}: template={rule_info.get('template', '')}, source_length={len(rule_source_code) if rule_source_code else 0}")

                # For active agents, the test event structure matches what lambda template expects
                test_event = {
                    "payload": webhook_data,  # SEMPRE usa webhook_data correto
                    "params": test_data.get("params", {}),
                    "credentials": test_data.get("credentials", data.rule_credentials),
                    "ignored_official_rules": test_data.get("ignored_official_rules", []),
                    "project_rules": project_rules,
                    "project": test_data.get("project", {}),
                }

                # Invoke lambda function
                invoke_result, invoke_start_time, invoke_end_time = lambda_client.invoke_function(
                    lambda_function.arn, test_event
                )

                response_data = invoke_result.get("response", {})
                # Check if there's an error in the response
                has_error = (
                    response_data.get("errorMessage") is not None
                    or response_data.get("errorType") is not None
                    or response_data.get("error") is not None
                )

                test_response: CLIResponse = {
                    "message": "Test case completed",
                    "data": {
                        "test_case": test_case,
                        "logs": invoke_result.get("logs"),
                        "duration": invoke_end_time - invoke_start_time,
                        "test_status_code": 400 if has_error else invoke_result.get("status_code", 200),
                        "test_response": invoke_result.get("response"),
                    },
                    "success": True,
                    "code": "TEST_CASE_COMPLETED",
                }

                yield send_response(test_response, request_id=request_id)

        except Exception as e:
            logger.error(
                f"Error processing active agent test run for project {data.project_uuid}: {str(e)} - request_id: {request_id}"
            )
            error_data: CLIResponse = {
                "message": "Error processing active agent test run",
                "data": {
                    "project_uuid": str(data.project_uuid),
                    "error": str(e),
                },
                "success": False,
                "code": "PROCESSING_ERROR",
            }
            yield send_response(error_data, request_id=request_id)
        finally:
            # Delete lambda function
            try:
                lambda_client.delete_function(function_name=function_name)
            except Exception as e:
                logger.error(f"Error deleting active agent lambda {function_name}: {str(e)}")

    return StreamingResponse(response_stream(), media_type="application/x-ndjson") 