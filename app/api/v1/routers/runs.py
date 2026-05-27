"""
Run endpoints for executing test cases against ephemeral Lambdas.

Supports two flavors selected by the ``type`` field of the request:
  - ``passive``  → builds and invokes a Tool Lambda (current behaviour)
  - ``active``   → builds and invokes an Active Agent Lambda (preprocessor + rules)
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import FormData, UploadFile

from app.api.v1.models.requests import RunRequestModel
from app.clients.aws import AWSLambdaClient
from app.core.response import CLIResponse, send_response
from app.services.runs import active_strategy, tool_strategy

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("")
async def run_test(  # noqa: PLR0915
    request: Request,
    data: Annotated[RunRequestModel, Form()],
    authorization: Annotated[str, Header()],
) -> StreamingResponse:
    request_id = str(uuid4())
    logger.info(
        f"Processing test run for project {data.project_uuid} - type: {data.type} - request_id: {request_id}"
    )
    logger.debug(f"Agent definition: {data.definition}")

    form = await request.form()

    function_name = f"cli-{str(uuid4())}"
    lambda_client = AWSLambdaClient()

    if data.type == "active":
        resources = await _extract_active_resources(form)
        if not resources:
            raise HTTPException(
                status_code=400,
                detail="At least one active agent resource (preprocessor + rules) is required",
            )

        async def active_response_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in active_strategy.run(
                    data=data,
                    resources=resources,
                    function_name=function_name,
                    request_id=request_id,
                    lambda_client=lambda_client,
                ):
                    yield chunk
            except Exception as e:
                async for chunk in _yield_processing_error(data, e, request_id):
                    yield chunk
            finally:
                _safe_delete_function(lambda_client, function_name)

        return StreamingResponse(active_response_stream(), media_type="application/x-ndjson")

    tool_folder_zip = form.get("tool")

    if not tool_folder_zip or not isinstance(tool_folder_zip, UploadFile):
        raise HTTPException(status_code=400, detail="Tool folder zip is required")

    folder_zip = await tool_folder_zip.read()
    logger.info(f"Found tool folder to process for project {data.project_uuid}")

    async def passive_response_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in tool_strategy.run(
                data=data,
                folder_zip=folder_zip,
                function_name=function_name,
                request_id=request_id,
                lambda_client=lambda_client,
            ):
                yield chunk
        except Exception as e:
            async for chunk in _yield_processing_error(data, e, request_id):
                yield chunk
        finally:
            _safe_delete_function(lambda_client, function_name)

    return StreamingResponse(passive_response_stream(), media_type="application/x-ndjson")


async def _extract_active_resources(form: FormData) -> list[tuple[str, bytes]]:
    """Extract every multipart entry whose key follows the ``<agent_key>:<resource_key>`` convention."""
    resources: list[tuple[str, bytes]] = []
    for key, value in form.items():
        if isinstance(value, UploadFile) and ":" in key:
            content = await value.read()
            resources.append((key, content))
    return resources


async def _yield_processing_error(
    data: RunRequestModel, error: Exception, request_id: str
) -> AsyncIterator[bytes]:
    logger.error(
        f"Error processing test run for project {data.project_uuid}: {str(error)} - request_id: {request_id}"
    )
    error_data: CLIResponse = {
        "message": "Error processing test run",
        "data": {
            "project_uuid": str(data.project_uuid),
            "error": str(error),
        },
        "success": False,
        "code": "PROCESSING_ERROR",
    }
    yield send_response(error_data, request_id=request_id)


def _safe_delete_function(lambda_client: AWSLambdaClient, function_name: str) -> None:
    try:
        lambda_client.delete_function(function_name=function_name)
    except Exception as error:
        logger.error(f"Error deleting lambda {function_name}: {str(error)}")
