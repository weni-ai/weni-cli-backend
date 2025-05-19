import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import status
from fastapi.responses import StreamingResponse

from app.clients.nexus_client import NexusClient
from app.core.response import CLIResponse, send_response
from app.services.agents.configurators import AgentConfigurator
from app.services.tool.packager import process_tool

logger = logging.getLogger(__name__)


class PassiveAgentConfigurator(AgentConfigurator):
    def __init__(
        self, project_uuid: str, definition: dict[str, Any], toolkit_version: str, request_id: str, authorization: str
    ):
        super().__init__(project_uuid, definition, toolkit_version, request_id, authorization)

    def configure_agents(
        self,
        agent_resources_entries: list[tuple[str, bytes]],
    ) -> StreamingResponse:
        resource_count = len(agent_resources_entries)

        # Async generator for streaming responses
        async def response_stream() -> AsyncIterator[bytes]:
            try:
                # Initial message
                initial_data: CLIResponse = {
                    "message": "Processing agents...",
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "total_files": resource_count,
                    },
                    "success": True,
                    "progress": 0.01,
                    "code": "PROCESSING_STARTED",
                }
                logger.info(f"Starting processing for {resource_count} resources")
                yield send_response(initial_data, request_id=self.request_id)

                tool_mapping = {}
                processed_count = 0

                # Process each tool file
                for key, folder_zip in agent_resources_entries:
                    agent_key, tool_key = key.split(":")
                    processed_count += 1

                    # Process the tool
                    response, tool_zip_bytes = await process_tool(
                        folder_zip,
                        key,
                        str(self.project_uuid),
                        agent_key,
                        tool_key,
                        self.definition,
                        processed_count,
                        resource_count,
                        self.toolkit_version,
                    )

                    # Send response for this tool
                    yield send_response(response, request_id=self.request_id)

                    # If tool processing failed, stop and raise an exception
                    if not tool_zip_bytes:
                        response_data: dict = response.get("data") or {}
                        error_message = response_data.get("error", "Unknown error processing tool")
                        raise Exception(f"Failed to process tool {key}: {error_message}")

                    # Add to mapping if successful (we'll only get here if processing succeeded)
                    tool_mapping[key] = tool_zip_bytes

                # Send progress update for Nexus upload
                nexus_response: CLIResponse = {
                    "message": "Updating your agents...",
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "tool_count": len(tool_mapping),
                    },
                    "success": True,
                    "code": "NEXUS_UPLOAD_STARTED",
                    "progress": 0.99,
                }
                yield send_response(nexus_response, request_id=self.request_id)

                # Push to Nexus - always push, even if no tools
                success, error_response = self.push_to_nexus(tool_mapping)

                if not success and error_response is not None:
                    # Type cast error_response to CLIResponse for type checker
                    typed_error_response = cast(CLIResponse, error_response)
                    yield send_response(typed_error_response, request_id=self.request_id)

                    response_data = typed_error_response.get("data") or {}
                    error_message = str(response_data.get("error", "Unknown error while pushing agents..."))
                    raise Exception(error_message)

                # Final message
                final_data: CLIResponse = {
                    "message": "Agents processed successfully",
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "total_files": resource_count,
                        "processed_files": len(tool_mapping),
                        "status": "completed",
                    },
                    "success": True,
                    "code": "PROCESSING_COMPLETED",
                    "progress": 1.0,
                }
                yield send_response(final_data, request_id=self.request_id)

            except Exception as e:
                logger.error(f"Error processing agents for project {self.project_uuid}: {str(e)}", exc_info=True)
                error_data: CLIResponse = {
                    "message": "Error processing agents",
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                    },
                    "success": False,
                    "code": "PROCESSING_ERROR",
                }
                yield send_response(error_data, request_id=self.request_id)

        return StreamingResponse(response_stream(), media_type="application/x-ndjson")

    def push_to_nexus(
        self,
        tool_mapping: dict[str, Any],
    ) -> tuple[bool, CLIResponse | None]:
        """
        Push processed tools to Nexus.

        Args:
            project_uuid: The UUID of the project
            definition: The agent definition
            tool_mapping: Dictionary of processed tools
            request_id: The request ID for correlation
            authorization: The authorization header value

        Returns:
            Tuple of (success, response)
        """
        try:
            logger.info(f"Sending {len(tool_mapping)} processed tools to Nexus for project {self.project_uuid}")
            nexus_client = NexusClient(self.authorization, self.project_uuid)

            # We need to change the entrypoint to the lambda function we've created
            for _, agent_data in self.definition["agents"].items():
                for tool in agent_data["tools"]:
                    tool["source"]["entrypoint"] = "lambda_function.lambda_handler"

            response = nexus_client.push_agents(self.definition, tool_mapping)

            if response.status_code != status.HTTP_200_OK:
                raise Exception(f"Failed to push agents: {response.status_code} {response.text}")

            logger.info(f"Successfully pushed agents to Nexus for project {self.project_uuid}")

            return True, None

        except Exception as e:
            logger.error(f"Failed to push agents to Nexus: {str(e)}", exc_info=True)
            nexus_error: CLIResponse = {
                "message": "Failed to push agents",
                "data": {
                    "project_uuid": str(self.project_uuid),
                    "error": str(e),
                    "tools_processed": len(tool_mapping),
                },
                "success": False,
                "code": "NEXUS_UPLOAD_ERROR",
                "progress": 0.9,
            }

            return False, nexus_error
