import logging
from collections.abc import AsyncIterator
from io import BytesIO
from typing import Any, cast

from fastapi import status
from fastapi.responses import StreamingResponse

from app.clients.gallery_client import GalleryClient
from app.core.response import CLIResponse, send_response
from app.services.agents.active.models import ActiveAgentResourceModel
from app.services.agents.active.processor import ActiveAgentProcessor
from app.services.agents.configurators import AgentConfigurator

logger = logging.getLogger(__name__)

PREPROCESSOR_RESOURCE_KEY = "preprocessor_folder"
PREPROCESSOR_OUTPUT_EXAMPLE_KEY = "preprocessor_example"

class ActiveAgentConfigurator(AgentConfigurator):
    def __init__(
        self, project_uuid: str, definition: dict[str, Any], toolkit_version: str, request_id: str, authorization: str
    ):
        super().__init__(project_uuid, definition, toolkit_version, request_id, authorization)

    def configure_agents(self, agent_resources_entries: list[tuple[str, bytes]]) -> StreamingResponse:
        resource_count = len(agent_resources_entries)

        # Async generator for streaming responses
        async def response_stream() -> AsyncIterator[bytes]:
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
            logger.info(f"Starting processing for {resource_count} rules")
            yield send_response(initial_data, request_id=self.request_id)

            # map each agent resources to the agents_keys
            agents_resources: dict[str, ActiveAgentResourceModel] = {}
            for key, resource in agent_resources_entries:
                agent_key, resource_key = key.split(":")

                if not agents_resources.get(agent_key):
                    agents_resources[agent_key] = ActiveAgentResourceModel(
                        preprocessor=b"", rules={}, preprocessor_example=None
                    )

                if resource_key == PREPROCESSOR_RESOURCE_KEY:
                    agents_resources[agent_key].preprocessor = resource
                elif resource_key == PREPROCESSOR_OUTPUT_EXAMPLE_KEY:
                    agents_resources[agent_key].preprocessor_example = resource
                else:
                    agents_resources[agent_key].rules[resource_key] = resource

            agents_lambda_map = {}

            # for each agent, create the lambda function for it
            for agent_key, agent_resource in agents_resources.items():
                # create the lambda function for the preprocessor
                print("Agent resource 🔥", agent_resource)
                processor = ActiveAgentProcessor(self.project_uuid, self.toolkit_version, agent_resource)
                agent_lambda = processor.process(agent_key)

                if not agent_lambda:
                    logger.error(f"Error processing agent {agent_key}")
                    raise Exception(f"Error processing agent {agent_key}")

                agents_lambda_map[agent_key] = agent_lambda

                # add the preprocessor output example to the definition if it exists
                if agent_resource.preprocessor_example:
                    self.definition["agents"][agent_key]["pre-processing"]["result_example"] = (
                        agent_resource.preprocessor_example
                    )

            # push the agents to the gallery
            success, response = self.push_to_gallery(agents_lambda_map)

            if not success:
                # TODO: check if this consecutive error is handled well in the CLI
                response_data = {} if not response else response.get("data") or {}
                error_message = str(response_data.get("error", "Unknown error while pushing agents..."))
                raise Exception(error_message)

            # Final message
            final_data: CLIResponse = {
                "message": "Agents processed successfully",
                "data": {
                    "project_uuid": str(self.project_uuid),
                    "total_files": resource_count,
                    "processed_files": len(agents_lambda_map),
                    "status": "completed",
                },
                "success": True,
                "code": "PROCESSING_COMPLETED",
                "progress": 1.0,
            }
            yield send_response(final_data, request_id=self.request_id)

        return StreamingResponse(response_stream(), media_type="application/x-ndjson")

    def push_to_gallery(self, agents_lambda_map: dict[str, BytesIO]) -> tuple[bool, CLIResponse | None]:
        try:
            logger.info(
                f"Sending {len(agents_lambda_map)} processed agents to Gallery for project {self.project_uuid}"
            )
            gallery_client = GalleryClient(self.project_uuid, self.authorization)

            response = gallery_client.push_agents(self.definition, agents_lambda_map)

            if response.status_code != status.HTTP_200_OK:
                raise Exception(f"Failed to push agents to Gallery: {response.status_code} {response.text}")

            logger.info(f"Successfully pushed agents to Gallery for project {self.project_uuid}")

            return True, None
        except Exception as e:
            logger.error(f"Error pushing agents to Gallery: {str(e)}", exc_info=True)
            return False, CLIResponse(
                message=str(e),
                success=False,
            )
