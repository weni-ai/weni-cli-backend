import logging
from collections.abc import AsyncIterator
from io import BytesIO
from typing import Any

from fastapi import status
from fastapi.responses import StreamingResponse

from app.clients.gallery_client import GalleryClient
from app.core.response import CLIResponse, send_response
from app.services.agents.active.models import ActiveAgentResourceModel, Resource, RuleResource
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
        response_status = {"code": status.HTTP_200_OK}

        # Async generator for streaming responses
        async def response_stream() -> AsyncIterator[bytes]:  # noqa: PLR0915
            try:
                success, response = self._validate_resources(agent_resources_entries, response_status)
                if not success:
                    if response:
                        yield response
                    return

                # Initial message
                yield self._send_initial_message(resource_count)

                # Process resources
                agents_resources, response = self._process_resources(agent_resources_entries)
                if not agents_resources:
                    if response:
                        yield response
                    return

                # Process agents
                agents_lambda_map, response = await self._process_agents(agents_resources)
                if not agents_lambda_map:
                    if response:
                        yield response
                    return

                # Push to gallery
                success, response = self._push_agents_to_gallery(agents_lambda_map, response_status)
                if not success:
                    if response:
                        yield response
                    return

                # Final message
                yield self._send_final_message(resource_count, len(agents_lambda_map))

            except Exception as e:
                response_status["code"] = status.HTTP_400_BAD_REQUEST  # noqa: PLR2004
                error_response: CLIResponse = {
                    "success": False,
                    "message": f"Unexpected error processing agents: {str(e)}",
                    "request_id": self.request_id,
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "error_code": "UNEXPECTED_ERROR",
                        "status_code": 400,  # noqa: PLR2004
                        "error_details": str(e)
                    }
                }
                yield send_response(error_response, request_id=self.request_id)

        return StreamingResponse(
            response_stream(),
            media_type="application/x-ndjson",
            status_code=response_status["code"]
        )

    def _validate_resources(
        self,
        agent_resources_entries: list[tuple[str, bytes]],
        response_status: dict
    ) -> tuple[bool, bytes | None]:
        if not agent_resources_entries:
            response_status["code"] = status.HTTP_400_BAD_REQUEST  # noqa: PLR2004
            error_response: CLIResponse = {
                "success": False,
                "message": "No agent resources provided for processing",
                "request_id": self.request_id,
                "data": {
                    "project_uuid": str(self.project_uuid),
                    "error_code": "NO_RESOURCES",
                    "status_code": 400  # noqa: PLR2004
                }
            }
            return False, send_response(error_response, request_id=self.request_id)
        return True, None

    def _send_initial_message(self, resource_count: int) -> bytes:
        initial_data: CLIResponse = {
            "message": "Starting agent processing...",
            "data": {
                "project_uuid": str(self.project_uuid),
                "total_files": resource_count,
            },
            "success": True,
            "progress": 0.01,
            "code": "PROCESSING_STARTED",
        }
        logger.info(f"Starting processing for {resource_count} rules")
        return send_response(initial_data, request_id=self.request_id)

    def _process_resources(
        self,
        agent_resources_entries: list[tuple[str, bytes]]
    ) -> tuple[dict[str, ActiveAgentResourceModel] | None, bytes | None]:
        agents_resources: dict[str, ActiveAgentResourceModel] = {}
        for key, resource in agent_resources_entries:
            try:
                agent_key, resource_key = key.split(":")
            except ValueError:
                error_response: CLIResponse = {
                    "success": False,
                    "message": f"Invalid resource format: {key}. Expected format is 'agent_key:resource_key'",
                    "request_id": self.request_id,
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "error_code": "INVALID_RESOURCE_FORMAT",
                        "status_code": 400,  # noqa: PLR2004
                        "resource_key": key
                    }
                }
                return None, send_response(error_response, request_id=self.request_id)

            if not agents_resources.get(agent_key):
                agents_resources[agent_key] = ActiveAgentResourceModel(
                    preprocessor=None,
                    rules=[],
                    preprocessor_example=None,
                )

            success, response = self._process_resource(agent_key, resource_key, resource, agents_resources)
            if not success:
                return None, response

        return agents_resources, None

    def _process_resource(
        self,
        agent_key: str,
        resource_key: str,
        resource: bytes,
        agents_resources: dict[str, ActiveAgentResourceModel]
    ) -> tuple[bool, bytes | None]:
        try:
            if resource_key == PREPROCESSOR_RESOURCE_KEY:
                agents_resources[agent_key].preprocessor = self.mount_preprocessor_resource(
                    agent_key, self.definition, resource
                )
            elif resource_key == PREPROCESSOR_OUTPUT_EXAMPLE_KEY:
                agents_resources[agent_key].preprocessor_example = resource
            else:
                agents_resources[agent_key].rules.append(
                    self.mount_rule_resource(agent_key, resource_key, self.definition, resource)
                )
            return True, None
        except Exception as e:
            error_code = "PREPROCESSOR_ERROR" if resource_key == PREPROCESSOR_RESOURCE_KEY else "RULE_PROCESSING_ERROR"
            error_response: CLIResponse = {
                "success": False,
                "message": f"Error processing {resource_key} for agent {agent_key}: {str(e)}",
                "request_id": self.request_id,
                "data": {
                    "project_uuid": str(self.project_uuid),
                    "error_code": error_code,
                    "status_code": 400,  # noqa: PLR2004
                    "agent_key": agent_key,
                    "resource_key": resource_key
                }
            }
            return False, send_response(error_response, request_id=self.request_id)

    async def _process_agents(
        self,
        agents_resources: dict[str, ActiveAgentResourceModel]
    ) -> tuple[dict[str, BytesIO] | None, bytes | None]:
        agents_lambda_map = {}

        # Check if all agents have preprocessor
        for agent_key, agent_resource in agents_resources.items():
            if not agent_resource.preprocessor:
                error_response: CLIResponse = {
                    "success": False,
                    "message": f"Agent {agent_key} has no preprocessor defined",
                    "request_id": self.request_id,
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "error_code": "MISSING_PREPROCESSOR",
                        "status_code": 400,  # noqa: PLR2004
                        "agent_key": agent_key
                    }
                }
                return None, send_response(error_response, request_id=self.request_id)

            success, response = await self._process_agent(agent_key, agent_resource, agents_lambda_map)
            if not success:
                return None, response

        return agents_lambda_map, None

    async def _process_agent(
        self,
        agent_key: str,
        agent_resource: ActiveAgentResourceModel,
        agents_lambda_map: dict[str, BytesIO]
    ) -> tuple[bool, bytes | None]:
        logger.info(f"Processing agent: {agent_key}")
        try:
            processor = ActiveAgentProcessor(
                self.project_uuid,
                self.toolkit_version,
                agent_resource
            )
            agent_lambda = processor.process(agent_key)

            if not agent_lambda:
                error_response: CLIResponse = {
                    "success": False,
                    "message": (
                        f"Failed to process agent {agent_key}. "
                        "Please verify if all required files were provided."
                    ),
                    "request_id": self.request_id,
                    "data": {
                        "project_uuid": str(self.project_uuid),
                        "error_code": "AGENT_PROCESSING_ERROR",
                        "status_code": 400,  # noqa: PLR2004
                        "agent_key": agent_key
                    }
                }
                return False, send_response(error_response, request_id=self.request_id)

            agents_lambda_map[agent_key] = agent_lambda

            # add the preprocessor output example to the definition if it exists
            if agent_resource.preprocessor_example:
                self.definition["agents"][agent_key]["pre-processing"]["result_example"] = (
                    agent_resource.preprocessor_example
                )

            return True, None
        except SyntaxError as e:
            error_response: CLIResponse = {
                "success": False,
                "message": f"Syntax error processing agent {agent_key}: {str(e)}",
                "request_id": self.request_id,
                "data": {
                    "project_uuid": str(self.project_uuid),
                    "error_code": "SYNTAX_ERROR",
                    "status_code": 400,  # noqa: PLR2004
                    "agent_key": agent_key,
                    "error_details": str(e)
                }
            }
            return False, send_response(error_response, request_id=self.request_id)
        except Exception as e:
            error_response: CLIResponse = {
                "success": False,
                "message": f"Error processing agent {agent_key}: {str(e)}",
                "request_id": self.request_id,
                "data": {
                    "project_uuid": str(self.project_uuid),
                    "error_code": "AGENT_PROCESSING_ERROR",
                    "status_code": 400,  # noqa: PLR2004
                    "agent_key": agent_key,
                    "error_details": str(e)
                }
            }
            return False, send_response(error_response, request_id=self.request_id)

    def _push_agents_to_gallery(
        self,
        agents_lambda_map: dict[str, BytesIO],
        response_status: dict
    ) -> tuple[bool, bytes | None]:
        logger.info(f"Sending {len(agents_lambda_map)} agents to Gallery")
        success, response = self.push_to_gallery(agents_lambda_map)

        if not success and response:
            response_status["code"] = response.get("data", {}).get(
                "status_code",
                status.HTTP_400_BAD_REQUEST  # noqa: PLR2004
            )
            return False, send_response(response, request_id=self.request_id)
        return True, None

    def _send_final_message(self, resource_count: int, processed_count: int) -> bytes:
        final_data: CLIResponse = {
            "message": "Agents processed successfully",
            "data": {
                "project_uuid": str(self.project_uuid),
                "total_files": resource_count,
                "processed_files": processed_count,
                "status": "completed",
            },
            "success": True,
            "code": "PROCESSING_COMPLETED",
            "progress": 1.0,
        }
        return send_response(final_data, request_id=self.request_id)

    def mount_preprocessor_resource(
        self,
        agent_key: str,
        definition: dict[str, Any],
        resource: bytes
    ) -> Resource:
        source = definition["agents"][agent_key]["pre-processing"]["source"]

        module = source["entrypoint"].split(".")[0]
        class_name = source["entrypoint"].split(".")[1]

        return Resource(
            content=resource,
            module_name=module,
            class_name=class_name,
        )

    def mount_rule_resource(
        self,
        agent_key: str,
        rule_key: str,
        definition: dict[str, Any],
        resource: bytes
    ) -> RuleResource:
        source = definition["agents"][agent_key]["rules"][rule_key]["source"]
        template = definition["agents"][agent_key]["rules"][rule_key]["template"]

        module = source["entrypoint"].split(".")[0]
        class_name = source["entrypoint"].split(".")[1]

        return RuleResource(
            key=rule_key,
            content=resource,
            module_name=f"{module}",
            class_name=class_name,
            template=template,
        )

    def push_to_gallery(self, agents_lambda_map: dict[str, BytesIO]) -> tuple[bool, CLIResponse | None]:
        """
        Push processed agents to Gallery.

        Args:
            agents_lambda_map: Dictionary mapping agent keys to their lambda function zip files

        Returns:
            Tuple of (success, response)
        """
        try:
            logger.info(
                f"Sending {len(agents_lambda_map)} processed agents to Gallery for project {self.project_uuid}"
            )
            gallery_client = GalleryClient(self.project_uuid, self.authorization)
            response = gallery_client.push_agents(self.definition, agents_lambda_map)

            if response.status_code != status.HTTP_201_CREATED:
                error_message = "Failed to send agents to Gallery"
                error_code = "GALLERY_ERROR"
                status_code = response.status_code

                try:
                    error_data = response.json()
                    if error_data.get("code"):
                        error_code = error_data["code"]
                        error_message = error_data.get("detail", error_message)
                except:  # noqa: E722
                    pass

                return False, CLIResponse(
                    success=False,
                    message=error_message,
                    request_id=self.request_id,
                    data={
                        "project_uuid": str(self.project_uuid),
                        "error_code": error_code,
                        "status_code": status_code
                    }
                )

            logger.info(f"Successfully pushed agents to Gallery for project {self.project_uuid}")
            return True, None

        except Exception as e:
            logger.error(f"Error pushing agents to Gallery: {str(e)}", exc_info=True)
            return False, CLIResponse(
                success=False,
                message=str(e),
                request_id=self.request_id,
                data={
                    "project_uuid": str(self.project_uuid),
                    "error": str(e),
                    "error_code": "UNEXPECTED_ERROR",
                    "status_code": 500
                }
            )