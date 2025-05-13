import logging
from typing import Any

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSAgentsClient:
    def __init__(self) -> None:
        self.client = boto3.client("bedrock-agent", region_name=settings.AWS_REGION)

    def get_agent(self, agent_id: str) -> Any:
        try:
            response = self.client.get_agent(agentId=agent_id)
            return response
        except Exception as e:
            logger.error(f"Error getting agent: {e}")
            raise e

    def list_agent_action_groups(self, agent_id: str, agent_version: str) -> Any:
        try:
            response = self.client.list_agent_action_groups(
                agentId=agent_id, agentVersion=agent_version
            )
            return response
        except Exception as e:
            logger.error(f"Error listing agent action groups: {e}")
            raise e

    def list_agent_versions(self, agent_id: str) -> Any:
        try:
            response = self.client.list_agent_versions(agentId=agent_id)
            return response
        except Exception as e:
            logger.error(f"Error listing agent versions: {e}")
            raise e

    def list_agents(
        self, next_token: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            if next_token:
                response = self.client.list_agents(nextToken=next_token, maxResults=100)
            else:
                response = self.client.list_agents(maxResults=100)

            if response.get("nextToken") and response.get("nextToken") != next_token:
                return response["agentSummaries"], response.get("nextToken")

            return response["agentSummaries"], None
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            raise e

    def find_agent_by_name(self, name: str, project_uuid: str) -> Any:
        agent_name = f"{name}-project-{project_uuid}"
        agents, next_token = self.list_agents()

        while next_token:
            for agent in agents:
                if agent["agentName"] == agent_name:
                    return agent

            agents, next_token = self.list_agents(next_token)

        return None

    def search_agent_by_name(self, search_string: str) -> Any:
        agents, next_token = self.list_agents()

        matches = []

        while next_token:
            for agent in agents:
                if search_string.lower() in agent["agentName"].lower():
                    matches.append(agent)

            agents, next_token = self.list_agents(next_token)

        return matches
