"""
Unit tests for NexusClient.
"""

import json

import pytest
import requests_mock
from fastapi import status

from app.clients.nexus_client import NexusClient
from app.core.config import settings


@pytest.fixture
def auth_token() -> str:
    """Return a sample authentication token."""
    return "Bearer sample_token"


@pytest.fixture
def nexus_client(auth_token: str) -> NexusClient:
    """Return a NexusClient instance."""
    return NexusClient(auth_token)


@pytest.fixture
def project_uuid() -> str:
    """Return a sample project UUID."""
    return "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def agents_definition() -> dict:
    """Return a sample agents definition."""
    return {
        "agents": {
            "test-agent": {
                "name": "Test Agent",
                "description": "Test Agent Description",
                "skills": [
                    {
                        "name": "test-skill",
                        "description": "Test Skill Description",
                        "slug": "test-skill",
                        "source": {
                            "path": "test_skill.zip",
                            "entrypoint": "main.TestSkill",
                        },
                        "parameters": [
                            {
                                "email": {
                                    "type": "string",
                                    "description": "The email address of the customer",
                                    "required": True,
                                }
                            }
                        ],
                    }
                ],
            }
        }
    }


@pytest.fixture
def skill_files() -> dict:
    """Return sample skill files."""
    return {"skill-test-skill": ("test_skill.zip", b"test skill content", "application/zip")}


class TestNexusClient:
    """Tests for the NexusClient class."""

    def test_init(self, auth_token: str) -> None:
        """Test initialization of NexusClient."""
        client = NexusClient(auth_token)

        assert client.headers == {"Authorization": auth_token}
        assert client.base_url == settings.NEXUS_BASE_URL

    def test_push_agents(  # noqa: PLR0913
        self,
        requests_mock: requests_mock.Mocker,
        nexus_client: NexusClient,
        project_uuid: str,
        agents_definition: dict,
        skill_files: dict,
    ) -> None:
        """Test the push_agents method."""
        # Setup mocked response
        expected_url = f"{settings.NEXUS_BASE_URL}/api/agents/push"
        requests_mock.post(expected_url, json={"success": True}, status_code=status.HTTP_200_OK)

        # Call the method
        response = nexus_client.push_agents(project_uuid, agents_definition, skill_files)

        # Assertions
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"success": True}

        # Verify request details
        last_request = requests_mock.last_request
        assert last_request is not None

        # Verify correct headers were passed
        assert last_request.headers["Authorization"] == nexus_client.headers.get("Authorization")

        # Form data is multipart, need to verify each part exists
        assert "project_uuid" in last_request.text
        assert project_uuid in last_request.text
        assert "agents" in last_request.text
        assert json.dumps(agents_definition) in last_request.text

        # Verify files were present in the request
        assert "skill-test-skill" in last_request.text

    def test_push_agents_error_response(  # noqa: PLR0913
        self,
        requests_mock: requests_mock.Mocker,
        nexus_client: NexusClient,
        project_uuid: str,
        agents_definition: dict,
        skill_files: dict,
    ) -> None:
        """Test push_agents method when receiving an error response."""
        # Setup mocked response
        expected_url = f"{settings.NEXUS_BASE_URL}/api/agents/push"
        requests_mock.post(expected_url, json={"error": "Bad request"}, status_code=status.HTTP_400_BAD_REQUEST)

        # Call the method
        response = nexus_client.push_agents(project_uuid, agents_definition, skill_files)

        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == {"error": "Bad request"}
