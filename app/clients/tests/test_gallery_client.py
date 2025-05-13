import json

import pytest
import requests_mock
from fastapi import status

from app.clients.gallery_client import GalleryClient
from app.core.config import settings


@pytest.fixture
def project_uuid() -> str:
    """Return a sample project UUID."""
    return "test_project_uuid"


@pytest.fixture
def user_auth_token() -> str:
    """Return a sample user auth token."""
    return "test_auth_token"


@pytest.fixture
def gallery_client(project_uuid: str, user_auth_token: str) -> GalleryClient:
    """Return a GalleryClient instance."""
    return GalleryClient(project_uuid=project_uuid, user_auth_token=user_auth_token)


@pytest.fixture
def agents_definition() -> dict:
    """Return a sample agents definition."""
    return {"agent1": "def1"}


@pytest.fixture
def rules_files() -> dict:
    """Return sample rules files."""
    # Assuming rules files are passed similarly to tool files in Nexus tests
    # Example: {"filename": (filename, file_content, content_type)}
    # Adjust if the actual structure is different
    return {"rules.yml": ("rules.yml", "rule content", "text/yaml")}


class TestGalleryClient:
    """Tests for the GalleryClient class."""

    def test_gallery_client_initialization(
        self,
        gallery_client: GalleryClient,
        project_uuid: str,
        user_auth_token: str,
    ) -> None:
        """Test initialization of GalleryClient."""
        assert gallery_client.project_uuid == project_uuid
        assert gallery_client.headers == {"Authorization": user_auth_token}
        assert gallery_client.base_url == settings.GALLERY_BASE_URL

    def test_push_agents(
        self,
        requests_mock: requests_mock.Mocker,
        gallery_client: GalleryClient,
        project_uuid: str,
        agents_definition: dict,
        rules_files: dict,
    ) -> None:
        """Test the push_agents method with a successful response."""
        expected_url = f"{settings.GALLERY_BASE_URL}/api/v3/agents/push/"
        mock_response_data = {"message": "Agents pushed successfully"}

        requests_mock.post(
            expected_url, json=mock_response_data, status_code=status.HTTP_200_OK
        )

        response = gallery_client.push_agents(
            agents_definition=agents_definition, rules_files=rules_files
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == mock_response_data

        # Verify request details
        last_request = requests_mock.last_request
        assert last_request is not None
        assert last_request.headers["Authorization"] == gallery_client.headers.get(
            "Authorization"
        )

        # Verify form data parts
        assert f'name="project_uuid"\r\n\r\n{project_uuid}' in last_request.text
        assert (
            f'name="agents"\r\n\r\n{json.dumps(agents_definition)}'
            in last_request.text
        )

        # Verify files part (checking for filename)
        assert 'filename="rules.yml"' in last_request.text
        assert "rule content" in last_request.text # Check for content too

    def test_push_agents_error(
        self,
        requests_mock: requests_mock.Mocker,
        gallery_client: GalleryClient,
        agents_definition: dict,
        rules_files: dict,
    ) -> None:
        """Test the push_agents method with an error response."""
        expected_url = f"{settings.GALLERY_BASE_URL}/api/v3/agents/push/"
        mock_error_data = {"detail": "Invalid request"}

        requests_mock.post(
            expected_url, json=mock_error_data, status_code=status.HTTP_400_BAD_REQUEST
        )

        response = gallery_client.push_agents(
            agents_definition=agents_definition, rules_files=rules_files
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == mock_error_data
