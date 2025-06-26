import json
from io import BytesIO
from unittest.mock import patch

from fastapi import status

from app.clients.gallery_client import GalleryClient


def test_gallery_push_agents_400():
    """Test pushing agents to Gallery with mocked 400 response."""
    # Mock configuration
    project_uuid = "test-project"
    auth_token = "test-token"
    
    # Sample agent definition
    agents_definition = {
        "agents": {
            "test_agent": {
                "name": "Test Agent",
                "description": "Test agent for 400 response"
            }
        }
    }
    
    # Sample rules files
    rules_files = {"test_agent": BytesIO(b"test content")}
    
    # Create a client instance
    client = GalleryClient(project_uuid, auth_token)
    
    # Mock the requests.post call to return 400
    with patch('requests.post') as mock_post:
        # Configure the mock to return a 400 response
        mock_post.return_value.status_code = status.HTTP_400_BAD_REQUEST
        mock_post.return_value.text = json.dumps({
            "detail": "Invalid agent configuration",
            "code": "INVALID_CONFIG"
        })
        
        # Make the request
        response = client.push_agents(agents_definition, rules_files)
        
        # Verify the response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = json.loads(response.text)
        assert response_data["detail"] == "Invalid agent configuration"
        assert response_data["code"] == "INVALID_CONFIG"

def test_gallery_push_agents_400_with_invalid_data():
    """Test pushing agents to Gallery with invalid data that should trigger 400."""
    project_uuid = "test-project"
    auth_token = "test-token"
    
    # Invalid agent definition (missing required fields)
    invalid_agents_definition = {
        "agents": {
            "test_agent": {}  # Missing required name and description
        }
    }
    
    rules_files = {"test_agent": BytesIO(b"test content")}
    
    client = GalleryClient(project_uuid, auth_token)
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = status.HTTP_400_BAD_REQUEST
        mock_post.return_value.text = json.dumps({
            "detail": "Missing required fields: name, description",
            "code": "MISSING_FIELDS"
        })
        
        response = client.push_agents(invalid_agents_definition, rules_files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = json.loads(response.text)
        assert "Missing required fields" in response_data["detail"]
        assert response_data["code"] == "MISSING_FIELDS" 