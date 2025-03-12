"""
Permissions endpoints
"""

from typing import Annotated, Any

from fastapi import APIRouter, Header, status

from app.api.v1.models.requests import VerifyPermissionRequestModel
from app.clients.connect_client import ConnectClient

router = APIRouter()


@router.post("/verify")
async def verify_permission(
    data: VerifyPermissionRequestModel,
    authorization: Annotated[str, Header()],
) -> dict[str, Any]:
    connect_client = ConnectClient(
        authorization,
        str(data.project_uuid),
    )

    try:
        response = connect_client.check_authorization()

        if response.status_code == status.HTTP_200_OK:
            return {"status": "ok"}
        else:
            return {"status": "error", "message": response.json()}

    except Exception as e:
        return {"status": "error", "message": str(e)}
