"""
Permissions endpoints
"""

from typing import Annotated, Any

from fastapi import APIRouter, Header, status
from fastapi.responses import JSONResponse

from app.api.v1.models.requests import VerifyPermissionRequestModel
from app.clients.connect_client import ConnectClient

router = APIRouter()


@router.post("/verify")
async def verify_permission(
    data: VerifyPermissionRequestModel,
    authorization: Annotated[str, Header()],
) -> JSONResponse:
    connect_client = ConnectClient(
        authorization,
        str(data.project_uuid),
    )

    try:
        response = connect_client.check_authorization()

        if response.status_code == status.HTTP_200_OK:
            return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok"})
        else:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED, content={"status": "error", "message": response.json()}
            )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"status": "error", "message": str(e)}
        )
