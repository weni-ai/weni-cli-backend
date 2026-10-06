from typing import cast

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.response import CLIResponse


class RequestRejectedError(Exception):
    def __init__(self, http_status: int, code: str, message: str, request_id: str) -> None:
        super().__init__(message)
        # sentry-sdk 2.24.1 captures handled exceptions that carry an integer status_code
        # inside failed_request_status_codes (401–598). With send_default_pii=True that
        # would send the bearer token and e-mail to Sentry.
        self.http_status = http_status
        self.code = code
        self.message = message
        self.request_id = request_id


async def handle_request_rejected(request: Request, exc: Exception) -> JSONResponse:
    error = cast(RequestRejectedError, exc)
    body: CLIResponse = {
        "message": error.message,
        "data": None,
        "success": False,
        "code": error.code,
        "request_id": error.request_id,
    }
    return JSONResponse(status_code=error.http_status, content=body)
