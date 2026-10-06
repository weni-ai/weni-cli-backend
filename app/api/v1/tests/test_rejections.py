import json
from uuid import uuid4

from starlette.requests import Request

import app.main
from app.api.v1.rejections import RequestRejectedError, handle_request_rejected

_HTTP_STATUS = 403
_CODE = "SOME_CODE"
_MESSAGE = "Some message."


def _error(request_id: str) -> RequestRejectedError:
    return RequestRejectedError(
        http_status=_HTTP_STATUS,
        code=_CODE,
        message=_MESSAGE,
        request_id=request_id,
    )


def test_request_rejected_error_has_no_status_code() -> None:
    error = _error(str(uuid4()))

    assert not hasattr(error, "status_code")


async def test_handle_request_rejected_returns_cli_response() -> None:
    request_id = str(uuid4())
    request = Request({"type": "http"})

    response = await handle_request_rejected(request, _error(request_id))

    assert response.status_code == _HTTP_STATUS
    assert response.headers["content-type"] == "application/json"
    assert json.loads(response.body) == {
        "message": _MESSAGE,
        "data": None,
        "success": False,
        "code": _CODE,
        "request_id": request_id,
    }


def test_handler_is_registered_on_the_application() -> None:
    assert app.main.app.exception_handlers[RequestRejectedError] is handle_request_rejected
