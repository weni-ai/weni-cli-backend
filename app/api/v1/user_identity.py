import jwt

BEARER_PREFIX = "Bearer "


def read_user_email(authorization: str | None) -> str | None:
    if authorization is None or not authorization.startswith(BEARER_PREFIX):
        return None
    token = authorization.removeprefix(BEARER_PREFIX)
    # AuthorizationMiddleware has already had Connect validate this exact token, so the
    # local decode reads a validated token without an extra call.
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return None
    email = payload.get("email")
    if isinstance(email, str) and email:
        return email
    return None
