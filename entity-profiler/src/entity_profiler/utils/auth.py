import os
from fastapi import Header, HTTPException, status


def _load_api_keys() -> set[str]:
    raw = os.getenv("EP_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def validate_token_or_key(bearer_token: str | None, api_key: str | None):
    token = os.getenv("EP_API_TOKEN")
    api_keys = _load_api_keys()
    if not token and not api_keys:
        return

    if token and bearer_token == token:
        return
    if api_keys and api_key and api_key in api_keys:
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def require_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, convert_underscores=False),
):
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    validate_token_or_key(bearer, x_api_key)
    return None
