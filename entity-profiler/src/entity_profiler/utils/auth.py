import os
from fastapi import Header, HTTPException, status


def _load_api_keys() -> set[str]:
    raw = os.getenv("EP_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, convert_underscores=False),
):
    token = os.getenv("EP_API_TOKEN")
    api_keys = _load_api_keys()
    if not token and not api_keys:
        return None  # auth disabled

    # Bearer token path
    if token:
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization.split(" ", 1)[1].strip()
            if provided == token:
                return None
        # fall through to API key check if bearer missing/invalid

    # API key path
    if api_keys and x_api_key and x_api_key in api_keys:
        return None

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
