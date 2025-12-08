import os
from fastapi import Header, HTTPException, status


def require_token(authorization: str | None = Header(default=None)):
    token = os.getenv("EP_API_TOKEN")
    if not token:
        return None  # auth disabled
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    provided = authorization.split(" ", 1)[1].strip()
    if provided != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return None
