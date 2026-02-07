from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..config import Paths
from ..security.models import UserStore, User, RoleName
from ..utils.auth import require_token, require_role
from ..utils.audit import NDJSONAuditLogger, make_audit_record


router = APIRouter(prefix="/users", tags=["users"])

paths = Paths()
_store = UserStore(paths.interim_dir / "users.json")
_audit = NDJSONAuditLogger(paths.interim_dir / "audit.ndjson")


class UserCreate(BaseModel):
    username: str
    roles: List[str] | None = None
    active: bool = True
    metadata: dict[str, str] | None = None


class UserResponse(BaseModel):
    user_id: str
    username: str
    roles: List[str]
    active: bool
    metadata: dict[str, str] | None = None


@router.get("/", response_model=List[UserResponse])
def list_users(_: None = Depends(require_token)):
    require_role(RoleName.ADMIN)
    users = _store.list_users()
    return [
        UserResponse(
            user_id=u.user_id,
            username=u.username,
            roles=u.roles,
            active=u.active,
            metadata=u.metadata or {},
        )
        for u in users
    ]


@router.post("/", response_model=UserResponse)
def create_user(payload: UserCreate, _: None = Depends(require_token)):
    require_role(RoleName.ADMIN)
    try:
        user = _store.create_user(
            username=payload.username,
            roles=payload.roles,
            active=payload.active,
            metadata=payload.metadata or {},
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    rec = make_audit_record(
        actor="api_client",
        action="user_create",
        resource=user.user_id,
        details={"username": user.username, "roles": user.roles},
    )
    try:
        _audit.append(rec)
    except Exception:
        pass

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=user.roles,
        active=user.active,
        metadata=user.metadata or {},
    )
