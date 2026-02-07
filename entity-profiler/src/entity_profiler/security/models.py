from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import json
import uuid


class RoleName(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    AUDITOR = "auditor"


@dataclass
class User:
    user_id: str
    username: str
    # For now we assume external auth (OIDC, etc.) or API-token binding.
    # Passwords or secrets are intentionally not stored here.
    roles: List[str]
    active: bool = True
    metadata: Dict[str, str] | None = None


class UserStore:
    """Simple JSON-backed user store.

    This is a minimal reference suitable for development and small deployments.
    Enterprise deployments should replace this with a proper database-backed
    implementation and stronger identity integration.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._users: Dict[str, User] = {}
        self._by_name: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for u in payload.get("users", []):
            user = User(
                user_id=u["user_id"],
                username=u["username"],
                roles=list(u.get("roles", [])),
                active=bool(u.get("active", True)),
                metadata=u.get("metadata") or {},
            )
            self._users[user.user_id] = user
            self._by_name[user.username] = user.user_id

    def _save(self) -> None:
        payload = {"users": [asdict(u) for u in self._users.values()]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def list_users(self) -> List[User]:
        return list(self._users.values())

    def list_active_users(self) -> List[User]:
        return [u for u in self._users.values() if u.active]

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        uid = self._by_name.get(username)
        if not uid:
            return None
        return self._users.get(uid)

    def create_user(
        self,
        username: str,
        roles: Optional[List[str]] = None,
        active: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ) -> User:
        if username in self._by_name:
            raise ValueError(f"User already exists: {username}")
        user_id = str(uuid.uuid4())
        user = User(
            user_id=user_id,
            username=username,
            roles=roles or [RoleName.VIEWER.value],
            active=active,
            metadata=metadata or {},
        )
        self._users[user_id] = user
        self._by_name[username] = user_id
        self._save()
        return user

    def update_user(
        self,
        user_id: str,
        *,
        roles: Optional[List[str]] = None,
        active: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> User:
        user = self._users.get(user_id)
        if not user:
            raise KeyError(user_id)
        data = asdict(user)
        if roles is not None:
            data["roles"] = roles
        if active is not None:
            data["active"] = active
        if metadata is not None:
            data["metadata"] = metadata
        updated = User(**data)
        self._users[user_id] = updated
        self._by_name[updated.username] = updated.user_id
        self._save()
        return updated

    def deactivate_user(self, user_id: str) -> None:
        user = self._users.get(user_id)
        if not user:
            return
        user.active = False
        self._users[user_id] = user
        self._save()

