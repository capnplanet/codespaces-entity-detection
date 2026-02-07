import os
from contextvars import ContextVar

from fastapi import Header, HTTPException, status

from ..config import Paths
from ..security.models import UserStore, User, RoleName
from ..utils.audit import NDJSONAuditLogger, make_audit_record


_USER_STORE: UserStore | None = None
_AUDIT_LOGGER: NDJSONAuditLogger | None = None
_CURRENT_USER: ContextVar[User | None] = ContextVar("_CURRENT_USER", default=None)


def _ensure_stores() -> None:
    global _USER_STORE, _AUDIT_LOGGER
    if _USER_STORE is None or _AUDIT_LOGGER is None:
        paths = Paths()
        if _USER_STORE is None:
            _USER_STORE = UserStore(paths.interim_dir / "users.json")
        if _AUDIT_LOGGER is None:
            _AUDIT_LOGGER = NDJSONAuditLogger(paths.interim_dir / "audit.ndjson")


def _get_or_create_user(username: str) -> User | None:
    """Lookup or create a User for a given username.

    Defaults new users to the VIEWER role; role elevation can be managed via
    the /users API.
    """

    if not username:
        return None
    if _USER_STORE is None:
        return None
    existing = _USER_STORE.get_by_username(username)
    if existing:
        return existing
    try:
        return _USER_STORE.create_user(username=username)
    except Exception:
        return _USER_STORE.get_by_username(username)


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

    _ensure_stores()

    # Map valid credentials to a User record when possible, storing it in a
    # per-request context variable. The mapping is configured via optional
    # environment variables:
    #   - EP_API_TOKEN_USER: username to associate with EP_API_TOKEN
    #   - EP_API_KEY_USER_MAP: comma-separated "key:username" pairs

    if token and bearer_token == token:
        username = os.getenv("EP_API_TOKEN_USER", "").strip()
        user = _get_or_create_user(username) if username else None
        _CURRENT_USER.set(user)
        if _AUDIT_LOGGER is not None:
            rec = make_audit_record(
                actor=user.username if user else "bearer_token",
                action="auth_success",
                resource="api",
                details={"method": "bearer"},
            )
            try:
                _AUDIT_LOGGER.append(rec)
            except Exception:
                pass
        return

    if api_keys and api_key and api_key in api_keys:
        mapping_raw = os.getenv("EP_API_KEY_USER_MAP", "").strip()
        mapped_username: str | None = None
        if mapping_raw:
            for pair in mapping_raw.split(","):
                pair = pair.strip()
                if not pair or ":" not in pair:
                    continue
                key_val, uname = pair.split(":", 1)
                if key_val.strip() == api_key:
                    mapped_username = uname.strip()
                    break
        user = _get_or_create_user(mapped_username) if mapped_username else None
        _CURRENT_USER.set(user)
        if _AUDIT_LOGGER is not None:
            rec = make_audit_record(
                actor=user.username if user else f"api_key:{api_key[:4]}...",
                action="auth_success",
                resource="api",
                details={"method": "api_key"},
            )
            try:
                _AUDIT_LOGGER.append(rec)
            except Exception:
                pass
        return

    if _AUDIT_LOGGER is not None:
        rec = make_audit_record(
            actor="unknown",
            action="auth_failure",
            resource="api",
            details={"has_bearer": bool(bearer_token), "has_api_key": bool(api_key)},
        )
        try:
            _AUDIT_LOGGER.append(rec)
        except Exception:
            pass
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


def current_user() -> User | None:
    """Return the current user bound to this request, if any.

    The user is set by validate_token_or_key based on the presented bearer
    token or API key and the configured environment mappings.
    """

    return _CURRENT_USER.get()


def require_role(required: RoleName) -> None:
    """Optionally enforce that the current user has a given role.

    Enforcement is controlled by the EP_ENFORCE_ROLES environment variable.
    When EP_ENFORCE_ROLES is falsy or unset, this function only emits audit
    records but does not block access.
    """

    enforce = os.getenv("EP_ENFORCE_ROLES", "").lower() in {"1", "true", "yes", "on"}
    user = current_user()
    actor = user.username if user is not None else "unknown"

    # Always audit that a privileged endpoint was reached.
    if _AUDIT_LOGGER is not None:
        try:
            rec = make_audit_record(
                actor=actor,
                action="privileged_endpoint_access",
                resource=f"role:{required.value}",
                details={"enforced": enforce},
            )
            _AUDIT_LOGGER.append(rec)
        except Exception:
            # Never fail the request because of audit issues.
            pass

    if not enforce:
        return

    if user is None:
        # When enforcement is enabled, absence of an authenticated user is an
        # authorization failure.
        raise HTTPException(status_code=403, detail="Forbidden: no user context")

    if required not in user.roles:
        if _AUDIT_LOGGER is not None:
            try:
                rec = make_audit_record(
                    actor=actor,
                    action="rbac_denied",
                    resource=f"role:{required.value}",
                )
                _AUDIT_LOGGER.append(rec)
            except Exception:
                pass
        raise HTTPException(status_code=403, detail="Forbidden: insufficient role")
