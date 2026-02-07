from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time


@dataclass
class AuditRecord:
    """Single audit log record.

    This is intentionally minimal and append-only. Higher-level systems can
    ingest these records and enforce retention, rotation, and signing.
    """

    timestamp: float
    actor: str
    action: str
    resource: str
    details: Dict[str, Any]


class NDJSONAuditLogger:
    """Append-only NDJSON audit logger.

    Patterned after NDJSONEventStore but focused on security-relevant actions
    (auth, configuration changes, event acknowledgements, exports).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")


def build_audit_logger(base_dir: Path | str, filename: str = "audit.ndjson") -> NDJSONAuditLogger:
    """Construct a default NDJSONAuditLogger under a base interim directory."""

    base = Path(base_dir)
    return NDJSONAuditLogger(base / filename)


def make_audit_record(
    actor: Optional[str],
    action: str,
    resource: str,
    details: Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    """Helper to build a minimal AuditRecord with current time.

    `actor` is a free-form identifier (e.g., API key ID, token label, or user).
    """

    return AuditRecord(
        timestamp=time.time(),
        actor=actor or "unknown",
        action=action,
        resource=resource,
        details=details or {},
    )
