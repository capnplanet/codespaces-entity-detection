from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HealthEvent:
    entity_id: str
    severity: str  # info|warning|critical
    type: str
    description: str
    timestamp: float
    context: Dict[str, Any]


_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def severity_allows(emitted: str, minimum: str) -> bool:
    return _SEVERITY_ORDER.get(emitted, 0) >= _SEVERITY_ORDER.get(minimum, 0)
