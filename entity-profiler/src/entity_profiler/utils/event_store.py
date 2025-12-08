from pathlib import Path
import json
from typing import Dict, Iterable


class NDJSONEventStore:
    """Append-only NDJSON event store for quick retention/analysis."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, events: Iterable[Dict]) -> None:
        if not events:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
