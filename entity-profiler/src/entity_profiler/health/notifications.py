from pathlib import Path
from typing import List
import json
import urllib.request

from .rules import HealthEvent


class Notifier:
    def send(self, event: HealthEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LogNotifier(Notifier):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, event: HealthEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__) + "\n")


class WebhookNotifier(Notifier):
    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url
        self.timeout = timeout

    def send(self, event: HealthEvent) -> None:
        data = json.dumps(event.__dict__).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as _:
                pass
        except Exception:
            # Swallow errors to avoid breaking pipeline; in production, log properly.
            return


def build_notifiers(targets: List, log_default_path: Path) -> List[Notifier]:
    notifiers: List[Notifier] = []
    for t in targets:
        if not isinstance(t, (list, tuple)) or len(t) < 2:
            continue
        kind, value = t[0], t[1]
        if kind == "log":
            notifiers.append(LogNotifier(Path(value)))
        elif kind == "webhook":
            notifiers.append(WebhookNotifier(str(value)))
    if not notifiers:
        # Fallback to a local log to avoid silent drops
        notifiers.append(LogNotifier(log_default_path))
    return notifiers
