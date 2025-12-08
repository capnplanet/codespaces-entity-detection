from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class WearableSample:
    device_id: str
    timestamp: float
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    raw: Optional[dict] = None


class WearableBuffer:
    """In-memory buffer for recent wearable samples keyed by device_id."""

    def __init__(self, ttl_seconds: float = 7200.0):
        self.ttl_seconds = ttl_seconds
        self._samples: Dict[str, List[WearableSample]] = {}

    def add_samples(self, samples: List[WearableSample]) -> None:
        for s in samples:
            bucket = self._samples.setdefault(s.device_id, [])
            bucket.append(s)
            # prune old samples per device relative to the newest sample time to stay test/deployment agnostic
            cutoff = s.timestamp - self.ttl_seconds
            self._samples[s.device_id] = [sm for sm in bucket if sm.timestamp >= cutoff]

    def query(self, device_id: str, start_ts: float, end_ts: float) -> List[WearableSample]:
        bucket = self._samples.get(device_id, [])
        return [s for s in bucket if start_ts <= s.timestamp <= end_ts]
