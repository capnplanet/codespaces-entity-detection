from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json


@dataclass
class RecordingSegment:
    camera_id: str
    path: str  # filesystem path to segment file
    start_ts: float
    end_ts: float


class RecordingIndex:
    """Append-only NDJSON index of recording segments.

    This lightweight index is sufficient for small deployments. For larger
    systems, a database-backed index is recommended.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, segments: Iterable[RecordingSegment]) -> None:
        seg_list = list(segments)
        if not seg_list:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for seg in seg_list:
                f.write(json.dumps(asdict(seg)) + "\n")

    def find_segments(self, camera_id: str, start_ts: float, end_ts: float) -> List[RecordingSegment]:
        """Return segments overlapping [start_ts, end_ts] for a camera.

        This performs a linear scan over the NDJSON index, which is acceptable
        for small installations.
        """

        results: List[RecordingSegment] = []
        if not self.path.exists():
            return results
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("camera_id") != camera_id:
                    continue
                seg_start = float(data.get("start_ts", 0.0))
                seg_end = float(data.get("end_ts", 0.0))
                if seg_end < start_ts or seg_start > end_ts:
                    continue
                results.append(
                    RecordingSegment(
                        camera_id=str(data.get("camera_id")),
                        path=str(data.get("path")),
                        start_ts=seg_start,
                        end_ts=seg_end,
                    )
                )
        return results

    def find_best_segment_for_event(
        self,
        camera_id: str,
        event_ts: float,
        pre_seconds: float = 15.0,
        post_seconds: float = 15.0,
    ) -> Optional[Dict[str, float | str]]:
        """Return a best-effort clip suggestion for an event.

        The clip is defined by a segment path and a recommended window within
        that segment around the event time.
        """

        window_start = event_ts - pre_seconds
        window_end = event_ts + post_seconds
        segments = self.find_segments(camera_id, window_start, window_end)
        if not segments:
            return None

        # Choose the segment that actually contains the event time if possible,
        # otherwise the one with the closest start time.
        chosen = None
        for seg in segments:
            if seg.start_ts <= event_ts <= seg.end_ts:
                chosen = seg
                break
        if chosen is None:
            chosen = min(segments, key=lambda s: abs(s.start_ts - event_ts))

        clip_start = max(chosen.start_ts, window_start)
        clip_end = min(chosen.end_ts, window_end)
        return {
            "camera_id": chosen.camera_id,
            "segment_path": chosen.path,
            "segment_start_ts": chosen.start_ts,
            "segment_end_ts": chosen.end_ts,
            "clip_start_ts": clip_start,
            "clip_end_ts": clip_end,
        }
