from dataclasses import dataclass
from typing import List, Dict, Any
import time

from ..config import SafetyConfig
from ..profiling.entity_store import EntityStore
from ..utils.time_utils import hour_of_day


@dataclass
class SafetyEvent:
    entity_id: str
    severity: str  # info|warning|critical
    type: str
    description: str
    timestamp: float
    context: Dict[str, Any]


_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def _severity_allows(emitted: str, minimum: str) -> bool:
    return _SEVERITY_ORDER.get(emitted, 0) >= _SEVERITY_ORDER.get(minimum, 0)


def _hour_in_window(hour: int, start: int, end: int) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _camera_span_seconds(profile, camera_id: str) -> float:
    times = [obs.timestamp for obs in profile.observations if obs.camera_id == camera_id]
    if len(times) < 2:
        return 0.0
    return float(max(times) - min(times))


def _recent_count(profile, now_ts: float, window_seconds: float) -> int:
    return sum(1 for obs in profile.observations if now_ts - obs.timestamp <= window_seconds)


def evaluate_safety_events(store: EntityStore, cfg: SafetyConfig, now_ts: float | None = None) -> List[SafetyEvent]:
    now_ts = time.time() if now_ts is None else now_ts
    events: List[SafetyEvent] = []

    for profile in store.get_all_profiles():
        if not profile.observations:
            continue

        last_obs = profile.observations[-1]
        last_hour = hour_of_day(last_obs.timestamp)
        area_meta = cfg.areas.get(last_obs.camera_id, {}) if cfg.areas else {}
        last_risk = area_meta.get("risk", "").lower()

        # Rule: motion during quiet hours on perimeter cameras
        q_start, q_end = cfg.quiet_hours
        if _hour_in_window(last_hour, q_start, q_end) and (
            not cfg.quiet_hours_cameras or last_obs.camera_id in cfg.quiet_hours_cameras
        ):
            severity = "critical" if last_risk == "high" else "warning"
            ev = SafetyEvent(
                entity_id=profile.entity_id,
                severity=severity,
                type="quiet_hours_motion",
                description=f"Motion on {last_obs.camera_id} during quiet hours {q_start}-{q_end}h",
                timestamp=now_ts,
                context={"camera_id": last_obs.camera_id, "hour": last_hour, "risk": last_risk},
            )
            if _severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

        # Rule: lingering on a camera beyond threshold
        seen_cameras = {obs.camera_id for obs in profile.observations}
        for cam_id in seen_cameras:
            span = _camera_span_seconds(profile, cam_id)
            if span >= cfg.linger_seconds:
                cam_meta = cfg.areas.get(cam_id, {}) if cfg.areas else {}
                risk = cam_meta.get("risk", "").lower()
                severity = "critical" if risk == "high" else "warning"
                ev = SafetyEvent(
                    entity_id=profile.entity_id,
                    severity=severity,
                    type="linger_detected",
                    description=f"Entity stayed near {cam_id} for {span:.1f}s",
                    timestamp=now_ts,
                    context={"camera_id": cam_id, "span_seconds": span, "risk": risk},
                )
                if _severity_allows(ev.severity, cfg.notify_min_severity):
                    events.append(ev)

        # Rule: burst of activity in a short window
        recent = _recent_count(profile, now_ts, cfg.burst_window_seconds)
        if recent >= cfg.burst_count_threshold:
            severity = "warning" if last_risk == "high" else "info"
            ev = SafetyEvent(
                entity_id=profile.entity_id,
                severity=severity,
                type="burst_activity",
                description=f"{recent} observations in last {cfg.burst_window_seconds:.0f}s",
                timestamp=now_ts,
                context={"count": recent, "window_seconds": cfg.burst_window_seconds, "camera_id": last_obs.camera_id},
            )
            if _severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

    return events
