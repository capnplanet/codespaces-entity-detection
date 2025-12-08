from dataclasses import dataclass
from typing import List, Dict, Any
import time

from ..config import HealthConfig
from ..profiling.entity_store import EntityStore, EntityProfile
from .metrics import last_seen_seconds, presence_histograms, gait_speed_stats


@dataclass
class HealthEvent:
    entity_id: str
    severity: str  # info|warning|critical
    type: str
    description: str
    timestamp: float
    context: Dict[str, Any]


_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def _severity_allows(emitted: str, minimum: str) -> bool:
    return _SEVERITY_ORDER.get(emitted, 0) >= _SEVERITY_ORDER.get(minimum, 0)


def evaluate_health_events(store: EntityStore, cfg: HealthConfig, now_ts: float | None = None) -> List[HealthEvent]:
    now_ts = time.time() if now_ts is None else now_ts
    events: List[HealthEvent] = []

    for profile in store.get_all_profiles():
        if not profile.observations:
            continue

        # Rule: no recent activity
        idle_seconds = last_seen_seconds(profile, now_ts)
        if idle_seconds > cfg.no_activity_hours * 3600:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="critical",
                type="no_recent_activity",
                description=f"No observations for {idle_seconds/3600:.1f}h",
                timestamp=now_ts,
                context={"idle_seconds": idle_seconds},
            )
            if _severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

        # Rule: night-time activity above threshold
        cam_hist, hour_hist = presence_histograms(profile)
        night_start, night_end = cfg.night_hours
        night_count = sum(v for h, v in hour_hist.items() if night_start <= h < night_end)
        if night_count >= cfg.night_activity_threshold:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="warning",
                type="night_activity",
                description=f"{night_count} observations during night window {night_start}-{night_end}h",
                timestamp=now_ts,
                context={"night_count": night_count, "window": (night_start, night_end)},
            )
            if _severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

        # Rule: low mobility
        speed_mean, _ = gait_speed_stats(profile)
        if speed_mean and speed_mean < cfg.low_mobility_speed_threshold:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="warning",
                type="low_mobility",
                description=f"Mean gait speed proxy {speed_mean:.3f} below threshold",
                timestamp=now_ts,
                context={"speed_mean": speed_mean, "threshold": cfg.low_mobility_speed_threshold},
            )
            if _severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

    return events
