from typing import List
from ..profiling.entity_store import EntityProfile
from ..config import HealthConfig
from .events import HealthEvent, severity_allows


def _recent_two_observations(profile: EntityProfile) -> List:
    if len(profile.observations) < 2:
        return []
    return profile.observations[-2:]


def fall_events(profile: EntityProfile, cfg: HealthConfig, now_ts: float) -> List[HealthEvent]:
    """Heuristic fall detection using soft biometrics over last two observations.

    Signals used:
    - height drop (soft_biometrics[0])
    - aspect ratio increase (soft_biometrics[1]) from tall to wide
    - area change (soft_biometrics[2])
    - short time window between observations
    """
    obs_pair = _recent_two_observations(profile)
    if not obs_pair:
        return []

    first, last = obs_pair[0], obs_pair[1]
    dt = last.timestamp - first.timestamp
    if dt <= 0 or dt > cfg.fall_time_window_seconds:
        return []

    sb0 = first.fused_features.soft_biometrics
    sb1 = last.fused_features.soft_biometrics
    if sb0 is None or sb1 is None or len(sb0) < 3 or len(sb1) < 3:
        return []

    h0, ar0, area0 = float(sb0[0]), float(sb0[1]), float(sb0[2])
    h1, ar1, area1 = float(sb1[0]), float(sb1[1]), float(sb1[2])

    if h0 <= 0:
        return []

    height_ratio = h1 / h0
    aspect_increase = ar1 / max(ar0, 1e-6)
    area_increase = (area1 - area0) / max(area0, 1e-6)

    if height_ratio <= cfg.fall_height_drop_ratio and aspect_increase >= cfg.fall_aspect_ratio_increase:
        if area_increase >= cfg.fall_area_increase:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="critical",
                type="fall_suspected",
                description="Rapid change from tall to wide silhouette in short window",
                timestamp=now_ts,
                context={
                    "height_ratio": height_ratio,
                    "aspect_increase": aspect_increase,
                    "area_increase": area_increase,
                    "dt": dt,
                },
            )
            if severity_allows(ev.severity, cfg.notify_min_severity):
                return [ev]
    return []


def activity_events(profile: EntityProfile, cfg: HealthConfig, now_ts: float) -> List[HealthEvent]:
    """Heuristic activity level: count observations in a sliding window; emit high-activity events."""
    window = cfg.activity_window_seconds
    if window <= 0:
        return []
    recent = [obs for obs in profile.observations if now_ts - obs.timestamp <= window]
    if len(recent) >= cfg.high_activity_count_threshold:
        ev = HealthEvent(
            entity_id=profile.entity_id,
            severity="info",
            type="high_activity",
            description=f"{len(recent)} observations in last {int(window)}s",
            timestamp=now_ts,
            context={"count": len(recent), "window_seconds": window},
        )
        if severity_allows(ev.severity, cfg.notify_min_severity):
            return [ev]
    return []


def fall_and_activity_events(profile: EntityProfile, cfg: HealthConfig, now_ts: float) -> List[HealthEvent]:
    events: List[HealthEvent] = []
    events.extend(fall_events(profile, cfg, now_ts))
    events.extend(activity_events(profile, cfg, now_ts))
    return events
