from typing import List
import math

from .events import HealthEvent, severity_allows
from ..profiling.entity_store import EntityProfile
from ..config import HealthConfig


def _recent_observations(profile: EntityProfile, k: int = 3):
    return profile.observations[-k:] if len(profile.observations) >= 2 else []


def _soft_metrics(obs) -> tuple:
    sb = obs.fused_features.soft_biometrics
    if sb is None or len(sb) < 3:
        return 0.0, 0.0, 0.0
    return float(sb[0]), float(sb[1]), float(sb[2])


def _gait_speed(obs) -> float:
    gait = obs.fused_features.gait
    if gait is None or len(gait) < 6:
        return 0.0
    return float(gait[4])  # speed_mean


def fall_model_events(profile: EntityProfile, cfg: HealthConfig, now_ts: float) -> List[HealthEvent]:
    obs = _recent_observations(profile)
    if len(obs) < 2:
        return []

    first, last = obs[-2], obs[-1]
    dt = last.timestamp - first.timestamp
    if dt <= 0 or dt > cfg.fall_time_window_seconds:
        return []

    h0, ar0, area0 = _soft_metrics(first)
    h1, ar1, area1 = _soft_metrics(last)
    if h0 <= 0:
        return []

    height_ratio = h1 / h0
    aspect_increase = ar1 / max(ar0, 1e-6)
    area_increase = (area1 - area0) / max(area0, 1e-6)

    s0 = _gait_speed(first)
    s1 = _gait_speed(last)
    speed_delta = s1 - s0

    # Simple fused score: penalize if height drops and aspect widens; boost if speed spikes.
    # Clamp components to stabilize.
    height_term = max(0.0, min(1.0, (cfg.fall_height_drop_ratio - height_ratio) / cfg.fall_height_drop_ratio))
    aspect_term = max(0.0, min(1.0, (aspect_increase - cfg.fall_aspect_ratio_increase) / cfg.fall_aspect_ratio_increase))
    speed_term = max(0.0, min(1.0, speed_delta / max(cfg.fall_speed_delta, 1e-6)))
    area_term = max(0.0, min(1.0, area_increase / max(cfg.fall_area_increase, 1e-6)))

    fused_score = 0.4 * height_term + 0.3 * aspect_term + 0.2 * speed_term + 0.1 * area_term

    if fused_score >= cfg.fall_score_threshold:
        ev = HealthEvent(
            entity_id=profile.entity_id,
            severity="critical",
            type="fall_model_suspected",
            description="Fall model detected rapid posture change",
            timestamp=now_ts,
            context={
                "height_ratio": height_ratio,
                "aspect_increase": aspect_increase,
                "area_increase": area_increase,
                "speed_delta": speed_delta,
                "dt": dt,
                "score": fused_score,
            },
        )
        if severity_allows(ev.severity, cfg.notify_min_severity):
            return [ev]
    return []
