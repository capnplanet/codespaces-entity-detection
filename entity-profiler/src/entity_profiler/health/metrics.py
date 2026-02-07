from collections import defaultdict
from typing import Dict, Tuple
import numpy as np

from ..profiling.entity_store import EntityProfile
from ..gait.gait_features import gait_speed_mean, gait_speed_std


def last_seen_seconds(profile: EntityProfile, now_ts: float) -> float:
    if not profile.observations:
        return float("inf")
    latest = max(obs.timestamp for obs in profile.observations)
    return float(max(0.0, now_ts - latest))


def presence_histograms(profile: EntityProfile) -> Tuple[Dict[str, int], Dict[int, int]]:
    by_camera = defaultdict(int)
    by_hour = defaultdict(int)
    for obs in profile.observations:
        by_camera[obs.camera_id] += 1
        hour = int((obs.timestamp % 86400) // 3600)
        by_hour[hour] += 1
    return dict(by_camera), dict(by_hour)


def gait_speed_stats(profile: EntityProfile) -> Tuple[float, float]:
    """Approximate gait speed from fused gait feature if present.

    Uses helper accessors from gait_features to avoid hard-coded indices and
    to remain resilient to future layout extensions.
    """

    speeds = []
    for obs in profile.observations:
        g = obs.fused_features.gait
        if g is None or g.size == 0:
            continue
        s = gait_speed_mean(g)
        if s > 0.0:
            speeds.append(s)
    if not speeds:
        return 0.0, 0.0
    arr = np.array(speeds, dtype=np.float32)
    return float(arr.mean()), float(arr.std())
