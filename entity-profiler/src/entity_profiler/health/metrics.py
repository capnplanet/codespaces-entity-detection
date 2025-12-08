from collections import defaultdict
from typing import Dict, Tuple
import numpy as np

from ..profiling.entity_store import EntityProfile


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

    gait_feature_from_sequence encodes speed_mean at index 4 and speed_std at index 5.
    """
    speeds = []
    for obs in profile.observations:
        g = obs.fused_features.gait
        if g.size >= 6:
            speeds.append(float(g[4]))
    if not speeds:
        return 0.0, 0.0
    arr = np.array(speeds, dtype=np.float32)
    return float(arr.mean()), float(arr.std())
