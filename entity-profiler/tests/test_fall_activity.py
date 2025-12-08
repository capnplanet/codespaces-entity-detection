import numpy as np
from entity_profiler.profiling.entity_store import EntityStore, Observation
from entity_profiler.features.fusion import FusedFeatures
from entity_profiler.health.rules import evaluate_health_events_with_wearables
from entity_profiler.config import HealthConfig


def _make_obs(ts: float, cam: str, height: float, aspect: float, area: float) -> Observation:
    fused = FusedFeatures(
        gait=np.zeros(10, dtype=np.float32),
        soft_biometrics=np.array([height, aspect, area], dtype=np.float32),
        clothing=np.zeros(24, dtype=np.float32),
    )
    return Observation(entity_id="e1", timestamp=ts, camera_id=cam, fused_features=fused)


def test_fall_detection_triggers():
    store = EntityStore()
    profile = store.create_entity()
    profile.observations.append(_make_obs(ts=0.0, cam="cam01", height=200.0, aspect=0.4, area=8000.0))
    profile.observations.append(_make_obs(ts=2.0, cam="cam01", height=60.0, aspect=1.2, area=12000.0))

    cfg = HealthConfig(
        fall_height_drop_ratio=0.4,
        fall_aspect_ratio_increase=1.5,
        fall_area_increase=0.2,
        fall_time_window_seconds=4.0,
    )

    events = evaluate_health_events_with_wearables(store, cfg, wearable_buffer=None, now_ts=2.0)
    types = {e.type for e in events}
    assert "fall_suspected" in types


def test_high_activity_emits_info():
    store = EntityStore()
    profile = store.create_entity()
    for i in range(10):
        profile.observations.append(_make_obs(ts=i * 10.0, cam="cam01", height=180.0, aspect=0.5, area=9000.0))

    cfg = HealthConfig(
        activity_window_seconds=200.0,
        high_activity_count_threshold=8,
        notify_min_severity="info",
    )

    events = evaluate_health_events_with_wearables(store, cfg, wearable_buffer=None, now_ts=200.0)
    types = {e.type for e in events}
    assert "high_activity" in types
