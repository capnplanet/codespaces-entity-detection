import numpy as np
from entity_profiler.profiling.entity_store import EntityStore, Observation
from entity_profiler.features.fusion import FusedFeatures
from entity_profiler.safety.rules import evaluate_safety_events
from entity_profiler.config import SafetyConfig


def _make_obs(ts: float, cam: str) -> Observation:
    fused = FusedFeatures(
        gait=np.zeros(10, dtype=np.float32),
        soft_biometrics=np.zeros(3, dtype=np.float32),
        clothing=np.zeros(24, dtype=np.float32),
    )
    return Observation(entity_id="e1", timestamp=ts, camera_id=cam, fused_features=fused)


def test_safety_rules_quiet_and_linger():
    store = EntityStore()
    profile = store.create_entity()
    profile.observations.append(_make_obs(ts=3600.0, cam="door_cam"))
    profile.observations.append(_make_obs(ts=3800.0, cam="door_cam"))

    cfg = SafetyConfig(quiet_hours=(23, 6), linger_seconds=150, burst_window_seconds=300, burst_count_threshold=2)
    events = evaluate_safety_events(store, cfg, now_ts=4000.0)
    types = {e.type for e in events}
    assert "quiet_hours_motion" in types
    assert "linger_detected" in types
