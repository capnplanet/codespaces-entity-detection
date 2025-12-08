import numpy as np
from entity_profiler.profiling.entity_store import EntityStore, EntityProfile, Observation
from entity_profiler.features.fusion import FusedFeatures
from entity_profiler.health.rules import evaluate_health_events
from entity_profiler.config import HealthConfig


def _make_obs(ts: float, cam: str) -> Observation:
    fused = FusedFeatures(
        gait=np.array([0, 0, 0, 0, 0.01, 0.0, 0, 0, 0, 0], dtype=np.float32),
        soft_biometrics=np.zeros(3, dtype=np.float32),
        clothing=np.zeros(24, dtype=np.float32),
    )
    return Observation(entity_id="e1", timestamp=ts, camera_id=cam, fused_features=fused)


def test_health_rules_idle_and_low_mobility():
    store = EntityStore()
    profile = store.create_entity()
    profile.observations.append(_make_obs(ts=0.0, cam="cam01"))

    cfg = HealthConfig(no_activity_hours=0.001, low_mobility_speed_threshold=0.05)
    events = evaluate_health_events(store, cfg, now_ts=10.0)
    types = {e.type for e in events}
    assert "no_recent_activity" in types
    assert "low_mobility" in types
