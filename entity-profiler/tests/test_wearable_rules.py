import numpy as np
from entity_profiler.profiling.entity_store import EntityStore, Observation
from entity_profiler.features.fusion import FusedFeatures
from entity_profiler.health.rules import evaluate_health_events_with_wearables
from entity_profiler.health.wearables import WearableBuffer, WearableSample
from entity_profiler.config import HealthConfig


def _make_obs(ts: float, cam: str) -> Observation:
    fused = FusedFeatures(
        gait=np.zeros(10, dtype=np.float32),
        soft_biometrics=np.zeros(3, dtype=np.float32),
        clothing=np.zeros(24, dtype=np.float32),
    )
    return Observation(entity_id="e1", timestamp=ts, camera_id=cam, fused_features=fused)


def test_wearable_hr_elevated_with_idle():
    store = EntityStore()
    profile = store.create_entity()
    profile.observations.append(_make_obs(ts=0.0, cam="cam01"))

    buffer = WearableBuffer(ttl_seconds=3600)
    buffer.add_samples([
        WearableSample(device_id="dev1", timestamp=900.0, heart_rate=120.0, spo2=95.0),
        WearableSample(device_id="dev1", timestamp=950.0, heart_rate=118.0, spo2=94.0),
    ])

    cfg = HealthConfig(
        no_activity_hours=1.0,
        wearable_window_seconds=1200,
        wearable_idle_grace_seconds=600,
        hr_high=110,
        hr_low=40,
        spo2_low=90,
        wearables=({"device_id": "dev1", "entity_id": profile.entity_id},),
    )

    events = evaluate_health_events_with_wearables(store, cfg, wearable_buffer=buffer, now_ts=1200.0)
    types = {e.type for e in events}
    assert "wearable_hr_elevated" in types
