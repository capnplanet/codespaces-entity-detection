from entity_profiler.profiling.entity_store import EntityStore
from entity_profiler.features.fusion import FusedFeatures
import numpy as np


def test_entity_creation_and_observation():
    store = EntityStore()
    profile = store.create_entity()
    fused = FusedFeatures(
        gait=np.zeros(10, dtype=np.float32),
        soft_biometrics=np.zeros(3, dtype=np.float32),
        clothing=np.zeros(24, dtype=np.float32),
    )
    store.add_observation(profile.entity_id, timestamp=0.0, camera_id="cam01", fused=fused)
    assert len(profile.observations) == 1


def test_entity_store_persistence_roundtrip(tmp_path):
    store = EntityStore()
    profile = store.create_entity()
    fused = FusedFeatures(
        gait=np.arange(10, dtype=np.float32),
        soft_biometrics=np.ones(3, dtype=np.float32),
        clothing=np.zeros(24, dtype=np.float32),
    )
    store.add_observation(profile.entity_id, timestamp=123.0, camera_id="cam02", fused=fused)

    out_file = tmp_path / "entity_store.json"
    store.save_json(out_file)

    loaded = EntityStore.load_json(out_file)
    profiles = loaded.get_all_profiles()
    assert len(profiles) == 1
    loaded_obs = profiles[0].observations
    assert len(loaded_obs) == 1
    assert np.allclose(loaded_obs[0].fused_features.gait, fused.gait)
    assert loaded_obs[0].camera_id == "cam02"
