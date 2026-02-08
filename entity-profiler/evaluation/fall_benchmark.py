"""Fall detection benchmark for Entity Profiler.

This script replays precomputed observations and optional wearable samples
through the health rule engine to evaluate fall-related events.

It operates purely on JSON/NDJSON event timelines, so external datasets can be
adapted into the expected observation format without changing core code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Dict, Iterable, List

from entity_profiler.config import HealthConfig
from entity_profiler.features.fusion import FusedFeatures
from entity_profiler.health.events import HealthEvent
from entity_profiler.health.rules import evaluate_health_events_with_wearables
from entity_profiler.health.wearables import WearableBuffer, WearableSample
from entity_profiler.profiling.entity_store import EntityProfile, Observation


@dataclasses.dataclass
class FallBenchmarkConfig:
    """Config for fall benchmark.

    Attributes
    ----------
    observations_path: NDJSON or JSONL file of Observation-like dicts.
    wearables_path: Optional NDJSON/JSONL file of WearableSample-like dicts.
    labels_path: JSON/NDJSON file with ground-truth fall events.
    health_config_path: JSON health config compatible with HealthConfig.
    """

    observations_path: pathlib.Path
    labels_path: pathlib.Path
    health_config_path: pathlib.Path
    wearables_path: pathlib.Path | None = None


@dataclasses.dataclass
class FallLabel:
    """Ground truth fall event for a given entity.

    Attributes
    ----------
    entity_id: pseudonymous entity identifier.
    timestamp: POSIX seconds.
    """

    entity_id: str
    timestamp: float


def load_config(path: str | pathlib.Path) -> FallBenchmarkConfig:
    data = json.loads(pathlib.Path(path).read_text())
    return FallBenchmarkConfig(
        observations_path=pathlib.Path(data["observations_path"]),
        labels_path=pathlib.Path(data["labels_path"]),
        health_config_path=pathlib.Path(data["health_config_path"]),
        wearables_path=pathlib.Path(data["wearables_path"]) if data.get("wearables_path") else None,
    )


def _iter_json_lines(path: pathlib.Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_observations(path: pathlib.Path) -> EntityStore:
    profiles: Dict[str, EntityProfile] = {}
    for obj in _iter_json_lines(path):
        entity_id = str(obj["entity_id"])
        timestamp = float(obj["timestamp"])
        camera_id = str(obj["camera_id"])
        fused_payload = obj.get("fused_features", {}) or {}
        fused = FusedFeatures.from_dict(fused_payload)

        profile = profiles.get(entity_id)
        if profile is None:
            profile = EntityProfile(entity_id=entity_id)
            profiles[entity_id] = profile

        obs = Observation(
            entity_id=entity_id,
            timestamp=timestamp,
            camera_id=camera_id,
            fused_features=fused,
        )
        profile.observations.append(obs)
    return _EvalStore(profiles)


def load_wearables(path: pathlib.Path | None) -> WearableBuffer | None:
    if path is None:
        return None
    buf = WearableBuffer()
    for obj in _iter_json_lines(path):
        sample = WearableSample(
            device_id=obj["device_id"],
            timestamp=float(obj["timestamp"]),
            heart_rate=float(obj["heart_rate"]),
            spo2=float(obj.get("spo2")) if obj.get("spo2") is not None else None,
            raw=obj.get("raw"),
        )
        buf.add_sample(sample)
    return buf


def load_labels(path: pathlib.Path) -> List[FallLabel]:
    labels: List[FallLabel] = []
    for obj in _iter_json_lines(path):
        labels.append(FallLabel(entity_id=obj["entity_id"], timestamp=float(obj["timestamp"])))
    return labels


def load_health_config(path: pathlib.Path) -> HealthConfig:
    data = json.loads(path.read_text())
    # Only keep keys that HealthConfig actually knows about; this allows
    # simple test configs without specifying every field.
    allowed = {
        "no_activity_hours",
        "night_hours",
        "night_activity_threshold",
        "low_mobility_speed_threshold",
        "notify_min_severity",
        "notification_targets",
        "areas",
        "hr_high",
        "hr_low",
        "spo2_low",
        "wearable_window_seconds",
        "wearable_idle_grace_seconds",
        "wearables",
        "fall_height_drop_ratio",
        "fall_aspect_ratio_increase",
        "fall_area_increase",
        "fall_time_window_seconds",
        "fall_speed_delta",
        "fall_score_threshold",
        "activity_window_seconds",
        "high_activity_count_threshold",
    }
    filtered = {k: v for k, v in data.items() if k in allowed}
    return HealthConfig(**filtered)


def evaluate(config: FallBenchmarkConfig) -> dict:
    store = load_observations(config.observations_path)
    wearable_buffer = load_wearables(config.wearables_path)
    labels = load_labels(config.labels_path)
    health_cfg = load_health_config(config.health_config_path)

    all_obs = [obs for profile in store.get_all_profiles() for obs in profile.observations]
    if not all_obs:
        return {
            "num_labels": len(labels),
            "num_events": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(labels),
        }

    # Use the last observation timestamp as "now" for evaluation.
    now_ts = max(o.timestamp for o in all_obs)

    events: List[HealthEvent] = list(
        evaluate_health_events_with_wearables(
            store=store,
            cfg=health_cfg,
            wearable_buffer=wearable_buffer,
            now_ts=now_ts,
        )
    )

    # Only consider fall-related event types for this benchmark.
    fall_types = {"fall_suspected", "fall_model_suspected"}
    fall_events = [e for e in events if e.type in fall_types]

    # Simple matching strategy: for each label, check if any fall event for the
    # same entity occurs within a fixed window around the label timestamp.
    window = 5.0  # seconds

    tp = 0
    fn = 0
    matched_events = set()

    for label in labels:
        matched = False
        for idx, ev in enumerate(fall_events):
            if idx in matched_events:
                continue
            if ev.entity_id != label.entity_id:
                continue
            if abs(ev.timestamp - label.timestamp) <= window:
                tp += 1
                matched_events.add(idx)
                matched = True
                break
        if not matched:
            fn += 1

    fp = len(fall_events) - len(matched_events)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "num_labels": len(labels),
        "num_events": len(fall_events),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "window_seconds": window,
    }


@dataclasses.dataclass
class _EvalStore:
    """Lightweight EntityStore-like wrapper for evaluation.

    It implements only ``get_all_profiles``, which is all the health rule
    engine requires. This avoids coupling the benchmark to the production
    EntityStore persistence format.
    """

    profiles: Dict[str, EntityProfile]

    def get_all_profiles(self) -> List[EntityProfile]:
        return list(self.profiles.values())


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entity Profiler fall benchmark")
    parser.add_argument("--config", required=True, help="Path to JSON config for fall benchmark")
    parser.add_argument("--output", required=True, help="Path to JSON file for results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = evaluate(cfg)

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
