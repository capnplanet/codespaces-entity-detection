"""Scenario-based evaluation for health and safety rules.

This script is designed to consume pre-built timelines of observations and
optional wearable samples annotated with expected rule outcomes.

It mirrors the structure used in unit tests under tests/, but operates on
JSON/NDJSON files so that larger synthetic or semi-synthetic scenarios can be
used.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Dict, Iterable, List, Set

from entity_profiler.config import HealthConfig, SafetyConfig
from entity_profiler.features.fusion import FusedFeatures
from entity_profiler.health.events import HealthEvent
from entity_profiler.health.rules import evaluate_health_events_with_wearables
from entity_profiler.health.wearables import WearableBuffer, WearableSample
from entity_profiler.profiling.entity_store import EntityProfile, Observation
from entity_profiler.safety.rules import SafetyEvent, evaluate_safety_events


@dataclasses.dataclass
class ScenarioConfig:
    """Config for health/safety scenario evaluation.

    Attributes
    ----------
    observations_path: NDJSON of Observation-like dicts.
    wearables_path: Optional NDJSON of WearableSample-like dicts.
    labels_path: NDJSON of label dicts with expected events per scenario.
    health_config_path: JSON config for HealthConfig.
    safety_config_path: JSON config for SafetyConfig.
    """

    observations_path: pathlib.Path
    labels_path: pathlib.Path
    health_config_path: pathlib.Path
    safety_config_path: pathlib.Path
    wearables_path: pathlib.Path | None = None


@dataclasses.dataclass
class ScenarioLabel:
    """Single expected event label.

    Attributes
    ----------
    category: "health" or "safety".
    type: event type string.
    entity_id: optional entity identifier.
    camera_id: optional camera identifier.
    """

    category: str
    type: str
    entity_id: str | None = None
    camera_id: str | None = None


def load_config(path: str | pathlib.Path) -> ScenarioConfig:
    data = json.loads(pathlib.Path(path).read_text())
    return ScenarioConfig(
        observations_path=pathlib.Path(data["observations_path"]),
        labels_path=pathlib.Path(data["labels_path"]),
        health_config_path=pathlib.Path(data["health_config_path"]),
        safety_config_path=pathlib.Path(data["safety_config_path"]),
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


def load_labels(path: pathlib.Path) -> List[ScenarioLabel]:
    labels: List[ScenarioLabel] = []
    for obj in _iter_json_lines(path):
        labels.append(
            ScenarioLabel(
                category=obj["category"],
                type=obj["type"],
                entity_id=obj.get("entity_id"),
                camera_id=obj.get("camera_id"),
            )
        )
    return labels


def load_health_config(path: pathlib.Path) -> HealthConfig:
    data = json.loads(path.read_text())
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


def load_safety_config(path: pathlib.Path) -> SafetyConfig:
    data = json.loads(path.read_text())
    allowed = {
        "quiet_hours",
        "quiet_hours_cameras",
        "linger_seconds",
        "burst_window_seconds",
        "burst_count_threshold",
        "notify_min_severity",
        "notification_targets",
        "areas",
    }
    filtered = {k: v for k, v in data.items() if k in allowed}
    return SafetyConfig(**filtered)


def _event_signature(event_type: str, entity_id: str | None, camera_id: str | None) -> str:
    return "::".join([
        event_type,
        entity_id or "*",
        camera_id or "*",
    ])


def evaluate(config: ScenarioConfig) -> dict:
    store = load_observations(config.observations_path)
    wearable_buffer = load_wearables(config.wearables_path)
    labels = load_labels(config.labels_path)
    health_cfg = load_health_config(config.health_config_path)
    safety_cfg = load_safety_config(config.safety_config_path)

    all_obs = [obs for profile in store.get_all_profiles() for obs in profile.observations]
    if not all_obs:
        return {"num_labels": len(labels), "num_health_events": 0, "num_safety_events": 0}

    now_ts = max(o.timestamp for o in all_obs)

    health_events: List[HealthEvent] = list(
        evaluate_health_events_with_wearables(
            store=store,
            cfg=health_cfg,
            wearable_buffer=wearable_buffer,
            now_ts=now_ts,
        )
    )
    safety_events: List[SafetyEvent] = list(
        evaluate_safety_events(
            store=store,
            cfg=safety_cfg,
            now_ts=now_ts,
        )
    )

    # Build sets of event signatures for simple presence/absence comparisons.
    health_sig: Set[str] = set(
        _event_signature(e.type, getattr(e, "entity_id", None), getattr(e, "camera_id", None)) for e in health_events
    )
    safety_sig: Set[str] = set(
        _event_signature(e.type, getattr(e, "entity_id", None), getattr(e, "camera_id", None)) for e in safety_events
    )

    tp = 0
    fn = 0
    fp_health = 0
    fp_safety = 0

    for label in labels:
        sig = _event_signature(label.type, label.entity_id, label.camera_id)
        if label.category == "health":
            if sig in health_sig:
                tp += 1
            else:
                fn += 1
        elif label.category == "safety":
            if sig in safety_sig:
                tp += 1
            else:
                fn += 1

    # Count any events that were not labeled as false positives.
    label_sigs_health: Set[str] = {
        _event_signature(l.type, l.entity_id, l.camera_id) for l in labels if l.category == "health"
    }
    label_sigs_safety: Set[str] = {
        _event_signature(l.type, l.entity_id, l.camera_id) for l in labels if l.category == "safety"
    }

    fp_health = len(health_sig - label_sigs_health)
    fp_safety = len(safety_sig - label_sigs_safety)

    precision_health = float(tp / (tp + fp_health)) if (tp + fp_health) > 0 else 0.0
    precision_safety = float(tp / (tp + fp_safety)) if (tp + fp_safety) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "num_labels": len(labels),
        "num_health_events": len(health_events),
        "num_safety_events": len(safety_events),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives_health": fp_health,
        "false_positives_safety": fp_safety,
        "precision_health": precision_health,
        "precision_safety": precision_safety,
        "recall": recall,
    }


@dataclasses.dataclass
class _EvalStore:
    """Lightweight EntityStore-like wrapper for evaluation.

    Only ``get_all_profiles`` is required by the rule engines.
    """

    profiles: Dict[str, EntityProfile]

    def get_all_profiles(self) -> List[EntityProfile]:
        return list(self.profiles.values())


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entity Profiler health/safety scenario benchmark")
    parser.add_argument("--config", required=True, help="Path to JSON config for health/safety benchmark")
    parser.add_argument("--output", required=True, help="Path to JSON file for results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = evaluate(cfg)

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
