import os
import random
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

DEFAULT_SEED = 1337


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for RNG sources to encourage deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = project_root / "data"
    raw_video_dir: Path = data_root / "raw"
    interim_dir: Path = data_root / "interim"
    processed_dir: Path = data_root / "processed"
    models_dir: Path = project_root / "models"

    def ensure(self) -> None:
        for p in [
            self.data_root,
            self.raw_video_dir,
            self.interim_dir,
            self.processed_dir,
            self.models_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ProfilingConfig:
    max_track_gap_seconds: float = 2.0
    fused_distance_threshold: float = 1.5
    min_observations_for_entity: int = 1


@dataclass
class HealthConfig:
    no_activity_hours: float = 8.0
    night_hours: tuple = (0, 6)
    night_activity_threshold: int = 5  # observations during night to flag wandering
    low_mobility_speed_threshold: float = 0.05  # approx gait speed proxy
    notify_min_severity: str = "warning"
    notification_targets: tuple = ()  # e.g., (("log", "data/interim/health_events.log"),)
    areas: dict = None  # camera_id -> {"label": str, "risk": str}
    hr_high: float = 110.0
    hr_low: float = 45.0
    spo2_low: float = 92.0
    wearable_window_seconds: float = 900.0
    wearable_idle_grace_seconds: float = 600.0
    wearables: tuple = ()  # iterable of {device_id, entity_id?, slot?, notes?}
    fall_height_drop_ratio: float = 0.35  # new_height / old_height <= ratio
    fall_aspect_ratio_increase: float = 1.8  # aspect goes from tall to wide
    fall_area_increase: float = 0.25  # relative area increase allowed
    fall_time_window_seconds: float = 4.0
    activity_window_seconds: float = 900.0
    high_activity_count_threshold: int = 8


@dataclass
class SafetyConfig:
    quiet_hours: tuple = (23, 6)  # start, end (24h); supports wrap-around
    quiet_hours_cameras: tuple = ()  # cameras considered perimeter/doors
    linger_seconds: float = 120.0  # how long someone can linger before alert
    burst_window_seconds: float = 60.0
    burst_count_threshold: int = 6
    notify_min_severity: str = "info"
    notification_targets: tuple = ()
    areas: dict = None  # camera_id -> {"label": str, "risk": str}


def load_config() -> ProfilingConfig:
    seed = int(os.getenv("EP_GLOBAL_SEED", DEFAULT_SEED))
    set_global_seed(seed)
    paths = Paths()
    paths.ensure()
    return ProfilingConfig()


def load_health_config(paths: Paths | None = None) -> HealthConfig:
    if paths is None:
        paths = Paths()
    paths.ensure()
    cfg_path = paths.data_root / "health_config.json"
    if not cfg_path.exists():
        return HealthConfig(areas={}, notification_targets=())
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return HealthConfig(areas={}, notification_targets=())

    return HealthConfig(
        no_activity_hours=float(payload.get("no_activity_hours", 8.0)),
        night_hours=tuple(payload.get("night_hours", (0, 6))),
        night_activity_threshold=int(payload.get("night_activity_threshold", 5)),
        low_mobility_speed_threshold=float(payload.get("low_mobility_speed_threshold", 0.05)),
        notify_min_severity=str(payload.get("notify_min_severity", "warning")),
        notification_targets=tuple(payload.get("notification_targets", ())),
        areas=payload.get("areas", {}) or {},
        hr_high=float(payload.get("hr_high", 110.0)),
        hr_low=float(payload.get("hr_low", 45.0)),
        spo2_low=float(payload.get("spo2_low", 92.0)),
        wearable_window_seconds=float(payload.get("wearable_window_seconds", 900.0)),
        wearable_idle_grace_seconds=float(payload.get("wearable_idle_grace_seconds", 600.0)),
        wearables=tuple(payload.get("wearables", ())),
        fall_height_drop_ratio=float(payload.get("fall_height_drop_ratio", 0.35)),
        fall_aspect_ratio_increase=float(payload.get("fall_aspect_ratio_increase", 1.8)),
        fall_area_increase=float(payload.get("fall_area_increase", 0.25)),
        fall_time_window_seconds=float(payload.get("fall_time_window_seconds", 4.0)),
        activity_window_seconds=float(payload.get("activity_window_seconds", 900.0)),
        high_activity_count_threshold=int(payload.get("high_activity_count_threshold", 8)),
    )


def load_safety_config(paths: Paths | None = None) -> SafetyConfig:
    if paths is None:
        paths = Paths()
    paths.ensure()
    cfg_path = paths.data_root / "safety_config.json"
    if not cfg_path.exists():
        return SafetyConfig(areas={}, notification_targets=())
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return SafetyConfig(areas={}, notification_targets=())

    return SafetyConfig(
        quiet_hours=tuple(payload.get("quiet_hours", (23, 6))),
        quiet_hours_cameras=tuple(payload.get("quiet_hours_cameras", ())),
        linger_seconds=float(payload.get("linger_seconds", 120.0)),
        burst_window_seconds=float(payload.get("burst_window_seconds", 60.0)),
        burst_count_threshold=int(payload.get("burst_count_threshold", 6)),
        notify_min_severity=str(payload.get("notify_min_severity", "info")),
        notification_targets=tuple(payload.get("notification_targets", ())),
        areas=payload.get("areas", {}) or {},
    )
