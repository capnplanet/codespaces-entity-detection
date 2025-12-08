from typing import List
import time

from ..config import HealthConfig
from ..profiling.entity_store import EntityStore, EntityProfile
from .metrics import last_seen_seconds, presence_histograms, gait_speed_stats
from .wearables import WearableBuffer
from .fall_activity import fall_and_activity_events
from .fall_model import fall_model_events
from .events import HealthEvent, severity_allows


def evaluate_health_events(store: EntityStore, cfg: HealthConfig, now_ts: float | None = None) -> List[HealthEvent]:
    now_ts = time.time() if now_ts is None else now_ts
    events: List[HealthEvent] = []

    for profile in store.get_all_profiles():
        if not profile.observations:
            continue

        # Rule: no recent activity
        idle_seconds = last_seen_seconds(profile, now_ts)
        if idle_seconds > cfg.no_activity_hours * 3600:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="critical",
                type="no_recent_activity",
                description=f"No observations for {idle_seconds/3600:.1f}h",
                timestamp=now_ts,
                context={"idle_seconds": idle_seconds},
            )
            if severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

        # Rule: night-time activity above threshold
        cam_hist, hour_hist = presence_histograms(profile)
        night_start, night_end = cfg.night_hours
        night_count = sum(v for h, v in hour_hist.items() if night_start <= h < night_end)
        if night_count >= cfg.night_activity_threshold:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="warning",
                type="night_activity",
                description=f"{night_count} observations during night window {night_start}-{night_end}h",
                timestamp=now_ts,
                context={"night_count": night_count, "window": (night_start, night_end)},
            )
            if severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

        # Rule: low mobility
        speed_mean, _ = gait_speed_stats(profile)
        if speed_mean and speed_mean < cfg.low_mobility_speed_threshold:
            ev = HealthEvent(
                entity_id=profile.entity_id,
                severity="warning",
                type="low_mobility",
                description=f"Mean gait speed proxy {speed_mean:.3f} below threshold",
                timestamp=now_ts,
                context={"speed_mean": speed_mean, "threshold": cfg.low_mobility_speed_threshold},
            )
            if severity_allows(ev.severity, cfg.notify_min_severity):
                events.append(ev)

    return events


def _matching_wearable_device(cfg: HealthConfig, profile: EntityProfile) -> str | None:
    for item in cfg.wearables or []:
        device_id = item.get("device_id")
        entity_id = item.get("entity_id")
        if device_id and entity_id == profile.entity_id:
            return str(device_id)
    return None


def evaluate_health_events_with_wearables(
    store: EntityStore,
    cfg: HealthConfig,
    wearable_buffer: WearableBuffer | None = None,
    now_ts: float | None = None,
) -> List[HealthEvent]:
    now_ts = time.time() if now_ts is None else now_ts
    base_events = evaluate_health_events(store, cfg, now_ts=now_ts)
    enriched: List[HealthEvent] = list(base_events)
    window = cfg.wearable_window_seconds
    idle_grace = cfg.wearable_idle_grace_seconds

    for profile in store.get_all_profiles():
        if not profile.observations:
            continue

        # add fall/activity heuristics regardless of wearable presence
        enriched.extend(fall_and_activity_events(profile, cfg, now_ts))
        enriched.extend(fall_model_events(profile, cfg, now_ts))

        device_id = _matching_wearable_device(cfg, profile)
        if not device_id or wearable_buffer is None:
            continue

        samples = wearable_buffer.query(device_id, now_ts - window, now_ts)
        if not samples:
            continue

        hrs = [s.heart_rate for s in samples if s.heart_rate is not None]
        spo2_vals = [s.spo2 for s in samples if s.spo2 is not None]

        idle_seconds = last_seen_seconds(profile, now_ts)

        if hrs:
            hr_mean = sum(hrs) / len(hrs)
            if hr_mean >= cfg.hr_high and idle_seconds >= idle_grace:
                ev = HealthEvent(
                    entity_id=profile.entity_id,
                    severity="warning",
                    type="wearable_hr_elevated",
                    description=f"Mean HR {hr_mean:.1f} with no camera activity for {idle_seconds:.0f}s",
                    timestamp=now_ts,
                    context={"hr_mean": hr_mean, "idle_seconds": idle_seconds, "device_id": device_id},
                )
                if severity_allows(ev.severity, cfg.notify_min_severity):
                    enriched.append(ev)

            if hr_mean <= cfg.hr_low:
                ev = HealthEvent(
                    entity_id=profile.entity_id,
                    severity="warning",
                    type="wearable_hr_low",
                    description=f"Mean HR {hr_mean:.1f} below threshold",
                    timestamp=now_ts,
                    context={"hr_mean": hr_mean, "device_id": device_id},
                )
                if severity_allows(ev.severity, cfg.notify_min_severity):
                    enriched.append(ev)

        if spo2_vals:
            spo2_min = min(spo2_vals)
            if spo2_min <= cfg.spo2_low:
                ev = HealthEvent(
                    entity_id=profile.entity_id,
                    severity="critical",
                    type="wearable_spo2_low",
                    description=f"SpO2 {spo2_min:.1f}% below threshold",
                    timestamp=now_ts,
                    context={"spo2_min": spo2_min, "device_id": device_id},
                )
                if severity_allows(ev.severity, cfg.notify_min_severity):
                    enriched.append(ev)

    return enriched
