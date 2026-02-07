# Data Model

This document gives a concise view of the primary data structures. For exhaustive field-level details and behaviors, see PLATFORM_CAPABILITIES.md and the `profiling/`, `health/`, `safety/`, and `security/` modules.

- **Observation** (profiling.entity_store)
  - `entity_id`
  - `timestamp`
  - `camera_id`
  - `fused_features` (93D embedding from gait, soft biometrics, clothing)

- **EntityProfile** (profiling.entity_store)
  - `entity_id`
  - `observations: List[Observation]`
  - Pattern-of-life summaries derived at query time (camera histogram, hour-of-day histogram, dominant camera/hour, span)

- **Event** (utils.event_store, health.events, safety.rules)
  - Core fields: `event_id`, `trace_id`, `category` (health/safety), `status`, `emitted_at`
  - Contextual fields: `entity_id`, `camera_id`, `device_id` (for wearables), `timestamp`, `severity`, `rule_type`, and rule-specific context payload
  - Persisted as append-only NDJSON in `data/interim/events.ndjson`

- **WearableSample** (health.wearables)
  - `device_id`
  - `timestamp`
  - `heart_rate` (optional)
  - `spo2` (optional)
  - Buffered in memory and associated with entities via configuration in `data/health_config.json`

- **Camera and Site Records** (camera.models)
  - **Site**: `site_id`, `name`, `timezone`, `metadata`
  - **Camera**: `camera_id`, `site_id`, `name`, `rtsp_url`, `location`, `risk_level`, `enabled`, `metadata`
  - Stored in JSON under `data/interim/camera_registry.json`

- **User and Role Records** (security.models)
  - **User**: `user_id`, `username`, `roles` (e.g., VIEWER, ADMIN), `metadata`
  - Backed by `data/interim/users.json` with audit events in `data/interim/audit.ndjson`

Entities and users are pseudonymous within this system and may only be mapped to real-world identities by external systems, subject to policy and law.
