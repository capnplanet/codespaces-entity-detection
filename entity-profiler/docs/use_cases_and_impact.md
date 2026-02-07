# Use Cases and Impact

This document describes example deployment scenarios for Entity Profiler and how its capabilities map to practical defense and commercial use cases, given the current API, eventing, and configuration surfaces.

## 1. Home and Residential Elder Care

**Scenario**: A resident living alone or in assisted living is monitored using low-resolution indoor cameras and an optional wearable device.

**Goals**:
- Detect falls and prolonged inactivity.
- Detect unusual night-time activity (e.g., wandering, restlessness).
- Provide caregivers with traceable alerts and simple pattern-of-life summaries.

**Relevant capabilities**:
- Fall heuristics and fall model (vision-based) using soft biometrics and gait dynamics.
- No-recent-activity and low-mobility health rules.
- Night-activity thresholds and pattern-of-life histograms by hour of day.
- Optional wearable HR/SpO2 integration for elevated-HR-while-idle and low-SpO2 alerts via `/ingest_wearable`.
- NDJSON event and audit logs for post-hoc review or downstream analytics.

## 2. Small Clinics and Field Triage Tents

**Scenario**: Cameras monitor triage bays, waiting areas, and corridors in a clinic or field medical tent. Some patients wear commercial-grade wearables.

**Goals**:
- Detect patients who have fallen or become inactive unexpectedly.
- Detect agitation or bursts of motion in sensitive areas.
- Provide clinicians or medics with event streams that can be integrated into existing dashboards.

**Relevant capabilities**:
- Vision-based fall and activity rules configured per area and camera via `data/health_config.json` and `data/safety_config.json`.
- Burst-activity and lingering rules tied to specific cameras and risk tags using the camera registry.
- Wearable integration to highlight physiological anomalies in combination with visual inactivity.
- Real-time event streaming and webhooks suitable for integration into existing C2/clinical dashboards.
- Lightweight recording index allowing mapping between events and stored media when a recorder is integrated.

## 3. Workshops, Depots, and Light Industrial Facilities

**Scenario**: Low-resolution cameras monitor shop floors, loading bays, and high-risk zones in maintenance depots or warehouses.

**Goals**:
- Detect lingering or unusual presence in high-risk zones, especially after hours.
- Detect bursts of motion that may correspond to incidents or unsafe behavior.
- Establish per-entity movement baselines across cameras (where allowed) without face recognition.

**Relevant capabilities**:
- Quiet-hours-motion rules driven by camera IDs and configured risk metadata.
- Linger detection based on span of observations per camera.
- Burst-activity rules over recent windows.
- Pattern-of-life analysis to distinguish routine behavior from anomalies.
- Role-aware API (viewer vs. admin) for camera registry management when role enforcement is enabled.

## 4. Monitoring in Temporary or Resource-Constrained Environments

**Scenario**: Expeditionary or temporary setups (e.g., forward operating bases, temporary shelter sites) where only low-cost cameras and limited compute are available.

**Goals**:
- Provide automated monitoring without relying on constant human observation.
- Use deterministic, low-footprint analytics rather than deep models that require GPUs.
- Maintain clear audit trails of emitted alerts.

**Relevant capabilities**:
- CPU-friendly HOG-based detection and deterministic feature pipeline, with optional ONNX acceleration when available.
- Configurable rules that can be tuned for each site without retraining models.
- Append-only NDJSON event store and log/webhook notifications.
- Simple deployment via Docker Compose or Kubernetes manifests.
- Portable configuration in JSON and environment variables for site-specific tuning.

## 5. Complementing Existing VMS and Security Operations

**Scenario**: An organization already operates a commercial VMS for recording and camera management, but wants richer, explainable health/safety signals.

**Goals**:
- Avoid replacing existing recording/video-wall infrastructure.
- Add entity-centric pattern-of-life, falls, and wearable fusion analytics.
- Feed alerts into existing alarm handling and incident management workflows.

**Relevant capabilities**:
- REST API for frame ingest that can be fed from the existing VMS or an intermediate service.
- REST API for wearable ingest when body-worn devices are present.
- Webhook notifications and SSE event streams into external systems.
- Pseudonymous entity IDs and per-entity patterns that can be cross-referenced with existing identity systems if policy allows.
- Basic users/roles and audit logging for API access when environment variables are configured.

## 6. Impact Considerations

Entity Profiler is not a turnkey product; it is a research and prototyping framework. Its impact derives from:
- Demonstrating how deterministic, pseudonymous analytics can be composed into explainable safety and health monitoring workflows.
- Showing how low-resolution cameras and commodity wearables can be fused without requiring face recognition or heavy models.
- Providing a concrete, auditable reference implementation (API + CLI + NDJSON logs) that integrators can adapt or embed into larger platforms.

Future documentation will add more quantitative impact estimates (e.g., reduction in operator load or false alarms) once evaluation and benchmarking scripts under evaluation/ are fully wired into this repository and run on representative datasets.
