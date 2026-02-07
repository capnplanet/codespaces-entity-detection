# System Architecture

This document summarizes how the current repository is wired end‑to‑end. It is intentionally high‑level; for a function‑by‑function view, see PLATFORM_CAPABILITIES.md.

## 1. Major Components

- **FastAPI service** (entity_profiler.api.main)
	- `/ingest_frame` for JPEG/PNG snapshots from cameras or RTSP pullers.
	- `/ingest_wearable` for heart‑rate/SpO2 samples.
	- `/events/stream` (SSE) and `/health/events`, `/safety/events` for monitoring.
	- `/entities`, `/sites`, `/cameras`, `/recordings`, `/users` for management and introspection.
- **Camera and site registry** (camera.models.CameraRegistry)
	- JSON‑backed registry of sites and cameras with metadata (location, risk level, RTSP URL, enabled flag).
- **Vision pipeline** (vision.*)
	- Person detection via OpenCV HOG+SVM or optional ONNX detector.
	- Optional pose estimation via ONNX model.
	- Soft biometrics and clothing descriptors per detection.
- **Gait and feature fusion** (gait.gait_features, features.fusion)
	- Multi-frame gait sequences buffered per track with mobility proxies and enriched deterministic descriptors (knee angles, asymmetry, ankle displacement).
	- Fused embeddings combining gait, soft biometrics, and clothing (currently 101 dimensions when all modalities are present).
- **Tracking and entity profiling** (vision.tracking, profiling.*)
	- Cosine‑similarity tracker for within‑camera track IDs.
	- Entity clustering across time/cameras and pattern‑of‑life summaries.
- **Health and safety rules engines** (health.*, safety.rules)
	- Vision‑only and wearable‑augmented health rules.
	- Safety rules for quiet‑hours motion, lingering, bursts, and related behaviors.
- **Event, audit, and recording indices** (utils.event_store, utils.audit, recording.*)
	- NDJSON event store for health/safety events.
	- NDJSON audit log for authentication and privileged actions.
	- Append‑only recording index for mapping stored media to events.
- **Authentication and RBAC** (utils.auth, security.models)
	- Bearer token and API‑key authentication.
	- Optional role enforcement and user store with audit trails.

## 2. Ingest and Processing Flow

The end‑to‑end flow for a single image snapshot is:

1. **Frame ingest**
	 - A client `POST`s to `/ingest_frame` with `camera_id`, `timestamp`, and `frame`.
	 - Authentication is enforced via bearer token and/or API key headers when configured.
2. **Detection and pose**
	 - `PersonDetector` (HOG+SVM or ONNX) finds person bounding boxes.
	 - `PoseEstimator` produces skeletal keypoints for each box when a pose model is present; otherwise this step yields empty pose lists.
3. **Feature extraction**
	 - `compute_soft_biometrics` derives height, aspect ratio, and area from each bounding box.
	 - `extract_clothing_features` computes HSV color histograms and gradient‑based texture descriptors from cropped patches.
	 - `GaitSequence` aggregates poses across frames for gait feature computation (on a per‑entity basis over time).
4. **Fusion and tracking**
	 - `fuse_features` concatenates gait, soft biometrics, and clothing into a single 93D embedding.
	 - `CosineTracker` associates detections to short‑lived tracks within each camera stream using cosine similarity between embeddings.
5. **Entity assignment and profiling**
	 - `EntityClusteringEngine` assigns each observation to an existing entity centroid or creates a new entity if distances exceed a threshold.
	 - `EntityStore` persists observations keyed by `entity_id`.
	 - `summarize_entity_pattern` exposes per‑entity pattern‑of‑life metrics (camera usage, hour‑of‑day histograms, time span) for API and CLI consumers.
6. **Rules evaluation and notifications**
	 - `evaluate_health_events_with_wearables` evaluates health rules based on recent entity observations and any wearable samples in `WearableBuffer`.
	 - `evaluate_safety_events` evaluates safety rules (quiet hours, lingering, bursts) using entity profiles, camera metadata, and the current timestamp.
	 - Events are wrapped into a common payload shape, appended to `NDJSONEventStore`, pushed to in‑memory recent‑event buffers, emitted over `/events/stream`, and forwarded to configured notifiers (log files or webhooks) by `health.notifications`.

## 3. Wearable Ingest Flow

Wearable samples follow a parallel, lighter‑weight path:

1. A client `POST`s JSON samples to `/ingest_wearable` (see README for schema).
2. Samples are buffered in `WearableBuffer`, keyed by device ID.
3. Health rules that depend on HR/SpO2 query this buffer when evaluating events for entities mapped to particular devices in `data/health_config.json`.
4. Wearable‑only alerts (e.g., low SpO2) share the same event pipeline and notifiers as vision‑based alerts.

## 4. Storage and Audit

- **Config and thresholds** live in `data/health_config.json` and `data/safety_config.json` plus environment variables (e.g., detector selection, tracker thresholds, role enforcement flags).
- **Profiles and pattern‑of‑life** are held in memory via `EntityStore` for the running process; CLI tools use persisted JSON stores for offline analysis.
- **Events** are durably stored as NDJSON in `data/interim/events.ndjson` for replay or downstream analytics.
- **Audit records** are appended to `data/interim/audit.ndjson` for authentication events and privileged endpoint access.
- **Recording metadata** is appended to `data/interim/recordings/index.ndjson` by the recording subsystem so that events can be cross‑referenced with stored media.

## 5. Deployment Surfaces

- **Docker Compose**: docker/docker-compose.yml defines a single‑service deployment suitable for local experiments.
- **Kubernetes**: infra/k8s manifests provide a minimal Deployment, Service, and ConfigMap for cluster deployment.
- **CLI tools**: entity_profiler.cli.* modules allow batch processing of stored video, generation of safety/health reports, and querying of entity profiles without running the API.

These surfaces all wrap the same deterministic core pipeline and rules engines described above.
