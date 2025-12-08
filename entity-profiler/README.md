# Entity Profiler

Safety monitoring built on deterministic movement profiling for low-resolution, Ring-style home cameras. The system builds simple numeric “fingerprints” (appearance + motion) for each observation, groups them into pseudonymous entities, and raises configurable safety and health events (e.g., quiet-hours motion, lingering near an entry, long inactivity).

> **Important:** This is a research / prototyping framework. It is *not* a medical or security product and must not be used as the sole basis for safety‑critical or clinical decisions.

## What it does (today)

- Detects people (OpenCV HOG+SVM + NMS) and, if you provide an ONNX pose model, estimates pose; otherwise it skips pose gracefully.
- Computes lightweight features per detection:
   - Soft biometrics (bbox height, aspect ratio, area)
   - Clothing color/texture histograms
   - Simple gait features (from pose, if available)
- Fuses features and clusters them into pseudonymous entities (no real identities).
- Summarizes pattern of life (camera usage, hour-of-day histogram, span of observations).
- Evaluates **safety rules** on each ingest (or in batch) using `data/safety_config.json`, emitting events for:
   - Quiet-hours motion on perimeter/door cameras
   - Lingering in view beyond a threshold
   - Bursts of motion in short windows
- Evaluates **health rules** on each ingest (or in batch) using `data/health_config.json`, emitting events for:
   - No recent activity beyond a threshold
   - Night-time activity above a threshold
   - Low mobility (gait speed proxy below a threshold)
   - (You can extend rules in `health/` as needed.)
- Sends events to configurable notifiers (log file or webhook), exposes recent events at `/health/events` and `/safety/events`, and appends all events to `data/interim/events.ndjson`.
- Persists events to `data/interim/events.ndjson` for quick inspection/retention.

## Quickstart (GitHub Codespaces)

1. Open this repository in a GitHub Codespace.
2. Install dependencies (usually already done by the devcontainer):  
   ```bash
   pip install -e .
   ```
3. Run the API:
   ```bash
   uvicorn entity_profiler.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Send a frame to the `/ingest_frame` endpoint, e.g. with `curl`:
   ```bash
   curl -X POST "http://localhost:8000/ingest_frame"      -F "camera_id=cam01"      -F "timestamp=1719945600.0"      -F "frame=@examples/sample_frame.jpg"
   ```

### Ring-style snapshots

- The `/ingest_frame` endpoint accepts a single JPEG/PNG frame; you can point a Ring-like webhook or snapshot script at it. Pass a stable `camera_id` per device and a POSIX `timestamp` in seconds.
- If your device exposes RTSP, you can extract still frames with `ffmpeg` and post them the same way; the profiling and safety logic remains the same.
- A helper script `examples/rtsp_snapshot_to_ingest.sh` shows how to grab one frame via `ffmpeg` and `curl` it to `/ingest_frame`.
- A simple puller `examples/rtsp_puller.py` can loop on an RTSP stream and post snapshots every few seconds; configure with `RTSP_URL`, `API_URL`, `CAMERA_ID`, `EP_API_TOKEN`.
- A lightweight dashboard `examples/dashboard.html` fetches recent health/safety events; open it in a browser and set your API URL and token/key.

### API authentication

- Set `EP_API_TOKEN` to require `Authorization: Bearer <token>` on all endpoints. Optionally set `EP_API_KEYS` to a comma-separated list of API keys accepted via `X-Api-Key` header.

### Wearable biometrics (optional)

- Send heart-rate/SpO2 samples to `/ingest_wearable` as JSON: `[ {"device_id": "fitbit_123", "timestamp": 1719945600.0, "heart_rate": 120, "spo2": 94} ]`.
- Map devices to entities in `data/health_config.json` under `wearables` and tune thresholds (`hr_high`, `hr_low`, `spo2_low`, windows).
- Health events will include wearable-derived alerts (elevated HR while idle, low SpO2) and flow through the same notifiers.
- A helper `examples/fitbit_pull.py` shows how to pull recent heart-rate data from the Fitbit Web API and forward to `/ingest_wearable`; set `FITBIT_ACCESS_TOKEN`, `FITBIT_USER_ID`, `API_URL`, `DEVICE_ID`.

### CLI utilities

- Build profiles from a video or image directory and persist the store + summaries:

   ```bash
   python -m entity_profiler.cli.build_tracks data/raw/demo.mp4 --frame-stride 5 --output entity_profiles.json --store-file entity_store.json
   ```

- Query persisted entities (prefers `entity_store.json`, falls back to summaries):

   ```bash
   python -m entity_profiler.cli.query_entity --entity-id <UUID>  # or omit to list all
   ```

### Safety monitoring (prototype)

- Copy `data/safety_config.example.json` to `data/safety_config.json` and tune quiet hours, linger thresholds, and notifier targets (log or webhook).
- Batch-evaluate a persisted store:

   ```bash
   python -m entity_profiler.cli.safety_report entity_store.json --output safety_events.json
   ```

- The API evaluates safety rules on each `/ingest_frame` call and emits events to configured notifiers; recent events are available at `/safety/events`.

### Health monitoring (prototype)

- Provide a simple `data/health_config.json` to set thresholds (idle hours, night window, low-mobility speed) and notification targets (log or webhook). A starter config is at `data/health_config.example.json`—copy it to `data/health_config.json` and tune for your environment.
- Run batch reports:

   ```bash
   python -m entity_profiler.cli.health_report entity_store.json --output health_events.json
   ```

- The API evaluates health rules on each `/ingest_frame` call and emits events to configured notifiers; recent events are available at `/health/events`.

### Pose model (bring your own)

To enable real pose estimation, place an ONNX pose model at `models/pose_estimator.onnx`.
If the file is absent or the ONNX runtime is unavailable, the pipeline skips pose
and returns empty pose lists without failing. A lightweight COCO-17 single-person
model is recommended for CPU use; the same interface will work with GPU-enabled
ONNX Runtime when you swap providers in deployment.

## Project layout

- `src/entity_profiler`: core library code
- `tests`: basic tests
- `docker`: containerization
- `infra/k8s`: Kubernetes deployment skeleton
- `.devcontainer`: GitHub Codespace configuration

## Limitations & Ethics

- This code is intentionally conservative and deterministic in its design.
- It **does not** reconstruct or hallucinate high-resolution faces.
- It **does not** assign real-world identities; it only builds pseudonymous entity profiles.
- Use only in compliance with local laws, regulations, and ethical guidelines.
