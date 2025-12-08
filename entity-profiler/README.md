# Entity Profiler

Deterministic profiling of unknown individuals from low-resolution video using gait, soft biometrics, clothing descriptors, and pattern-of-life analysis.

> **Important:** This project is a research / prototyping framework. It is *not* a turnkey identification system and must not be used as the sole basis for legal or safety‑critical decisions.

## Features

- Extracts person detections using a deterministic HOG+SVM model with light NMS.
- Computes:
  - Soft biometrics (height in pixels, aspect ratio, bounding-box area)
  - Coarse clothing descriptors (color + texture histograms)
  - Simple gait features from pose sequences
- Fuses these features into a vector for each observation.
- Clusters observations into pseudonymous entities.
- Builds basic pattern-of-life summaries per entity (camera usage, time-of-day histogram).
- Exposes a FastAPI endpoint to ingest frames and receive entity assignments & summaries.

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

### CLI utilities

- Build profiles from a video or image directory and persist the store + summaries:

   ```bash
   python -m entity_profiler.cli.build_tracks data/raw/demo.mp4 --frame-stride 5 --output entity_profiles.json --store-file entity_store.json
   ```

- Query persisted entities (prefers `entity_store.json`, falls back to summaries):

   ```bash
   python -m entity_profiler.cli.query_entity --entity-id <UUID>  # or omit to list all
   ```

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
