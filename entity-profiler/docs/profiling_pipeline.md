# Profiling Pipeline

This document describes the core profiling pipeline as it exists today. For full details and parameter values, see PLATFORM_CAPABILITIES.md.

## 1. Vision and Feature Pipeline

1. **Ingest frame** with `camera_id` and `timestamp` via the `/ingest_frame` API or CLI tooling.
2. **Detect persons** using either:
	- OpenCV HOG+SVM pedestrian detector (default, CPU‑only), or
	- An optional ONNX detector when `models/detector.onnx` and ONNX Runtime are available.
3. **Estimate pose (optional)** for each detection when a pose model is present at `models/pose_estimator.onnx`; otherwise pose lists are empty and downstream components degrade gracefully.
4. **Compute soft biometrics** from the bounding box (height, aspect ratio, area).
5. **Extract clothing descriptors** from the cropped patch (HSV color histogram + gradient‑based texture histogram).
6. **Build gait features (optional)** from recent pose sequences to derive a gait speed proxy and related mobility signals.
7. **Fuse features** into a unified 93‑dimensional vector by concatenating gait, soft biometrics, and clothing features.

## 2. Tracking and Entity Profiling

8. **Associate detections to tracks** using a cosine‑similarity tracker that maintains short‑term track IDs within each camera stream.
9. **Cluster observations into entities** using `EntityClusteringEngine`, which assigns each fused feature vector to an existing entity centroid or creates a new entity when distances exceed a threshold.
10. **Store observation** under that entity in `EntityStore` along with timestamp and camera ID.
11. **Summarize pattern‑of‑life** for that entity on demand (camera usage histogram, hour‑of‑day histogram, dominant camera/hour, observation span).

## 3. Rules and Events

12. **Evaluate health rules** (e.g., no recent activity, night‑time activity, low mobility, fall indicators, high activity bursts) using the current time, entity profiles, and—when configured—recent wearable samples.
13. **Evaluate safety rules** (e.g., quiet‑hours motion, lingering, high‑activity bursts on perimeter cameras) using entity profiles, camera metadata, and configured thresholds.
14. **Emit structured events** into an append‑only NDJSON event store, in‑memory recent‑event buffers, and configured notifiers (log files or webhooks), while also exposing them over HTTP endpoints and an SSE stream.

This profiling and rules pipeline is shared by both the online API and offline CLI tools, enabling consistent behavior across real‑time and batch workflows.
