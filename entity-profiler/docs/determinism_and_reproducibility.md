# Determinism and Reproducibility

The project enforces deterministic behavior where practical, especially in profiling, clustering, and rules evaluation. Some upstream model components (e.g., ONNX detectors/pose estimators) and low-level libraries may still introduce minor numerical differences across hardware and platforms.

## 1. Sources of Determinism

- **Global RNG seeding**
	- `config.set_global_seed` and `utils.deterministic.deterministic_context` set global seeds for Python, NumPy, and, where applicable, other libraries.
- **Deterministic feature engineering**
	- Soft biometrics, clothing descriptors, gait features, and feature fusion are all pure, deterministic functions for a given input frame and pose.
- **Deterministic tracking and clustering**
	- Tracking and entity clustering rely on fixed thresholds and explicit distance/similarity metrics; given the same sequence of detections and features, they produce the same assignments.
- **Rules and alerts**
	- Health and safety rules are deterministic threshold checks over stored observations and configuration values; for a fixed store, config, and timestamp, they produce the same events.
- **Append-only logs**
	- Event and audit logs are written as append-only NDJSON, preserving a stable, replayable record of system behavior.

## 2. Non-Deterministic Boundaries

Despite best efforts, some components can introduce minor variation:

- ONNX Runtime and deep models (if enabled) may have small numeric differences across hardware/backends.
- Underlying BLAS/linear-algebra libraries used by NumPy/OpenCV may introduce tiny floating-point differences.
- Asynchronous delivery of events over SSE and webhooks can vary in timing, although the set of events and their content remain stable.

For evaluation or debugging, prefer running under `deterministic_context` with fixed seeds, a fixed model configuration (or HOG-only detection), and stable input data so that end-to-end behavior can be reproduced.
