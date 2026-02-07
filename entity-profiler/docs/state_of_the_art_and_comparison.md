# State of the Art and Comparison

This document positions the Entity Profiler platform relative to traditional CCTV/video management systems (VMS) and modern multimodal monitoring approaches. It is intended to complement PLATFORM_CAPABILITIES.md by describing *external* baselines and how this repository differs.

## 1. Problem Context

Entity Profiler targets safety and coarse health monitoring using low-resolution, consumer- or field-grade cameras with optional wearable sensor integration. It is designed as analytics middleware that can sit between cameras and a command-and-control (C2) or dashboard system, or run as a standalone API with simple dashboards.

The core problems addressed are:
- Detecting concerning behaviors and conditions (falls, prolonged inactivity, unusual night activity, quiet-hours motion, lingering, bursts of motion) in cluttered environments without relying on face recognition.
- Providing deterministic, explainable alerts with full traceability from raw observations and wearable samples to events (via NDJSON event and audit logs).
- Supporting pseudonymous, entity-centric pattern-of-life analysis over days to weeks, with per-entity summaries exposed via API and CLI tools.

## 2. Traditional CCTV and VMS Capabilities (Baseline)

Modern commercial VMS platforms typically provide:
- **Camera and recording management**: device registration, health monitoring, continuous/event-based recording, retention policies, clip export.
- **Operator interfaces**: video walls, multi-camera layouts, playback timelines, bookmarks and annotations.
- **Built-in analytics**: motion, line crossing, zone intrusion, loitering, basic people/vehicle detection, sometimes crowding or left-object alerts.
- **Enterprise features**: user accounts, RBAC, audit logs for configuration and evidence exports, TLS, export signing/watermarking.
- **Integrations**: ONVIF/RTSP cameras, access control systems, alarm panels, and occasionally body-worn cameras.

This repository does *not* attempt to replicate full NVR/VMS functionality (recording, multi-site orchestration, rich operator video UIs). Instead it focuses on the analytics and event layer that could be embedded inside or atop a VMS.

## 3. How Entity Profiler Differs from Typical VMS Analytics

Relative to baseline VMS analytics, Entity Profiler emphasizes:

- **Deterministic, rule-based decision logic** rather than opaque, learned scoring models, with configuration in JSON and environment variables rather than retrained models.
- **Pseudonymous entity profiles** built from appearance + movement features, not real-world identities, with no face recognition in the pipeline.
- **Per-entity pattern-of-life summaries** (camera usage, hour-of-day histograms, observation span) instead of only camera- or zone-level triggers.
- **Integrated wearable streams** (heart rate, SpO2) fused with vision-based activity for richer context when available; vision-only deployments remain fully supported.
- **Explicit explainability and auditability**: each alert includes a rule type, human-readable description, thresholds, and observed values, and is written to append-only NDJSON logs alongside authentication and RBAC audit records.

Operationally, the platform:
- Exposes a **FastAPI service** with REST endpoints for frame ingest, wearable ingest, entity summaries, events, sites, cameras, recordings, and basic user/role management.
- Maintains a **camera and site registry** so rules can be targeted to specific locations and risk levels.
- Offers both **real-time streaming** (via Server-Sent Events) and **batch CLI tooling** over the same profiling and rules engines.

These design choices complement, rather than replace, conventional VMS features. For example, an existing VMS can remain the system of record for recordings and operator video walls, while Entity Profiler contributes higher-level health/safety events derived from low-resolution views and optional wearables.

## 4. Relation to Multimodal Stress/Health Monitoring Research

Recent research has explored multimodal stress and health monitoring using facial video, pose, and physiological signals (e.g., heart rate, skin conductance). Typical characteristics of that line of work include:
- Use of high-resolution facial imagery and dense physiological measurements.
- Deep learning models trained for stress or affect classification.
- Focus on accuracy metrics over curated datasets, often in controlled settings.

Entity Profiler differs in several respects:
- It is **engineered for low-resolution, privacy-preserving cameras** that do not support detailed facial analysis.
- It uses **simple, deterministic feature engineering** (gait speed proxies, posture heuristics, soft biometrics, clothing histograms) and explicit rules instead of complex learned classifiers.
- Wearable integration is designed as a configurable rule layer (elevated HR while idle, low SpO2) rather than as a black-box model.
- It explicitly separates **model-backed detection/pose components** (optional ONNX models) from downstream logic, keeping all profiling, clustering, and rule evaluation deterministic and inspectable.
- The emphasis is on **operational interpretability and auditability** over maximizing classification accuracy on benchmark datasets.

The platform can be used alongside model-heavy approaches: pattern-of-life summaries and event streams from Entity Profiler can provide context to, or serve as features for, more advanced stress/health models if desired.

## 5. Current Limitations Versus Industry-Standard Platforms

As of this repository state, Entity Profiler intentionally omits many full-VMS features:
- No built-in multi-camera recording UI or timeline playback; only entity profiles, events, and a lightweight recording index are persisted.
- No rich operator UI for multi-camera monitoring or PTZ control.
- No ONVIF discovery or turnkey, vendor-specific camera management.
- No full IAM/SSO stack; user and role management are deliberately minimal and file-backed.

Instead, it focuses on being:
- A **deterministic analytics engine** for entity-level behavior and health/safety events running on modest CPU resources.
- A **middleware component** that can feed existing C2 systems, dashboards, or VMS products through webhooks, REST APIs, and streaming endpoints.
- A **reference implementation** of pseudonymous, auditable pattern-of-life and health/safety analytics that integrators can embed or extend.

Future work in this repo is focused on:
- Expanding evaluation and benchmark scripts under evaluation/.
- Tightening security and audit defaults (e.g., stronger role enforcement in more endpoints by default).
- Improving guidance on integrating with traditional VMS stacks and wearable ecosystems.
