# State of the Art and Comparison

This document positions the Entity Profiler platform relative to traditional CCTV/video management systems (VMS) and modern multimodal monitoring approaches. It is intended to complement PLATFORM_CAPABILITIES.md by describing *external* baselines and how this repository differs.

## 1. Problem Context

Entity Profiler targets safety and coarse health monitoring using low-resolution, consumer- or field-grade cameras with optional wearable sensor integration. It is designed as analytics middleware that can sit between cameras and a command-and-control (C2) or dashboard system.

The core problems addressed are:
- Detecting concerning behaviors and conditions (falls, prolonged inactivity, unusual night activity, quiet-hours motion, lingering) in cluttered environments without relying on face recognition.
- Providing deterministic, explainable alerts with full traceability from raw observations to events.
- Supporting pseudonymous, entity-centric pattern-of-life analysis over days to weeks.

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

- **Deterministic, rule-based decision logic** rather than opaque, learned scoring models.
- **Pseudonymous entity profiles** built from appearance + movement features, not real-world identities.
- **Per-entity pattern-of-life summaries** (camera usage, hour-of-day histograms, observation span) instead of only camera- or zone-level triggers.
- **Integrated wearable streams** (heart rate, SpO2) fused with vision-based activity for richer context.
- **Explicit explainability**: each alert includes a concrete rule type, human-readable description, thresholds, and observed values in its context payload.

These design choices complement, rather than replace, conventional VMS features. For example, an existing VMS can remain the system of record for recordings and operator video walls, while Entity Profiler contributes higher-level health/safety events derived from low-resolution views and wearables.

## 4. Relation to Multimodal Stress/Health Monitoring Research

Recent research has explored multimodal stress and health monitoring using facial video, pose, and physiological signals (e.g., heart rate, skin conductance). Typical characteristics of that line of work include:
- Use of high-resolution facial imagery and dense physiological measurements.
- Deep learning models trained for stress or affect classification.
- Focus on accuracy metrics over curated datasets, often in controlled settings.

Entity Profiler differs in several respects:
- It is **engineered for low-resolution, privacy-preserving cameras** that do not support detailed facial analysis.
- It uses **simple, deterministic feature engineering** (gait speed proxies, posture heuristics, soft biometrics, clothing histograms) and explicit rules instead of complex learned classifiers.
- Wearable integration is designed as a configurable rule layer (elevated HR while idle, low SpO2) rather than as a black-box model.
- The emphasis is on **operational interpretability and auditability** over maximizing classification accuracy on benchmark datasets.

The platform can be used alongside model-heavy approaches: pattern-of-life summaries and event streams from Entity Profiler can provide context to, or serve as features for, more advanced stress/health models if desired.

## 5. Current Limitations Versus Industry-Standard Platforms

As of this repository state, Entity Profiler intentionally omits many full-VMS features:
- No built-in video recording or playback; only entity profiles and events are persisted.
- No rich camera-management UI, PTZ control, or ONVIF discovery.
- No multi-user RBAC, SSO, or evidentiary export tooling (these are under active design).

Instead, it focuses on being:
- A **deterministic analytics engine** for entity-level behavior and health/safety events.
- A **middleware component** that can feed existing C2 systems, dashboards, or VMS products through webhooks and streaming APIs.

Future work in this repo is adding benchmarks, security/audit improvements, lightweight camera registry, and clearer guidance on integrating with traditional VMS stacks.
