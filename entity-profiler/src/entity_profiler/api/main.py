from typing import List
import asyncio
import json
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import load_config, load_health_config, load_safety_config, Paths
from ..vision.detection import PersonDetector
try:
    from ..vision.detector_onnx import OnnxPersonDetector
except Exception:  # pragma: no cover - optional dependency
    OnnxPersonDetector = None
from ..vision.pose_estimation import PoseEstimator
from ..vision.soft_biometrics import compute_soft_biometrics
from ..vision.clothing_features import extract_clothing_features
from ..vision.tracking import CosineTracker
from ..gait.gait_features import GaitSequence
from ..features.fusion import fuse_features
from ..profiling.entity_store import EntityStore
from ..profiling.clustering import EntityClusteringEngine
from ..profiling.pattern_of_life import summarize_entity_pattern
from ..health.rules import evaluate_health_events_with_wearables
from ..health.notifications import build_notifiers
from ..health.wearables import WearableBuffer, WearableSample
from ..safety.rules import evaluate_safety_events
from ..utils.event_store import NDJSONEventStore
from ..utils.auth import require_token, validate_token_or_key

app = FastAPI(title="Entity Profiler API")

# Allow browser dashboard access; keep origins open for demo, tighten in prod.
app.add_middleware(
    __import__("fastapi.middleware.cors", fromlist=["CORSMiddleware"]).CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

cfg = load_config()
health_cfg = load_health_config()
safety_cfg = load_safety_config()
paths = Paths()
store = EntityStore()
cluster_engine = EntityClusteringEngine(store)
if OnnxPersonDetector is not None and cfg.use_onnx_detector:
    _onnx_detector = OnnxPersonDetector()
    if getattr(_onnx_detector, "_runtime_available", False):
        detector = _onnx_detector
    else:
        detector = PersonDetector()
else:
    detector = PersonDetector()
pose_estimator = PoseEstimator()
wearable_buffer = WearableBuffer()
tracker = CosineTracker(sim_threshold=cfg.tracker_sim_threshold, max_age_seconds=cfg.tracker_max_age_seconds)
health_notifiers = build_notifiers(
    list(health_cfg.notification_targets), paths.interim_dir / "health_events.log"
)
safety_notifiers = build_notifiers(
    list(safety_cfg.notification_targets), paths.interim_dir / "safety_events.log"
)
event_store = NDJSONEventStore(paths.interim_dir / "events.ndjson")
_recent_events: List[dict] = []  # in-memory buffer of recent events
_recent_safety_events: List[dict] = []
_recent_wearable_events: List[dict] = []
_event_queue: asyncio.Queue[dict] = asyncio.Queue()
_event_status: dict[str, dict] = {}


def _profile_camera(entity_id: str) -> str | None:
    profile = store.get_profile(entity_id)
    if profile and profile.observations:
        return profile.observations[-1].camera_id
    return None


def _event_payload(ev_dict: dict, category: str) -> dict:
    payload = {
        "event_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "category": category,
        "status": "open",
        "emitted_at": time.time(),
    }
    payload.update(ev_dict)
    if "camera_id" not in payload:
        cam = _profile_camera(ev_dict.get("entity_id", ""))
        if cam:
            payload["camera_id"] = cam
    if "severity" not in payload and ev_dict.get("severity"):
        payload["severity"] = ev_dict["severity"]
    if "timestamp" not in payload and ev_dict.get("timestamp"):
        payload["timestamp"] = ev_dict["timestamp"]
    # include device_id if present in context
    if "device_id" not in payload:
        ctx = ev_dict.get("context") or {}
        if isinstance(ctx, dict) and ctx.get("device_id"):
            payload["device_id"] = ctx["device_id"]
    return payload


class EntityObservationResponse(BaseModel):
    entity_id: str
    num_observations: int
    profile_summary: dict
    track_id: int | None = None


@app.post("/ingest_frame", response_model=List[EntityObservationResponse])
async def ingest_frame(
    camera_id: str = Form(...),
    timestamp: float = Form(...),
    frame: UploadFile = File(...),
    _: None = Depends(require_token),
):
    """Ingest a single frame and update entity profiles."""
    data = await frame.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    detections = detector.detect(img, frame_index=0)
    responses: List[EntityObservationResponse] = []

    det_payloads = []
    for det in detections:
        poses = pose_estimator.estimate(img, [det.bbox], frame_index=0)
        pose_seq = GaitSequence(entity_id=None, poses=poses)
        soft_vec = compute_soft_biometrics(det.bbox)
        clothing_desc = extract_clothing_features(img, det.bbox)
        fused = fuse_features(pose_seq, soft_vec, clothing_desc)
        det_payloads.append((det, fused))

    # Update tracker using fused embeddings for continuity
    det_tuples = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score) for d, _ in det_payloads]
    det_embs = [f.as_array() for _, f in det_payloads]
    tracks = tracker.update(det_tuples, det_embs, frame_index=0, now_ts=timestamp)

    for (det, fused), track in zip(det_payloads, tracks):
        profile = cluster_engine.assign_observation(
            timestamp=timestamp, camera_id=camera_id, fused=fused
        )
        summary = summarize_entity_pattern(profile)
        responses.append(
            EntityObservationResponse(
                entity_id=profile.entity_id,
                num_observations=len(profile.observations),
                profile_summary=summary,
                track_id=track.track_id,
            )
        )

    # Evaluate health events once per ingest call (frame-level granularity)
    events = evaluate_health_events_with_wearables(store, health_cfg, wearable_buffer=wearable_buffer, now_ts=timestamp)
    for ev in events:
        payload = _event_payload(ev.__dict__, category="health")
        _recent_events.append(payload)
        try:
            _event_queue.put_nowait(payload)
        except Exception:
            pass
        for n in health_notifiers:
            n.send(ev)
        event_store.append([payload])

    # Evaluate safety events (quiet hours, lingering, bursts)
    safety_events = evaluate_safety_events(store, safety_cfg, now_ts=timestamp)
    for ev in safety_events:
        payload = _event_payload(ev.__dict__, category="safety")
        _recent_safety_events.append(payload)
        try:
            _event_queue.put_nowait(payload)
        except Exception:
            pass
        for n in safety_notifiers:
            n.send(ev)
        event_store.append([payload])

    return responses


@app.get("/entities")
def list_entities(_: None = Depends(require_token)):
    """List all current entity summaries."""
    return [summarize_entity_pattern(p) for p in store.get_all_profiles()]


@app.get("/health/events")
def list_health_events(severity: str | None = None, _: None = Depends(require_token)):
    """List recent health events observed in this process."""
    events = _recent_events[-100:]
    if severity:
        events = [e for e in events if e.get("severity") == severity]
    return events


@app.get("/safety/events")
def list_safety_events(severity: str | None = None, _: None = Depends(require_token)):
    """List recent safety events observed in this process."""
    events = _recent_safety_events[-100:]
    if severity:
        events = [e for e in events if e.get("severity") == severity]
    return events


async def _event_stream():
    while True:
        ev = await _event_queue.get()
        # merge latest status if present
        status_entry = _event_status.get(ev.get("event_id", ""))
        if status_entry:
            ev = {**ev, **status_entry}
        yield f"data: {json.dumps(ev)}\n\n"


@app.get("/events/stream")
async def stream_events(token: str | None = None, api_key: str | None = None):
    # Support query token/api_key for browser EventSource; falls back to disabled if none set.
    validate_token_or_key(token, api_key)
    return StreamingResponse(_event_stream(), media_type="text/event-stream")


class WearableIngest(BaseModel):
    device_id: str
    timestamp: float
    heart_rate: float | None = None
    spo2: float | None = None


class EventAck(BaseModel):
    event_id: str
    status: str  # open|acknowledged|resolved


@app.post("/ingest_wearable")
def ingest_wearable(samples: List[WearableIngest], _: None = Depends(require_token)):
    """Ingest one or more wearable biometric samples (heart rate, SpO2)."""
    to_store: List[WearableSample] = []
    for s in samples:
        to_store.append(
            WearableSample(
                device_id=s.device_id,
                timestamp=s.timestamp,
                heart_rate=s.heart_rate,
                spo2=s.spo2,
                raw=None,
            )
        )
    wearable_buffer.add_samples(to_store)
    return {"accepted": len(to_store)}


@app.post("/events/ack")
def ack_event(payload: EventAck, _: None = Depends(require_token)):
    if payload.status not in {"open", "acknowledged", "resolved"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid status")
    _event_status[payload.event_id] = {"status": payload.status, "updated_at": time.time()}
    return {"event_id": payload.event_id, "status": payload.status}
