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
from ..camera.models import CameraRegistry
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
from ..utils.auth import require_token, validate_token_or_key, require_role
from ..security.models import RoleName
from ..utils.audit import NDJSONAuditLogger, build_audit_logger, make_audit_record
from ..recording.index import RecordingIndex
from . import users as users_router

app = FastAPI(title="Entity Profiler API")

# Allow browser dashboard access; keep origins open for demo, tighten in prod.
app.add_middleware(
    __import__("fastapi.middleware.cors", fromlist=["CORSMiddleware"]).CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(users_router.router)

cfg = load_config()
health_cfg = load_health_config()
safety_cfg = load_safety_config()
paths = Paths()
store = EntityStore()
camera_registry = CameraRegistry(paths.interim_dir / "camera_registry.json")
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
audit_logger = NDJSONAuditLogger(paths.interim_dir / "audit.ndjson")
recording_index = RecordingIndex(paths.interim_dir / "recordings" / "index.ndjson")
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


class SiteCreate(BaseModel):
    name: str
    timezone: str = "UTC"
    metadata: dict[str, str] | None = None


class SiteResponse(BaseModel):
    site_id: str
    name: str
    timezone: str
    metadata: dict[str, str] | None = None


class CameraCreate(BaseModel):
    site_id: str
    name: str
    rtsp_url: str | None = None
    location: str | None = None
    risk_level: str | None = None
    enabled: bool = True
    metadata: dict[str, str] | None = None


class CameraResponse(BaseModel):
    camera_id: str
    site_id: str
    name: str
    rtsp_url: str | None = None
    location: str | None = None
    risk_level: str | None = None
    enabled: bool
    metadata: dict[str, str] | None = None


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


@app.get("/sites", response_model=List[SiteResponse])
def list_sites(_: None = Depends(require_token)):
    """List all configured sites."""
    sites = []
    for s in camera_registry.list_sites():
        sites.append(
            SiteResponse(
                site_id=s.site_id,
                name=s.name,
                timezone=s.timezone,
                metadata=s.metadata or {},
            )
        )
    return sites


@app.post("/sites", response_model=SiteResponse)
def create_site(payload: SiteCreate, _: None = Depends(require_token)):
    # In future, require admin role once identity binding is available.
    require_role(RoleName.ADMIN)
    site = camera_registry.create_site(
        name=payload.name,
        timezone=payload.timezone,
        metadata=payload.metadata or {},
    )
    return SiteResponse(
        site_id=site.site_id,
        name=site.name,
        timezone=site.timezone,
        metadata=site.metadata or {},
    )


@app.get("/cameras", response_model=List[CameraResponse])
def list_cameras(site_id: str | None = None, _: None = Depends(require_token)):
    """List cameras, optionally filtered by site."""
    if site_id:
        cams = camera_registry.list_cameras_for_site(site_id)
    else:
        cams = camera_registry.list_cameras()
    results: List[CameraResponse] = []
    for c in cams:
        results.append(
            CameraResponse(
                camera_id=c.camera_id,
                site_id=c.site_id,
                name=c.name,
                rtsp_url=c.rtsp_url,
                location=c.location,
                risk_level=c.risk_level,
                enabled=c.enabled,
                metadata=c.metadata or {},
            )
        )
    return results


@app.post("/cameras", response_model=CameraResponse)
def create_camera(payload: CameraCreate, _: None = Depends(require_token)):
    # In future, require operator or admin role.
    require_role(RoleName.OPERATOR)
    try:
        cam = camera_registry.create_camera(
            site_id=payload.site_id,
            name=payload.name,
            rtsp_url=payload.rtsp_url,
            location=payload.location,
            risk_level=payload.risk_level,
            enabled=payload.enabled,
            metadata=payload.metadata or {},
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="site not found")
    return CameraResponse(
        camera_id=cam.camera_id,
        site_id=cam.site_id,
        name=cam.name,
        rtsp_url=cam.rtsp_url,
        location=cam.location,
        risk_level=cam.risk_level,
        enabled=cam.enabled,
        metadata=cam.metadata or {},
    )


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


@app.get("/events/{event_id}/recording")
def get_event_recording(event_id: str, _: None = Depends(require_token)):
    """Best-effort lookup of a recording segment for a given event.

    Scans the NDJSON event store for the event_id, then queries the recording
    index for a segment around the event timestamp.
    """

    # Find event in NDJSON store
    ev_data = None
    try:
        path = event_store.path  # type: ignore[attr-defined]
    except AttributeError:
        path = paths.interim_dir / "events.ndjson"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("event_id") == event_id:
                    ev_data = data
                    break

    if not ev_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="event not found")

    camera_id = ev_data.get("camera_id")
    ts = ev_data.get("timestamp") or ev_data.get("emitted_at")
    if not camera_id or ts is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="event missing camera_id/timestamp")

    clip = recording_index.find_best_segment_for_event(str(camera_id), float(ts))
    if not clip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no recording segment found")
    return clip


@app.post("/events/ack")
def ack_event(payload: EventAck, _: None = Depends(require_token)):
    if payload.status not in {"open", "acknowledged", "resolved"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid status")
    now = time.time()
    _event_status[payload.event_id] = {"status": payload.status, "updated_at": now}

    # Minimal audit record for status changes
    record = make_audit_record(
        actor="api_client",
        action="event_status_update",
        resource=payload.event_id,
        details={"status": payload.status, "timestamp": now},
    )
    try:
        audit_logger.append(record)
    except Exception:
        # Audit failures must not break main control flow
        pass

    return {"event_id": payload.event_id, "status": payload.status}
