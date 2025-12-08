from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from ..config import load_config
from ..vision.detection import PersonDetector
from ..vision.pose_estimation import PoseEstimator
from ..vision.soft_biometrics import compute_soft_biometrics
from ..vision.clothing_features import extract_clothing_features
from ..gait.gait_features import GaitSequence
from ..features.fusion import fuse_features
from ..profiling.entity_store import EntityStore
from ..profiling.clustering import EntityClusteringEngine
from ..profiling.pattern_of_life import summarize_entity_pattern

app = FastAPI(title="Entity Profiler API")

cfg = load_config()
store = EntityStore()
cluster_engine = EntityClusteringEngine(store)
detector = PersonDetector()
pose_estimator = PoseEstimator()


class EntityObservationResponse(BaseModel):
    entity_id: str
    num_observations: int
    profile_summary: dict


@app.post("/ingest_frame", response_model=List[EntityObservationResponse])
async def ingest_frame(
    camera_id: str = Form(...),
    timestamp: float = Form(...),
    frame: UploadFile = File(...),
):
    """Ingest a single frame and update entity profiles."""
    data = await frame.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    detections = detector.detect(img, frame_index=0)
    responses: List[EntityObservationResponse] = []
    for det in detections:
        poses = pose_estimator.estimate(img, [det.bbox], frame_index=0)
        pose_seq = GaitSequence(entity_id=None, poses=poses)
        soft_vec = compute_soft_biometrics(det.bbox)
        clothing_desc = extract_clothing_features(img, det.bbox)
        fused = fuse_features(pose_seq, soft_vec, clothing_desc)
        profile = cluster_engine.assign_observation(
            timestamp=timestamp, camera_id=camera_id, fused=fused
        )
        summary = summarize_entity_pattern(profile)
        responses.append(
            EntityObservationResponse(
                entity_id=profile.entity_id,
                num_observations=len(profile.observations),
                profile_summary=summary,
            )
        )
    return responses


@app.get("/entities")
def list_entities():
    """List all current entity summaries."""
    return [summarize_entity_pattern(p) for p in store.get_all_profiles()]
