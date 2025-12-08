import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from ..vision.detection import PersonDetector
from ..vision.pose_estimation import PoseEstimator
from ..vision.soft_biometrics import compute_soft_biometrics
from ..vision.clothing_features import extract_clothing_features
from ..gait.gait_features import GaitSequence
from ..features.fusion import fuse_features
from ..profiling.entity_store import EntityStore
from ..profiling.clustering import EntityClusteringEngine
from ..profiling.pattern_of_life import summarize_all_entities


def _iter_frames(path: Path, frame_stride: int) -> Iterable[Tuple[int, np.ndarray]]:
    if path.is_dir():
        image_files = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        for idx, img_path in enumerate(image_files):
            if idx % frame_stride != 0:
                continue
            frame = cv2.imread(str(img_path))
            if frame is not None:
                yield idx, frame
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {path}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_stride == 0 and frame is not None:
                yield idx, frame
            idx += 1
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Build entity profiles from a video file or directory of frames."
    )
    parser.add_argument("source", type=Path, help="Video file or directory of images")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Process every Nth frame to reduce runtime (default: 5)",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam01",
        help="Camera identifier to tag observations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("entity_profiles.json"),
        help="Where to write the summarized entity profiles",
    )
    parser.add_argument(
        "--store-file",
        type=Path,
        default=Path("entity_store.json"),
        help="Path to persist the full entity store for later querying",
    )
    args = parser.parse_args()

    store = EntityStore()
    cluster_engine = EntityClusteringEngine(store)
    detector = PersonDetector()
    pose_estimator = PoseEstimator()

    frame_count = 0
    obs_count = 0
    for frame_idx, frame in _iter_frames(args.source, args.frame_stride):
        frame_count += 1
        detections = detector.detect(frame, frame_index=frame_idx)
        for det in detections:
            poses = pose_estimator.estimate(frame, [det.bbox], frame_index=frame_idx)
            pose_seq = GaitSequence(entity_id=None, poses=poses)
            soft_vec = compute_soft_biometrics(det.bbox)
            clothing_desc = extract_clothing_features(frame, det.bbox)
            fused = fuse_features(pose_seq, soft_vec, clothing_desc)
            cluster_engine.assign_observation(
                timestamp=float(frame_idx), camera_id=args.camera_id, fused=fused
            )
            obs_count += 1

    summaries = summarize_all_entities(store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    store.save_json(args.store_file)

    print(
        f"Processed {frame_count} frames, generated {obs_count} observations, "
        f"wrote {len(summaries)} entity profiles to {args.output}, "
        f"store persisted to {args.store_file}"
    )


if __name__ == "__main__":
    main()
