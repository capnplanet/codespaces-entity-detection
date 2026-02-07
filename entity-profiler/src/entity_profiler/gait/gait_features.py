from dataclasses import dataclass
from typing import List

import numpy as np

from ..vision.pose_estimation import Pose


@dataclass
class GaitSequence:
    entity_id: str | None
    poses: List[Pose]


# Versioning and layout helpers for gait feature vectors.
#
# Layout v1 (historical, still preserved in the first 10 slots):
#   0: mean_x (normalized)
#   1: mean_y (normalized)
#   2: std_x
#   3: std_y
#   4: speed_mean
#   5: speed_std
#   6: num_poses (sequence length)
#   7: torso_len (mean torso length in pixels)
#   8: reserved_0
#   9: reserved_1
#
# Extended layout (v2) appends extra descriptors but keeps the first 10
# elements unchanged so downstream rules (e.g., low-mobility) remain
# backward-compatible.

GAIT_FEATURE_VERSION: int = 2

GAIT_IDX_MEAN_X = 0
GAIT_IDX_MEAN_Y = 1
GAIT_IDX_STD_X = 2
GAIT_IDX_STD_Y = 3
GAIT_IDX_SPEED_MEAN = 4
GAIT_IDX_SPEED_STD = 5
GAIT_IDX_NUM_POSES = 6
GAIT_IDX_TORSO_LEN = 7


# Approximate COCO-style indices for lower-body joints used in enriched
# descriptors. These follow the conventional 17-joint COCO order and are
# only used for additional angles/dispersion features; the base 10-D layout
# does not depend on them.
JOINT_LEFT_HIP = 11
JOINT_RIGHT_HIP = 12
JOINT_LEFT_KNEE = 13
JOINT_RIGHT_KNEE = 14
JOINT_LEFT_ANKLE = 15
JOINT_RIGHT_ANKLE = 16


def gait_speed_mean(gait: np.ndarray) -> float:
    """Return the gait speed_mean component if present, else 0.0.

    This helper centralizes the layout so downstream code does not rely on
    hard-coded indices. It is safe against shorter-than-expected vectors.
    """

    if gait is None or gait.size <= GAIT_IDX_SPEED_MEAN:
        return 0.0
    return float(gait[GAIT_IDX_SPEED_MEAN])


def gait_speed_std(gait: np.ndarray) -> float:
    """Return the gait speed_std component if present, else 0.0."""

    if gait is None or gait.size <= GAIT_IDX_SPEED_STD:
        return 0.0
    return float(gait[GAIT_IDX_SPEED_STD])


def gait_feature_from_sequence(sequence: GaitSequence) -> np.ndarray:
    """Compute a simple deterministic gait feature from a sequence of poses.

    This function is intentionally conservative and CPU-friendly. It operates
    on image-space joints normalized by mean torso length. More advanced gait
    descriptors can be layered on top while preserving this base layout.
    """

    if not sequence.poses:
        return np.zeros(18, dtype=np.float32)

    joints_stack = np.stack([p.joints for p in sequence.poses], axis=0)

    # Compute per-frame torso lengths and drop clearly degenerate frames to
    # reduce the impact of spurious poses (e.g., extreme occlusion or tiny
    # detections). We treat very small or NaN torso lengths as invalid.
    torso_lengths = np.linalg.norm(joints_stack[:, 1, :] - joints_stack[:, 8, :], axis=-1)
    valid_mask = np.isfinite(torso_lengths) & (torso_lengths > 1e-3)
    if not np.any(valid_mask):
        return np.zeros(18, dtype=np.float32)

    valid_joints = joints_stack[valid_mask]
    valid_torso_lengths = torso_lengths[valid_mask]
    torso_len = float(np.mean(valid_torso_lengths + 1e-6))

    norm_joints = valid_joints / torso_len

    mean_coords = norm_joints.mean(axis=(0, 1))
    std_coords = norm_joints.std(axis=(0, 1))

    if norm_joints.shape[0] < 2:
        speed_mean = 0.0
        speed_std = 0.0
    else:
        vel = np.diff(norm_joints, axis=0)
        speed = np.linalg.norm(vel, axis=-1).mean(axis=1)
        speed_mean = float(speed.mean())
        speed_std = float(speed.std())

    # Enriched, still-deterministic descriptors: knee angles and simple
    # vertical ankle displacement statistics as crude cadence proxies and
    # asymmetry indicators.

    def _joint_angle(frame_joints: np.ndarray, ia: int, ib: int, ic: int) -> float:
        if (
            ia >= frame_joints.shape[0]
            or ib >= frame_joints.shape[0]
            or ic >= frame_joints.shape[0]
        ):
            return float("nan")
        a = frame_joints[ia]
        b = frame_joints[ib]
        c = frame_joints[ic]
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return float("nan")
        cosang = float(np.dot(v1, v2) / max(n1 * n2, 1e-6))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        return float(np.arccos(cosang))

    left_knee_angles = []
    right_knee_angles = []
    for frame in norm_joints:
        lk = _joint_angle(frame, JOINT_LEFT_HIP, JOINT_LEFT_KNEE, JOINT_LEFT_ANKLE)
        rk = _joint_angle(frame, JOINT_RIGHT_HIP, JOINT_RIGHT_KNEE, JOINT_RIGHT_ANKLE)
        if np.isfinite(lk):
            left_knee_angles.append(lk)
        if np.isfinite(rk):
            right_knee_angles.append(rk)

    def _mean_std(values: List[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

    lk_mean, lk_std = _mean_std(left_knee_angles)
    rk_mean, rk_std = _mean_std(right_knee_angles)
    asym_knee_mean_diff = abs(lk_mean - rk_mean) if (lk_mean or rk_mean) else 0.0

    # Vertical ankle displacement across frames (simple motion magnitude
    # proxy for stepping behaviour).
    vert_disps: List[float] = []
    if norm_joints.shape[0] >= 2:
        if JOINT_LEFT_ANKLE < norm_joints.shape[1]:
            la_y = norm_joints[:, JOINT_LEFT_ANKLE, 1]
            vert_disps.extend(np.abs(np.diff(la_y)).tolist())
        if JOINT_RIGHT_ANKLE < norm_joints.shape[1]:
            ra_y = norm_joints[:, JOINT_RIGHT_ANKLE, 1]
            vert_disps.extend(np.abs(np.diff(ra_y)).tolist())
    disp_mean, disp_std = _mean_std(vert_disps)

    num_angle_frames = float(max(len(left_knee_angles), len(right_knee_angles)))

    return np.array(
        [
            float(mean_coords[0]),
            float(mean_coords[1]),
            float(std_coords[0]),
            float(std_coords[1]),
            speed_mean,
            speed_std,
            float(len(sequence.poses)),
            float(torso_len),
            0.0,
            0.0,
            lk_mean,
            lk_std,
            rk_mean,
            rk_std,
            disp_mean,
            disp_std,
            asym_knee_mean_diff,
            num_angle_frames,
        ],
        dtype=np.float32,
    )
