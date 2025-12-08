from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Pose:
    frame_index: int
    bbox: Tuple[int, int, int, int]
    joints: np.ndarray  # shape (num_joints, 2)


class PoseEstimator:
    """Pose estimator abstraction.

    This demo uses a dummy skeleton. Substitute with a fixed pre-trained
    pose model for real deployments.
    """

    def __init__(self, num_joints: int = 17):
        self.num_joints = num_joints

    def estimate(
        self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]], frame_index: int
    ) -> List[Pose]:
        poses: List[Pose] = []
        for bbox in bboxes:
            x, y, w, h = bbox
            joints = np.zeros((self.num_joints, 2), dtype=np.float32)
            cx, cy = x + w / 2.0, y + h / 2.0
            for j in range(self.num_joints):
                joints[j, 0] = cx
                joints[j, 1] = cy - (j / max(self.num_joints - 1, 1)) * h
            poses.append(Pose(frame_index=frame_index, bbox=bbox, joints=joints))
        return poses
