import numpy as np

from entity_profiler.features.fusion import fuse_features
from entity_profiler.gait.gait_features import GaitSequence
from entity_profiler.vision.pose_estimation import Pose
from entity_profiler.vision.soft_biometrics import compute_soft_biometrics


def test_fuse_features_shapes():
    pose = Pose(frame_index=0, bbox=(0, 0, 10, 20), joints=np.zeros((17, 2), dtype=np.float32))
    seq = GaitSequence(entity_id=None, poses=[pose])
    soft = compute_soft_biometrics((0, 0, 10, 20))
    fused = fuse_features(seq, soft, None)
    vec = fused.as_array()
    assert vec.ndim == 1
    assert vec.size > 0
