from typing import Tuple, List, Optional
import numpy as np


POSTURE_UPRIGHT = "upright"
POSTURE_HORIZONTAL = "horizontal"
POSTURE_UNKNOWN = "unknown"


def classify_posture(bbox: Tuple[int, int, int, int], joints: Optional[np.ndarray] = None) -> str:
    """Heuristic posture classification.

    - If pose joints are provided, use their vertical spread.
    - Otherwise fall back to bbox aspect ratio.
    """
    x, y, w, h = bbox
    if joints is not None and joints.size > 0:
        ys = joints[:, 1]
        vertical_span = float(ys.max() - ys.min()) if ys.size else 0.0
        horizontal_span = float((joints[:, 0].max() - joints[:, 0].min()) if joints.shape[0] else 0.0)
        if vertical_span < 1e-3:
            return POSTURE_UNKNOWN
        ratio = vertical_span / max(horizontal_span, 1e-3)
        if ratio >= 1.0:
            return POSTURE_UPRIGHT
        return POSTURE_HORIZONTAL

    # bbox-only heuristic
    if h <= 0 or w <= 0:
        return POSTURE_UNKNOWN
    ar = h / w
    if ar >= 1.0:
        return POSTURE_UPRIGHT
    if ar < 0.6:
        return POSTURE_HORIZONTAL
    return POSTURE_UNKNOWN
