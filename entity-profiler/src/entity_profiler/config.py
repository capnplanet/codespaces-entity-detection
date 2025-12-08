import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SEED = 1337


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for RNG sources to encourage deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = project_root / "data"
    raw_video_dir: Path = data_root / "raw"
    interim_dir: Path = data_root / "interim"
    processed_dir: Path = data_root / "processed"
    models_dir: Path = project_root / "models"

    def ensure(self) -> None:
        for p in [
            self.data_root,
            self.raw_video_dir,
            self.interim_dir,
            self.processed_dir,
            self.models_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ProfilingConfig:
    max_track_gap_seconds: float = 2.0
    fused_distance_threshold: float = 1.5
    min_observations_for_entity: int = 1


def load_config() -> ProfilingConfig:
    seed = int(os.getenv("EP_GLOBAL_SEED", DEFAULT_SEED))
    set_global_seed(seed)
    paths = Paths()
    paths.ensure()
    return ProfilingConfig()
