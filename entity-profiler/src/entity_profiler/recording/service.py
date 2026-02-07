from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

from ..camera.models import CameraRegistry, Camera
from ..config import Paths
from .index import RecordingIndex, RecordingSegment


@dataclass
class RecordingConfig:
    segment_seconds: float = 60.0
    output_root: Path = Path("recordings")


class CameraRecorder(threading.Thread):
    """Simple per-camera recorder writing time-segmented files.

    This is a minimal reference implementation suitable for small deployments
    and experimentation. Production systems should extend it with better error
    handling, backoff, and health reporting.
    """

    def __init__(
        self,
        camera: Camera,
        cfg: RecordingConfig,
        stop_flag: threading.Event,
        index: RecordingIndex | None = None,
    ):
        super().__init__(daemon=True)
        self.camera = camera
        self.cfg = cfg
        self.stop_flag = stop_flag
        self._index = index

    def run(self) -> None:
        if not self.camera.rtsp_url:
            return
        cap = cv2.VideoCapture(self.camera.rtsp_url)
        if not cap.isOpened():
            return

        # Basic properties; defaults if unavailable
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        frames_per_segment = int(max(1, fps * self.cfg.segment_seconds))
        out_dir = self.cfg.output_root / self.camera.camera_id
        out_dir.mkdir(parents=True, exist_ok=True)

        while not self.stop_flag.is_set():
            start_ts = time.time()
            segment_name = f"{int(start_ts)}.mp4"
            out_path = out_dir / segment_name
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            frames_written = 0

            while frames_written < frames_per_segment and not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                frames_written += 1

            writer.release()

            # Approximate end timestamp based on frames written and fps
            if self._index is not None and frames_written > 0:
                duration = frames_written / float(fps or 1.0)
                end_ts = start_ts + duration
                seg = RecordingSegment(
                    camera_id=self.camera.camera_id,
                    path=str(out_path),
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                try:
                    self._index.append([seg])
                except Exception:
                    # Index failures must not stop recording
                    pass

        cap.release()


class RecordingService:
    """Manage recorders for all enabled cameras in the registry.

    For now this is a single-process service intended for small deployments.
    """

    def __init__(self, registry_path: Path | str, cfg: Optional[RecordingConfig] = None):
        paths = Paths()
        self.registry = CameraRegistry(registry_path)
        self.cfg = cfg or RecordingConfig(output_root=paths.interim_dir / "recordings")
        self._index = RecordingIndex(paths.interim_dir / "recordings" / "index.ndjson")
        self._stop_flag = threading.Event()
        self._threads: Dict[str, CameraRecorder] = {}

    def start(self) -> None:
        for cam in self.registry.list_cameras():
            if not cam.enabled or not cam.rtsp_url:
                continue
            if cam.camera_id in self._threads:
                continue
            recorder = CameraRecorder(cam, self.cfg, self._stop_flag, self._index)
            self._threads[cam.camera_id] = recorder
            recorder.start()

    def stop(self) -> None:
        self._stop_flag.set()
        for rec in self._threads.values():
            rec.join(timeout=5.0)
        self._threads.clear()


def run_recording_service(registry_path: Path | str | None = None) -> None:
    """Convenience entrypoint to run a recording service.

    This can be invoked from a small CLI or separate process.
    """

    paths = Paths()
    reg_path = registry_path or (paths.interim_dir / "camera_registry.json")
    service = RecordingService(reg_path)
    service.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        service.stop()
