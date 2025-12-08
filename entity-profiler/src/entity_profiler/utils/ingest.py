"""Ingest helpers with retry/backoff, pacing, and basic metrics.

These utilities are intended for lightweight RTSP/file pullers that POST frames to
`/ingest_frame`. They remain dependency-light and CPU-friendly.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import requests

logger = logging.getLogger(__name__)


@dataclass
class IngestMetrics:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    dropped: int = 0
    last_latency_ms: float = 0.0
    queue_depth: int = 0


@dataclass
class BackoffPolicy:
    base: float = 0.5
    factor: float = 2.0
    max_backoff: float = 8.0
    max_retries: int = 3

    def delays(self):
        delay = self.base
        for _ in range(self.max_retries):
            yield delay
            delay = min(self.max_backoff, delay * self.factor)


class FrameIngestor:
    """Pulls frames from a capture source and posts to the API with retry/backoff.

    Use `step()` in a loop to pace reads. This class avoids threads; drive it from
    an external scheduler or a simple while loop.
    """

    def __init__(
        self,
        api_url: str,
        token: str | None = None,
        camera_id: str = "cam01",
        capture_factory: Optional[Callable[[], cv2.VideoCapture]] = None,
        pace_seconds: float = 1.0,
        backoff: BackoffPolicy | None = None,
        timeout: float = 5.0,
        drop_on_fail: bool = False,
    ):
        self.api_url = api_url.rstrip("/") + "/ingest_frame"
        self.token = token
        self.camera_id = camera_id
        self.capture_factory = capture_factory or (lambda: cv2.VideoCapture(0))
        self.cap = self.capture_factory()
        self.pace_seconds = pace_seconds
        self.backoff = backoff or BackoffPolicy()
        self.timeout = timeout
        self.drop_on_fail = drop_on_fail
        self.metrics = IngestMetrics()
        self._last_step = 0.0

    def _headers(self):
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _read_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()

    def step(self, now_ts: Optional[float] = None):
        now = now_ts or time.time()
        if now - self._last_step < self.pace_seconds:
            return
        self._last_step = now

        payload = self._read_frame()
        if payload is None:
            logger.warning("Ingest: failed to read frame")
            self.metrics.failures += 1
            return

        self.metrics.attempts += 1
        ts = time.time()
        for delay in self.backoff.delays():
            try:
                resp = requests.post(
                    self.api_url,
                    headers=self._headers(),
                    timeout=self.timeout,
                    data={"camera_id": self.camera_id, "timestamp": str(time.time())},
                    files={"frame": ("frame.jpg", payload, "image/jpeg")},
                )
                if resp.status_code < 500:
                    break
            except Exception as e:
                logger.warning("Ingest retry after error: %s", e)
            time.sleep(delay)
        else:
            # exhausted retries
            self.metrics.failures += 1
            if self.drop_on_fail:
                self.metrics.dropped += 1
            return

        self.metrics.last_latency_ms = (time.time() - ts) * 1000.0
        if resp.ok:
            self.metrics.successes += 1
        else:
            self.metrics.failures += 1

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def paced_loop(ingestor: FrameIngestor, duration_seconds: float | None = None):
    start = time.time()
    try:
        while True:
            now = time.time()
            if duration_seconds is not None and now - start >= duration_seconds:
                break
            ingestor.step(now_ts=now)
    finally:
        ingestor.release()
