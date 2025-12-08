"""RTSP frame puller with retry/backoff, pacing, and metrics.

Requirements: opencv-python, requests.
Configure via env vars or CLI args.
"""

import argparse
import os

import cv2
from entity_profiler.utils.ingest import FrameIngestor, BackoffPolicy, paced_loop


def main():
    parser = argparse.ArgumentParser(description="RTSP puller -> ingest_frame with backoff")
    parser.add_argument("--rtsp", default=os.getenv("RTSP_URL", ""))
    parser.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000/ingest_frame"))
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "cam01"))
    parser.add_argument("--pace", type=float, default=float(os.getenv("PULL_INTERVAL", "5")), help="Seconds between frames")
    parser.add_argument("--token", default=os.getenv("EP_API_TOKEN", ""))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("INGEST_MAX_RETRIES", "3")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("INGEST_TIMEOUT", "5")))
    args = parser.parse_args()

    if not args.rtsp:
        raise SystemExit("RTSP URL required (set --rtsp or RTSP_URL)")

    def factory():
        cap = cv2.VideoCapture(args.rtsp)
        return cap

    ingestor = FrameIngestor(
        api_url=args.api,
        token=args.token or None,
        camera_id=args.camera_id,
        capture_factory=factory,
        pace_seconds=args.pace,
        backoff=BackoffPolicy(max_retries=args.max_retries),
        timeout=args.timeout,
    )
    paced_loop(ingestor)


if __name__ == "__main__":
    main()
