"""Minimal RTSP frame puller that posts snapshots to the ingest API.

Requirements: opencv-python, requests.
Configure via env vars or CLI args.
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import requests


def send_frame(api_url: str, camera_id: str, frame, token: str | None):
    retval, buf = cv2.imencode(".jpg", frame)
    if not retval:
        return False
    files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    data = {
        "camera_id": camera_id,
        "timestamp": str(time.time()),
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(api_url, data=data, files=files, headers=headers, timeout=5)
    resp.raise_for_status()
    return True


def main():
    parser = argparse.ArgumentParser(description="RTSP puller -> ingest_frame")
    parser.add_argument("--rtsp", default=os.getenv("RTSP_URL", ""))
    parser.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000/ingest_frame"))
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "cam01"))
    parser.add_argument("--interval", type=float, default=float(os.getenv("PULL_INTERVAL", "5")))
    parser.add_argument("--token", default=os.getenv("EP_API_TOKEN", ""))
    args = parser.parse_args()

    if not args.rtsp:
        raise SystemExit("RTSP URL required (set --rtsp or RTSP_URL)")

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open RTSP stream: {args.rtsp}")

    token = args.token or None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(args.rtsp)
                continue
            try:
                send_frame(args.api, args.camera_id, frame, token)
            except Exception:
                # keep going on errors
                pass
            time.sleep(args.interval)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
