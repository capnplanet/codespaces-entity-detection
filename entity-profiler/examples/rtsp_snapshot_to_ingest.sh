#!/usr/bin/env bash
# Capture a single frame from an RTSP camera and POST to the ingest endpoint.
# Requirements: ffmpeg, curl. Adjust RTSP_URL, CAMERA_ID, API_URL as needed.

set -euo pipefail

RTSP_URL=${RTSP_URL:-"rtsp://user:pass@camera/stream"}
CAMERA_ID=${CAMERA_ID:-"door_cam"}
API_URL=${API_URL:-"http://localhost:8000/ingest_frame"}
TMP_FRAME=${TMP_FRAME:-"/tmp/ep_snapshot.jpg"}
TIMESTAMP=$(python - <<'PY'
import time
print(time.time())
PY
)

# Grab one frame
ffmpeg -y -rtsp_transport tcp -i "$RTSP_URL" -frames:v 1 -q:v 2 "$TMP_FRAME" </dev/null >/dev/null 2>&1

# Post to API
curl -X POST "$API_URL" \
  -F "camera_id=$CAMERA_ID" \
  -F "timestamp=$TIMESTAMP" \
  -F "frame=@$TMP_FRAME"

echo "\nSent snapshot from $CAMERA_ID at $TIMESTAMP"
