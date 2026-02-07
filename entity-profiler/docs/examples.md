# Examples

This repository includes several concrete examples that exercise the current API, rules, and event surfaces:

- **API usage walkthrough**
	- See examples/api_usage.md for step-by-step examples of calling `/ingest_frame`, listing entities, and querying events.

- **End-to-end demo script**
	- See examples/run_end_to_end_demo.sh for a simple script that starts the API, pushes sample frames, and inspects resulting events.

- **RTSP snapshot ingestion**
	- examples/rtsp_snapshot_to_ingest.sh shows how to grab a still frame from an RTSP stream with `ffmpeg` and post it to `/ingest_frame`.
	- examples/rtsp_puller.py demonstrates periodic snapshot pulling from an RTSP source and ingestion into the API.

- **Wearable integration (Fitbit)**
	- examples/fitbit_pull.py illustrates pulling heart-rate data from the Fitbit Web API and forwarding it to `/ingest_wearable`, enabling health rules that combine vision and wearable context.

- **Dashboard and event streaming**
	- examples/dashboard.html is a lightweight web dashboard that connects to `/events/stream` (SSE) and the recent health/safety endpoints to visualize live events.

Together, these examples cover both real-time and batch-style usage of the current system and can be used as starting points for custom integrations.
