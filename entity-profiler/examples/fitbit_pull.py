"""Pull recent heart-rate samples from Fitbit Web API and forward to /ingest_wearable.

Requires:
- FITBIT_ACCESS_TOKEN: OAuth2 bearer token with heartrate scope
- FITBIT_USER_ID: usually "-" for current user
- API_URL: destination ingest_wearable endpoint (default http://localhost:8000/ingest_wearable)
- DEVICE_ID: logical device id to map in health_config.json
"""

import os
import time
import requests

FITBIT_API = "https://api.fitbit.com"


def fetch_intraday_hr(user_id: str, token: str, seconds: int = 300):
    now = int(time.time())
    start = now - seconds
    # Fitbit intraday endpoint; requires premium scopes in some tiers
    url = f"{FITBIT_API}/1/user/{user_id}/activities/heart/date/today/1d/1sec/time/{time.strftime('%H:%M:%S', time.gmtime(start))}/{time.strftime('%H:%M:%S', time.gmtime(now))}.json"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    series = data.get("activities-heart-intraday", {}).get("dataset", [])
    results = []
    for point in series:
        # point: {"time": "12:00:01", "value": 78}
        t = point.get("time")
        if not t:
            continue
        h, m, s = map(int, t.split(":"))
        ts = time.mktime(time.localtime(now))
        # approximate absolute timestamp using today's date and hh:mm:ss
        today = time.localtime(now)
        ts = time.mktime((today.tm_year, today.tm_mon, today.tm_mday, h, m, s, 0, 0, -1))
        results.append({"timestamp": ts, "heart_rate": float(point.get("value", 0))})
    return results


def forward(samples, api_url: str, device_id: str, token: str | None):
    if not samples:
        return 0
    payload = [
        {
            "device_id": device_id,
            "timestamp": s["timestamp"],
            "heart_rate": s.get("heart_rate"),
        }
        for s in samples
    ]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(api_url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return len(payload)


def main():
    fitbit_token = os.getenv("FITBIT_ACCESS_TOKEN")
    fitbit_user = os.getenv("FITBIT_USER_ID", "-")
    api_url = os.getenv("API_URL", "http://localhost:8000/ingest_wearable")
    device_id = os.getenv("DEVICE_ID", "fitbit_bridge")
    ep_token = os.getenv("EP_API_TOKEN", "")
    if not fitbit_token:
        raise SystemExit("FITBIT_ACCESS_TOKEN required")
    samples = fetch_intraday_hr(fitbit_user, fitbit_token, seconds=300)
    sent = forward(samples, api_url, device_id, ep_token or None)
    print(f"Forwarded {sent} samples from Fitbit to {api_url} for device {device_id}")


if __name__ == "__main__":
    main()
