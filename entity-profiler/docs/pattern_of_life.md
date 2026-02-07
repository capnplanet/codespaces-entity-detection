# Pattern of Life

Pattern-of-life summaries describe how often and when a pseudonymous entity is observed, without assigning real-world identities. These summaries are derived from the `EntityStore` and exposed via API and CLI tooling.

For each entity, the current implementation computes:

- **Histogram of camera usage**
	- Count of observations per `camera_id`.
	- Highlights frequently visited locations.
- **Histogram of hour-of-day**
	- Count of observations per hour (0–23).
	- Highlights typical active hours and possible anomalies (e.g., unusually late activity).
- **Dominant camera and dominant hour**
	- The camera and hour bins with the highest counts.
- **Time span of observations**
	- Duration between first and last observation timestamps.

These metrics together form a behavioral signature that:

- Can be used to configure or tune health and safety rules (e.g., idle thresholds, night-time windows).
- Supports anomaly detection when combined with simple policies (e.g., new cameras, unusual hours).
- Remains pseudonymous within this system; any mapping to real identities occurs only in external systems.

The same pattern-of-life machinery is used by the online API (for `/entities` and event context) and offline CLI tools, ensuring consistent behavior across real-time and batch analysis.
