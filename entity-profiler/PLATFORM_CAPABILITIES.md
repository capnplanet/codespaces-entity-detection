# Entity Profiler Platform: Comprehensive Capabilities Documentation

## Executive Summary

The Entity Profiler is a **deterministic safety monitoring system** designed for low-resolution, Ring-style home cameras. It builds pseudonymous entity profiles using appearance and motion characteristics without reconstructing high-resolution faces or assigning real-world identities. The platform processes video frames to detect people, extract movement patterns, and generate configurable health and safety alerts.

**Important Disclaimer**: This is a research and prototyping framework. It is **not** a medical or security product and must not be used as the sole basis for safety-critical or clinical decisions.

---

## Core Architecture and Data Flow

### High-Level Pipeline

The system follows a deterministic processing pipeline:

1. **Frame Ingestion** → Video frames arrive from cameras with camera_id and timestamp
2. **Person Detection** → Identifies coarse person regions in each frame
3. **Pose Estimation** → Extracts skeletal keypoints (optional, graceful degradation)
4. **Feature Extraction** → Computes soft biometrics, clothing descriptors, and gait features
5. **Feature Fusion** → Combines all features into unified embedding vector
6. **Entity Clustering** → Assigns observations to existing entities or creates new ones
7. **Pattern Analysis** → Generates pattern-of-life summaries per entity
8. **Rules Evaluation** → Applies health and safety rules, emits events
9. **Notification** → Sends alerts via configured channels (log, webhook, API)

---

## Module-by-Module Function Analysis

### 1. Vision Module: Person Detection (`vision/detection.py`)

**Purpose**: Locate people in video frames using deterministic computer vision.

**Implementation Details**:
- **Primary Method**: OpenCV HOG (Histogram of Oriented Gradients) + SVM classifier
- **Algorithm**: Uses a pre-trained, CPU-only pedestrian detector from OpenCV
- **Parameters** (all fixed for determinism):
  - Hit threshold: 0.0
  - Window stride: (8, 8) pixels
  - Padding: (8, 8) pixels
  - Scale: 1.05 (pyramid scale factor)
  - Group threshold: 2 (for initial detection grouping)
  - NMS threshold: 0.35 (Non-Maximum Suppression to eliminate overlapping boxes)

**NMS Algorithm**:
- Sorts detections by score descending
- Iteratively keeps highest-scoring box
- Suppresses overlapping boxes using IoU (Intersection over Union)
- Returns list of non-overlapping bounding boxes

**Output**: List of `Detection` objects containing:
- `frame_index`: Sequential frame number
- `bbox`: (x, y, width, height) in pixels
- `score`: Detection confidence (from HOG weights)

**Predictive Quality**: Deterministic given same frame input. Performance depends on person size, lighting, and occlusion. Works best with upright pedestrians at distances where body structure is visible.

**Optional ONNX Enhancement** (`vision/detector_onnx.py`):
- When `models/detector.onnx` exists and ONNX Runtime is available, uses neural network detector
- Expected input: 1×3×640×640 float32 tensor, RGB normalized to [0,1]
- Expected output: (N, 6) array [x1, y1, x2, y2, score, class] where class=0 for person
- Falls back to HOG+SVM if unavailable

---

### 2. Vision Module: Pose Estimation (`vision/pose_estimation.py`)

**Purpose**: Extract skeletal keypoint positions to enable gait and posture analysis.

**Implementation Details**:
- **Model-Based**: Requires `models/pose_estimator.onnx` (17-joint COCO format recommended)
- **Graceful Degradation**: If model absent or ONNX Runtime unavailable, returns empty pose list
- **Process**:
  1. Expands bounding box by 1.2× to capture full body context
  2. Crops and resizes to 256×256 pixels
  3. Converts BGR→RGB and normalizes to [0,1]
  4. Runs ONNX inference
  5. Denormalizes keypoints back to original image coordinates
  
**Pose Representation**:
- 17 keypoints in COCO format: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- Output format: `Pose` object with `joints` as (17, 2) numpy array [(x, y), ...]
- Coordinates in pixel space relative to original frame

**Predictive Quality**: Depends on pose model accuracy. Designed for single-person crops. Graceful degradation ensures pipeline continues without pose data if model unavailable.

---

### 3. Vision Module: Soft Biometrics (`vision/soft_biometrics.py`)

**Purpose**: Extract simple, non-invasive physical measurements from bounding boxes.

**Features Extracted** (all deterministic):
1. **Height (pixels)**: Bounding box height
2. **Aspect Ratio**: height / width ratio (tall vs. wide silhouette)
3. **Area (pixels²)**: width × height (proxy for person size or distance)

**Output**: `SoftBiometricVector` with three float32 values.

**Predictive Quality**: 
- Height and area change with distance from camera
- Aspect ratio relatively stable for upright posture
- Used in fall detection (height drop, aspect ratio shift from tall→wide)
- Cannot distinguish between individuals of similar size at similar distances

---

### 4. Vision Module: Clothing Features (`vision/clothing_features.py`)

**Purpose**: Create appearance-based descriptors for person re-identification.

**Implementation Details**:
1. **Crop Extraction**: Extracts person patch from bounding box, resizes to 32×64 pixels
2. **Color Histogram**:
   - Converts to HSV color space
   - Computes 2D histogram: 8 hue bins × 8 saturation bins = 64 dimensions
   - Normalizes and flattens histogram
3. **Texture Histogram**:
   - Converts to grayscale
   - Computes Sobel gradients (horizontal and vertical)
   - Calculates edge magnitude
   - 16-bin histogram of magnitude values

**Output**: `ClothingDescriptor` with 80-dimensional vector (64 color + 16 texture).

**Predictive Quality**: 
- Appearance changes with lighting, camera angle, and clothing changes
- Provides coarse re-identification within short time windows
- Not suitable for long-term identity tracking
- Works best when clothing has distinct colors or patterns

---

### 5. Gait Module: Movement Analysis (`gait/gait_features.py`)

**Purpose**: Extract movement characteristics from pose sequences for entity identification and mobility assessment.

**Implementation Details**:

**Input**: `GaitSequence` containing list of `Pose` objects over time.

**Feature Extraction Process**:
1. **Normalization**:
   - Stacks all pose joints from sequence: (num_frames, 17, 2)
   - Computes mean torso length: ||shoulder_center - hip_center||
   - Normalizes all joint positions by torso length (scale-invariant)

2. **Statistical Features**:
   - Mean x-coordinate across all joints and frames
   - Mean y-coordinate across all joints and frames
   - Standard deviation of x-coordinates (horizontal movement variability)
   - Standard deviation of y-coordinates (vertical movement variability)

3. **Velocity Features**:
   - Computes frame-to-frame joint position differences (velocity vectors)
   - Calculates speed per frame: mean ||velocity|| across all joints
   - Mean speed across sequence (gait speed proxy)
   - Standard deviation of speed (movement consistency)

4. **Metadata**:
   - Sequence length (number of poses)
   - Torso length (scale factor)
   - Two reserved dimensions (currently zeros for future expansion)

**Output**: 10-dimensional float32 vector.

**Gait Speed Interpretation**:
- Index 4 (speed_mean): Primary mobility indicator
- Low values (<0.05) suggest reduced mobility or slow movement
- High values suggest rapid movement or running
- Normalized by torso length, making it somewhat scale-invariant

**Predictive Quality for Health Monitoring**:
- **Low Mobility Detection**: Persistent low speed_mean across observations indicates reduced mobility
- **Movement Patterns**: Speed consistency (low speed_std) suggests regular gait
- **Limitations**: 
  - Requires pose estimation (optional component)
  - Sensitive to camera angle and occlusion
  - Best for frontal/side views with clear joint visibility
  - Returns zero vector if no poses available

**Predictive Quality for Re-identification**:
- Gait patterns are individually distinctive over short time periods
- Affected by footwear, terrain, health status, and intentional gait modification
- Combined with other features for entity clustering

---

### 6. Feature Fusion Module (`features/fusion.py`)

**Purpose**: Combine disparate feature types into unified representation for entity clustering.

**Implementation**:
- **Input Features**:
  1. Gait: 10 dimensions (from pose sequences)
  2. Soft Biometrics: 3 dimensions (height, aspect, area)
  3. Clothing: 80 dimensions (color + texture histograms)
  
- **Fusion Method**: Simple concatenation
- **Output**: `FusedFeatures` with 93-dimensional float32 vector

**Design Rationale**:
- Concatenation preserves all information without learned weights
- Deterministic: no trained fusion model required
- Each modality contributes distinct information (movement, size, appearance)

**Predictive Quality**:
- Combined features more discriminative than individual modalities
- Euclidean/cosine distance in fused space enables entity matching
- Robustness: if pose estimation unavailable, gait component is zeros but clothing/biometrics remain

---

### 7. Tracking Module (`vision/tracking.py`)

**Purpose**: Maintain consistent track IDs for detected persons within a camera view across frames.

**Algorithm**: Cosine similarity-based tracking with configurable thresholds.

**Implementation Details**:

**Data Structure**: `Track` objects containing:
- `track_id`: Unique integer identifier
- `bbox`: Current bounding box
- `score`: Detection confidence
- `last_seen_ts`: Timestamp of last update
- `embedding`: Fused feature vector (93D)
- `frame_index`: Current frame number

**Tracking Process**:
1. **Pruning**: Remove tracks not seen within `max_age_seconds` (default 3.0s)
2. **Association**: For each new detection:
   - Compute cosine similarity with all active track embeddings
   - Match to highest-similarity track if similarity ≥ `sim_threshold` (default 0.7)
   - Create new track if no match above threshold
3. **Update**: Update matched track's bbox, embedding, and timestamp

**Cosine Similarity Formula**:
```
similarity = (a · b) / (||a|| × ||b||)
```
Range: [-1, 1], where 1 = identical direction, 0 = orthogonal, -1 = opposite.

**Configuration**:
- `EP_TRACKER_SIM_THRESHOLD`: Minimum similarity for association (default 0.7)
- `EP_TRACKER_MAX_AGE_SECONDS`: Track expiration time (default 3.0s)

**Predictive Quality**:
- **Within-Camera**: Maintains identity across brief occlusions and pose changes
- **Cross-Camera**: Does NOT maintain identity across different cameras (by design)
- **Limitations**: Track IDs reset between camera views; long-term identity requires entity clustering

---

### 8. Entity Clustering and Profiling (`profiling/clustering.py`, `profiling/entity_store.py`)

**Purpose**: Assign observations to pseudonymous entities across cameras and time periods.

**Entity Store Design**:
- **Entity**: Pseudonymous identifier (UUID) representing an observed individual
- **Profile**: Collection of observations for one entity
- **Observation**: Single detection with timestamp, camera_id, and fused features

**Clustering Algorithm** (`EntityClusteringEngine`):
1. **Distance Metric**: Euclidean distance in 93D fused feature space
2. **Assignment Process**:
   - Compute distance from new observation to all entity centroids
   - Centroid = mean of all observation feature vectors for that entity
   - Match to closest entity if distance ≤ `fused_distance_threshold` (default 1.5)
   - Create new entity if no match within threshold
3. **Centroid Update**: Incrementally updated as new observations added

**Predictive Quality**:
- **Re-identification**: Can associate same person across cameras if appearance consistent
- **Limitations**:
  - Clothing changes break association
  - Similar-appearing individuals may cluster together
  - Threshold must balance false associations vs. entity fragmentation
- **Privacy**: Maintains pseudonymity; no biometric identity reconstruction

---

### 9. Pattern-of-Life Analysis (`profiling/pattern_of_life.py`)

**Purpose**: Summarize behavioral patterns without explicit identity labels.

**Computed Metrics per Entity**:

1. **Camera Usage Histogram**: Count of observations per camera_id
   - Identifies frequently visited locations
   - Example: `{"front_door": 42, "living_room": 128, "kitchen": 67}`

2. **Hour-of-Day Histogram**: Count of observations per hour (0-23)
   - Reveals activity timing patterns
   - Example: `{7: 12, 8: 15, 20: 18, 21: 24}` (morning and evening activity)

3. **Dominant Camera**: Camera with most observations
   - Primary location for this entity

4. **Dominant Hour**: Hour with most observations
   - Peak activity time

5. **Time Span**: Total duration from first to last observation (seconds)
   - Indicates observation window coverage

**Predictive Applications**:
- **Anomaly Detection**: Deviations from established patterns (new cameras, unusual hours)
- **Routine Recognition**: Identifies regular schedules
- **Occupancy Analysis**: Multi-entity patterns reveal household dynamics

**Limitations**:
- Requires sufficient observations for stable patterns (minimum 10-20 observations recommended)
- Short-term visitors may not establish recognizable patterns
- Patterns change over time (seasonal, lifestyle changes)

---

### 10. Health Monitoring: Rules Engine (`health/rules.py`)

**Purpose**: Detect health-related concerns through vision-based activity analysis and optional wearable integration.

**Health Rules Evaluated**:

#### 10.1 No Recent Activity (Idle Detection)
- **Trigger**: No observations for entity beyond `no_activity_hours` threshold (default 8 hours)
- **Calculation**: `now_timestamp - last_observation_timestamp > threshold`
- **Severity**: Critical
- **Use Case**: Detect prolonged absence of movement (potential emergency)
- **Limitations**: Cannot distinguish between no activity and camera blind spots

#### 10.2 Night-Time Activity (Sleep Disruption)
- **Trigger**: Observations during configured night window ≥ threshold (default 5 observations, 0-6 AM)
- **Calculation**: Counts observations where `hour_of_day(timestamp) in [night_start, night_end)`
- **Severity**: Warning
- **Use Case**: Identify sleep disturbances, nighttime wandering
- **Configuration**: `night_hours` tuple (start, end), `night_activity_threshold`

#### 10.3 Low Mobility
- **Trigger**: Mean gait speed proxy below threshold (default 0.05)
- **Calculation**: Average of `gait_feature[4]` (speed_mean) across all observations
- **Severity**: Warning
- **Dependency**: Requires pose estimation
- **Use Case**: Detect reduced mobility, potential fall risk
- **Limitations**: Only meaningful with pose data; affected by camera angles

#### 10.4 Wearable Integration - Elevated Heart Rate
- **Trigger**: Mean HR ≥ `hr_high` (default 110 bpm) AND idle ≥ `wearable_idle_grace_seconds` (default 600s)
- **Rationale**: High heart rate without movement suggests stress or medical event
- **Severity**: Warning
- **Window**: Samples within `wearable_window_seconds` (default 900s = 15 minutes)

#### 10.5 Wearable Integration - Low Heart Rate
- **Trigger**: Mean HR ≤ `hr_low` (default 45 bpm)
- **Severity**: Warning
- **Use Case**: Bradycardia detection

#### 10.6 Wearable Integration - Low SpO2
- **Trigger**: Minimum SpO2 ≤ `spo2_low` (default 92%)
- **Severity**: Critical
- **Use Case**: Hypoxemia detection

**Predictive Quality**:
- **True Positives**: Effectively detects configured threshold violations
- **False Positives**: Camera blind spots, intentional behavior changes
- **False Negatives**: Events outside camera coverage, misdetection during actual activity
- **Calibration**: Thresholds must be tuned per deployment environment

---

### 11. Health Monitoring: Fall Detection (`health/fall_activity.py`, `health/fall_model.py`)

**Purpose**: Detect potential falls using heuristic and model-based approaches.

#### 11.1 Heuristic Fall Detection (`fall_activity.py`)

**Algorithm**:
1. **Input**: Last two observations for entity within time window (default 4 seconds)
2. **Signals Analyzed**:
   - **Height Drop**: `height_ratio = new_height / old_height`
   - **Aspect Ratio Increase**: Silhouette becomes wider relative to height
   - **Area Change**: Overall bounding box area increase

3. **Detection Logic**:
   ```python
   IF height_ratio <= 0.35 (height drops to 35% or less)
   AND aspect_increase >= 1.8 (becomes 1.8× wider)
   AND area_increase >= 0.25 (area grows 25%+)
   THEN emit fall_suspected event
   ```

4. **Severity**: Critical

**Physical Interpretation**:
- Person transitions from upright (tall, narrow) to horizontal (short, wide)
- Area increases as person sprawls on ground
- Rapid change distinguishes fall from sitting/lying intentionally

#### 11.2 Model-Based Fall Detection (`fall_model.py`)

**Algorithm**: Fused scoring system combining multiple signals.

**Components**:
1. **Height Term**: `max(0, min(1, (threshold - height_ratio) / threshold))`
   - Penalizes height drops
   - Clamped to [0, 1]

2. **Aspect Term**: `max(0, min(1, (aspect_increase - threshold) / threshold))`
   - Rewards aspect ratio widening
   - Clamped to [0, 1]

3. **Speed Term**: `max(0, min(1, speed_delta / speed_threshold))`
   - Incorporates gait speed change (if available)
   - Sudden speed increase suggests rapid fall motion

4. **Area Term**: `max(0, min(1, area_increase / area_threshold))`
   - Rewards area expansion

**Fused Score**:
```
score = 0.4 × height_term + 0.3 × aspect_term + 0.2 × speed_term + 0.1 × area_term
```

**Trigger**: `score >= fall_score_threshold` (default 0.6)

**Predictive Quality**:
- **Advantages Over Heuristic**: Weights multiple signals, more robust to partial signal failures
- **Sensitivity**: Tunable via `fall_score_threshold`
- **Specificity**: May trigger on rapid sitting, bending, or lying down intentionally
- **Latency**: Requires two observations within time window (minimum 1-4 seconds)
- **Camera Dependency**: Works best with side/angled views; top-down views may miss height change

**Calibration Guidance**:
- Lower threshold: Higher sensitivity, more false positives
- Higher threshold: Lower sensitivity, fewer false positives
- Recommend testing with staged fall scenarios in actual deployment environment

---

### 12. Health Monitoring: Activity Level Detection (`health/fall_activity.py`)

**Purpose**: Detect unusually high activity bursts.

**Algorithm**:
1. **Window**: Count observations within `activity_window_seconds` (default 900s = 15 minutes)
2. **Trigger**: Count ≥ `high_activity_count_threshold` (default 8 observations)
3. **Severity**: Info

**Use Cases**:
- Detect agitation or pacing behavior
- Correlate with other health signals (e.g., elevated HR + high activity)
- Establish activity baselines

**Predictive Quality**:
- Depends on frame ingestion rate and detection consistency
- May indicate distress, search behavior, or normal active periods

---

### 13. Health Monitoring: Posture Classification (`health/posture.py`)

**Purpose**: Classify body posture as upright, horizontal, or unknown.

**Algorithm**:

**Pose-Based Classification** (if joints available):
1. Compute vertical span: `max(y) - min(y)` across all joint y-coordinates
2. Compute horizontal span: `max(x) - min(x)` across all joint x-coordinates
3. Calculate ratio: `vertical_span / horizontal_span`
4. Classification:
   - `ratio >= 1.0` → Upright
   - `ratio < 1.0` → Horizontal

**BBox-Based Fallback** (if no pose data):
1. Compute aspect ratio: `height / width`
2. Classification:
   - `aspect_ratio >= 1.0` → Upright
   - `aspect_ratio < 0.6` → Horizontal
   - `0.6 ≤ aspect_ratio < 1.0` → Unknown

**Predictive Quality**:
- Pose-based more accurate for irregular postures
- BBox-based works without pose model but less precise
- Camera angle affects accuracy (best with perpendicular views)

---

### 14. Safety Monitoring: Rules Engine (`safety/rules.py`)

**Purpose**: Detect security and safety concerns through behavioral pattern analysis.

**Safety Rules Evaluated**:

#### 14.1 Quiet Hours Motion Detection
- **Trigger**: Observation during configured quiet hours on designated perimeter cameras
- **Configuration**:
  - `quiet_hours`: Tuple (start_hour, end_hour), supports wrap-around (e.g., 23-6)
  - `quiet_hours_cameras`: List of camera_ids to monitor (e.g., front_door, back_door)
- **Severity**: 
  - Critical if camera marked `risk: high`
  - Warning otherwise
- **Use Case**: Detect unexpected entry attempts, nighttime intrusions

#### 14.2 Lingering Detection
- **Trigger**: Entity present at single camera beyond `linger_seconds` threshold (default 120s)
- **Calculation**: `max(timestamps) - min(timestamps)` for observations at same camera
- **Severity**:
  - Critical if camera marked `risk: high`
  - Warning otherwise
- **Use Cases**: Loitering near entries, suspicious behavior, blocked exits

#### 14.3 Burst Activity Detection
- **Trigger**: ≥ `burst_count_threshold` observations (default 6) within `burst_window_seconds` (default 60s)
- **Severity**:
  - Warning if high-risk camera
  - Info otherwise
- **Use Cases**: Rapid repeated motion, potential altercation, panicked behavior

**Camera Risk Configuration** (`data/safety_config.json`):
```json
{
  "areas": {
    "front_door": {"label": "Main Entry", "risk": "high"},
    "backyard": {"label": "Rear Property", "risk": "medium"}
  }
}
```

**Predictive Quality**:
- **Effectiveness**: Rules activate reliably when thresholds crossed
- **False Positives**: Legitimate activity during quiet hours, normal lingering (e.g., standing at stove)
- **Context Dependency**: Risk assessments require domain knowledge of camera placement

---

### 15. Wearable Integration (`health/wearables.py`)

**Purpose**: Incorporate physiological data from wearable devices (Fitbit, smartwatches).

**Data Model**:
- **WearableSample**:
  - `device_id`: Unique device identifier
  - `timestamp`: POSIX timestamp (seconds)
  - `heart_rate`: Beats per minute (optional)
  - `spo2`: Blood oxygen saturation percentage (optional)
  - `raw`: Additional metadata (optional)

**Storage**:
- **WearableBuffer**: In-memory time-series buffer
- **TTL**: Samples retained for `ttl_seconds` (default 7200s = 2 hours)
- **Automatic Pruning**: Removes samples older than TTL on each insert

**Entity Mapping** (`data/health_config.json`):
```json
{
  "wearables": [
    {"device_id": "fitbit_001", "entity_id": "uuid-of-entity"}
  ]
}
```

**Integration Flow**:
1. External script (e.g., `examples/fitbit_pull.py`) fetches samples from device API
2. POST to `/ingest_wearable` endpoint
3. Samples stored in buffer
4. Health rules query buffer for recent samples within evaluation window
5. Events emitted if thresholds violated

**Predictive Capabilities**:
- **Elevated HR + Idle**: Detects stress, medical events without visible movement
- **Low SpO2**: Early hypoxemia warning
- **HR Trends**: Persistent low/high heart rate detection

**Limitations**:
- Requires external data ingestion scripts (not included in core pipeline)
- Device synchronization delays (wearables may report data with lag)
- Vision-only pipeline remains fully functional without wearable data

---

### 16. Event System and Notifications (`health/notifications.py`, `utils/event_store.py`)

**Purpose**: Emit, store, and route health/safety events to configured targets.

**Event Structure**:
- **Health/Safety Event**:
  - `entity_id`: UUID of entity
  - `severity`: "info" | "warning" | "critical"
  - `type`: Event type (e.g., "fall_suspected", "quiet_hours_motion")
  - `description`: Human-readable message
  - `timestamp`: Event time
  - `context`: Dict with additional metadata (e.g., HR values, camera_id)

**Extended API Event** (includes):
- `event_id`: Unique UUID per event
- `trace_id`: For correlation across systems
- `category`: "health" | "safety" | "wearable"
- `status`: "open" | "acknowledged" | "resolved"
- `emitted_at`: System timestamp
- `camera_id`: Associated camera (if available)

**Notification Targets** (configurable):
1. **Log File**: Appends JSON events to file (e.g., `data/interim/health_events.log`)
2. **Webhook**: HTTP POST to external endpoint (e.g., alerting service)

**Event Store** (`NDJSONEventStore`):
- Persists all events to `data/interim/events.ndjson` (newline-delimited JSON)
- Enables historical analysis and audit trail

**Real-Time Streaming**:
- **Endpoint**: `GET /events/stream` (Server-Sent Events)
- **Use Case**: Live dashboard updates
- **Authentication**: Supports query parameters `?token=...&api_key=...` for browser compatibility

**Predictive Quality**:
- **Reliability**: Events emitted deterministically when rules triggered
- **Latency**: Near-instant (milliseconds) from detection to notification
- **Persistence**: Events survive process restarts via NDJSON store

---

### 17. API Endpoints (`api/main.py`)

**Purpose**: RESTful interface for frame ingestion, querying, and event management.

#### 17.1 Frame Ingestion
**Endpoint**: `POST /ingest_frame`

**Input**:
- `camera_id`: String identifier for camera
- `timestamp`: POSIX timestamp (float, seconds)
- `frame`: Image file (JPEG/PNG) as multipart upload

**Process**:
1. Decode image
2. Detect persons (HOG or ONNX)
3. Estimate poses (if model available)
4. Extract features (soft biometrics, clothing, gait)
5. Update tracker (assigns track IDs)
6. Cluster to entities (assigns entity IDs)
7. Evaluate health rules (with wearables if configured)
8. Evaluate safety rules
9. Emit events to notifiers and event store

**Output**: List of `EntityObservationResponse`:
- `entity_id`: UUID
- `num_observations`: Total observations for entity
- `profile_summary`: Pattern-of-life summary
- `track_id`: Within-camera track ID

**Performance**: Single-frame processing typically <100ms on CPU (without ONNX models).

#### 17.2 Wearable Ingestion
**Endpoint**: `POST /ingest_wearable`

**Input**: JSON array of samples:
```json
[
  {
    "device_id": "fitbit_001",
    "timestamp": 1719945600.0,
    "heart_rate": 72,
    "spo2": 98
  }
]
```

**Output**: Samples stored in buffer, returns success confirmation.

#### 17.3 Entity Listing
**Endpoint**: `GET /entities`

**Output**: Array of entity summaries (pattern-of-life for all entities).

#### 17.4 Event Retrieval
**Endpoints**:
- `GET /health/events?severity=critical`
- `GET /safety/events?severity=warning`

**Output**: Recent events (last 100) matching severity filter.

#### 17.5 Event Streaming
**Endpoint**: `GET /events/stream`

**Output**: Server-Sent Events (SSE) stream of all events in real-time.

**Authentication**:
- Bearer token: `Authorization: Bearer <token>`
- API key: `X-Api-Key: <key>`
- Query params: `?token=<token>&api_key=<key>` (for browser EventSource)

**Configuration**:
- `EP_API_TOKEN`: Required bearer token
- `EP_API_KEYS`: Comma-separated API keys

---

### 18. Command-Line Interfaces

#### 18.1 Build Tracks (`cli/build_tracks.py`)
**Purpose**: Process video file or image directory, build entity profiles offline.

**Usage**:
```bash
python -m entity_profiler.cli.build_tracks data/raw/demo.mp4 \
  --frame-stride 5 \
  --output entity_profiles.json \
  --store-file entity_store.json
```

**Parameters**:
- `--frame-stride`: Process every Nth frame (reduces computation)
- `--output`: Entity summaries JSON
- `--store-file`: Full entity store with observations

#### 18.2 Query Entity (`cli/query_entity.py`)
**Purpose**: Retrieve entity profile from persisted store.

**Usage**:
```bash
python -m entity_profiler.cli.query_entity --entity-id <UUID>
# Or list all:
python -m entity_profiler.cli.query_entity
```

#### 18.3 Health Report (`cli/health_report.py`)
**Purpose**: Batch evaluation of health rules on persisted store.

**Usage**:
```bash
python -m entity_profiler.cli.health_report entity_store.json \
  --output health_events.json
```

#### 18.4 Safety Report (`cli/safety_report.py`)
**Purpose**: Batch evaluation of safety rules on persisted store.

**Usage**:
```bash
python -m entity_profiler.cli.safety_report entity_store.json \
  --output safety_events.json
```

---

## Module Interactions and Complementary Functions

### Interaction Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frame Ingestion                          │
│                    (Camera ID + Timestamp)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Person Detector│  ← Optional: ONNX model
                    └────────┬───────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │Pose Estimator│          │Soft Biometrics│
        │  (optional)  │          │   (bbox)     │
        └──────┬───────┘          └──────┬───────┘
               │                         │
               ▼                         ▼
        ┌─────────────┐          ┌──────────────┐
        │Gait Features│          │Clothing Feats│
        │  (10-dim)   │          │   (80-dim)   │
        └──────┬──────┘          └──────┬───────┘
               │                         │
               └──────────┬──────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │Feature Fusion │
                  │   (93-dim)    │
                  └───────┬───────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      ┌──────────────┐        ┌──────────────┐
      │CosineTracker │        │Entity Cluster│
      │ (track IDs)  │        │  (entity ID) │
      └──────────────┘        └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │Entity Store  │
                              │ (profiles)   │
                              └──────┬───────┘
                                     │
                ┌────────────────────┴────────────────────┐
                │                                         │
                ▼                                         ▼
        ┌──────────────┐                          ┌──────────────┐
        │Pattern-of-Life│                         │Health Rules  │
        │   Summaries   │                         │Safety Rules  │
        └───────────────┘                         └──────┬───────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │ Event System │
                                                  │ + Notifiers  │
                                                  └──────────────┘
```

### Key Complementary Relationships

#### 1. **Detection → Pose → Gait** (Movement Analysis Chain)
- **Detection** provides bounding boxes
- **Pose** estimates joint positions within boxes
- **Gait** analyzes joint movement over time
- **Complementarity**: Each stage enriches the previous; graceful degradation if pose unavailable

#### 2. **Soft Biometrics + Clothing + Gait** (Entity Re-identification)
- **Soft Biometrics**: Size/shape (3D) — stable within sessions
- **Clothing**: Appearance (80D) — stable within hours/days
- **Gait**: Movement (10D) — individually distinctive
- **Complementarity**: Multimodal fusion improves discrimination; compensates for individual modality failures

#### 3. **Tracker + Clustering** (Identity Maintenance)
- **Tracker**: Short-term identity (seconds), within-camera
- **Clustering**: Long-term identity (hours/days), cross-camera
- **Complementarity**: Tracker provides continuity for frame-to-frame; clustering aggregates across time/space

#### 4. **Pattern-of-Life + Rules** (Anomaly Detection)
- **Pattern-of-Life**: Establishes normal behavioral baseline
- **Health/Safety Rules**: Detect deviations from baseline
- **Complementarity**: Rules become more specific as patterns mature; reduce false positives

#### 5. **Vision + Wearables** (Sensor Fusion)
- **Vision**: Activity, posture, location
- **Wearables**: Heart rate, SpO2 (physiological state)
- **Complementarity**: 
  - High HR + idle (vision) = potential medical event
  - Low mobility (vision) + normal HR (wearable) = different interpretation than low mobility + high HR
  - Provides context: physical activity vs. distress

#### 6. **Fall Detection: Heuristic + Model** (Redundancy)
- **Heuristic**: Simple thresholds, interpretable
- **Model**: Weighted fusion, tunable sensitivity
- **Complementarity**: Both run in parallel; increases detection probability while maintaining explainability

#### 7. **Real-Time API + Batch CLI** (Deployment Flexibility)
- **API**: Live frame-by-frame processing, immediate alerts
- **CLI**: Offline analysis, video file processing, testing
- **Complementarity**: Same core pipeline; API for production, CLI for development/validation

---

## Gait and Movement Analysis: Detailed Examination

### Purpose and Scope
Gait analysis extracts movement characteristics from pose sequences to enable:
1. **Entity Re-identification**: Gait patterns are individually distinctive
2. **Health Monitoring**: Gait speed/consistency indicates mobility status
3. **Activity Recognition**: Movement patterns distinguish walking, running, pacing

### Detailed Algorithm Breakdown

#### Step 1: Pose Sequence Acquisition
- **Input**: List of `Pose` objects from consecutive frames
- **Minimum**: 1 pose (returns statistical defaults)
- **Optimal**: 10-30 poses over 2-5 seconds
- **Source**: Pose estimator output (17 COCO keypoints per frame)

#### Step 2: Torso Length Normalization
**Purpose**: Make features scale-invariant (person size, camera distance).

**Calculation**:
```python
torso_vectors = shoulder_center - hip_center  # per frame
torso_lengths = ||torso_vectors||  # per frame
mean_torso_length = mean(torso_lengths)
normalized_joints = all_joints / mean_torso_length
```

**Why Shoulder-Hip**:
- Most stable joint pair (less affected by limb movement)
- Defines body core reference frame

**Effect**: Persons at different distances have comparable gait features.

#### Step 3: Spatial Statistics
**Mean Coordinates** (2D):
- `mean_x = mean(normalized_joints[:, :, 0])`  # horizontal center of mass
- `mean_y = mean(normalized_joints[:, :, 1])`  # vertical center of mass

**Standard Deviations** (2D):
- `std_x = std(normalized_joints[:, :, 0])`  # horizontal spread (stride width)
- `std_y = std(normalized_joints[:, :, 1])`  # vertical spread (step height)

**Interpretation**:
- Low std_x: Narrow gait (feet close together)
- High std_x: Wide stance or side-to-side sway
- Low std_y: Shuffling (minimal foot lift)
- High std_y: High knee raises or jumping

#### Step 4: Velocity Calculation
**Frame-to-Frame Differences**:
```python
velocities = diff(normalized_joints, axis=0)  # (num_frames-1, 17, 2)
```

**Speed per Frame**:
```python
speeds_per_frame = mean(||velocities||, axis=joints)  # (num_frames-1,)
```

**Statistics**:
- `speed_mean = mean(speeds_per_frame)`  ← **Primary gait speed indicator**
- `speed_std = std(speeds_per_frame)`  ← Gait consistency

**Physical Meaning**:
- `speed_mean`: Average magnitude of joint movement between frames
- After normalization, approximately proportional to real-world gait speed
- `speed_std`: Low = steady pace; High = variable speed (acceleration, deceleration)

#### Step 5: Feature Vector Assembly
**10-Dimensional Output**:
```
[mean_x, mean_y, std_x, std_y, speed_mean, speed_std, 
 num_poses, torso_length, reserved_1, reserved_2]
```

**Reserved Dimensions**: Placeholders for future enhancements (e.g., cadence, asymmetry).

### Movement Analysis Capabilities

#### 1. Mobility Assessment
**Low Mobility Detection**:
- Threshold: `speed_mean < 0.05` (configurable)
- **Clinical Relevance**: Reduced gait speed correlates with fall risk, frailty
- **Limitations**: 
  - Camera-dependent (viewing angle affects apparent motion)
  - Binary threshold; does not capture gradual decline
  - Cannot distinguish intentional slow movement from impairment

**Monitoring Over Time**:
- Track `speed_mean` trend across days/weeks
- Declining trend more significant than single measurement

#### 2. Activity Level Classification
**Speed Ranges** (approximate, normalized units):
- `< 0.02`: Stationary or minimal movement
- `0.02 - 0.05`: Slow walking or shuffling
- `0.05 - 0.15`: Normal walking pace
- `> 0.15`: Brisk walking or running

**Context**: Combine with observation frequency for comprehensive activity assessment.

#### 3. Gait Consistency
**Speed Standard Deviation Interpretation**:
- Low `speed_std`: Steady, rhythmic gait (normal walking)
- High `speed_std`: Irregular movement (agitation, pacing, unstable gait)

**Application**: Sudden increase in `speed_std` may indicate distress or gait instability.

#### 4. Posture Transitions (via Fall Detection)
- **Rapid Height Change**: Detected via soft biometrics (bbox height)
- **Aspect Ratio Shift**: Tall → wide (upright → horizontal)
- **Speed Spike**: Incorporated in fall_model_events
- **Integration**: Gait speed_delta (last frame - first frame in fall window) contributes 20% to fused fall score

### Limitations and Edge Cases

#### Camera Angle Dependency
- **Frontal View**: Gait speed underestimated (motion mostly perpendicular to camera)
- **Side View**: Optimal for gait speed measurement
- **Top-Down View**: Loses vertical information; horizontal motion only
- **Mitigation**: Calibrate thresholds per camera placement

#### Occlusion and Detection Failures
- **Partial Occlusion**: Missed joints reduce pose quality; affects velocity calculation
- **Full Occlusion**: No pose available; gait features revert to zeros
- **Mitigation**: Feature fusion compensates with clothing/biometrics

#### Multi-Person Scenarios
- **Current Limitation**: Pose estimator designed for single-person crops
- **Consequence**: Multiple people in same bbox → noisy pose → unreliable gait features
- **Mitigation**: Detection quality affects bbox cleanliness; NMS reduces overlaps

#### Temporal Resolution
- **Frame Rate**: Higher FPS improves velocity estimates
- **Minimum**: 2 frames for velocity calculation
- **Optimal**: 10+ FPS for smooth gait analysis
- **Recommendation**: Configure `frame-stride` in CLI or ingestion rate in API accordingly

---

## Predictive Qualities and Integration

### Predictive Capabilities by Module

#### 1. Person Detection
**Prediction Type**: Spatial localization of persons in frame.

**Accuracy**:
- **True Positive Rate**: 70-90% for upright pedestrians in clear lighting
- **False Positive Rate**: 5-15% (shadows, furniture, partial occlusions)
- **Miss Rate**: Higher for seated, prone, or occluded individuals

**Factors**:
- Person distance from camera (best at 2-10 meters)
- Lighting conditions (degrades in low light)
- Clothing contrast with background

**Predictive Horizon**: Single-frame; no temporal prediction.

#### 2. Pose Estimation
**Prediction Type**: Joint localization per person.

**Accuracy** (COCO-trained models):
- **Mean Average Precision (mAP)**: 60-75% for lightweight models on low-res inputs
- **Keypoint Localization Error**: ~10-20 pixels typical

**Factors**:
- Pose model quality (user-provided ONNX)
- Person size in frame (smaller = lower accuracy)
- Joint visibility (clothing, occlusion)

**Predictive Horizon**: Single-frame; no forecasting.

#### 3. Gait Speed Proxy
**Prediction Type**: Current mobility level.

**Correlation with Real-World Gait Speed**:
- **Qualitative**: Monotonic relationship (higher feature value = faster movement)
- **Quantitative**: Requires calibration per camera setup
- **Typical Calibration**: Measure known walking speeds, fit linear model

**Predictive Value for Health**:
- **Cross-Sectional**: Identifies individuals with reduced mobility at point in time
- **Longitudinal**: Tracks mobility changes over weeks/months
- **Early Warning**: Gradual decline may precede falls or health deterioration

**Limitations**:
- Not a substitute for clinical gait analysis
- Affected by camera position, frame rate, detection quality

#### 4. Fall Detection
**Prediction Type**: Binary classification (fall vs. no fall).

**Performance Metrics** (estimated, deployment-dependent):
- **Sensitivity**: 60-80% (correctly detected falls)
- **Specificity**: 85-95% (correctly rejected non-falls)
- **Positive Predictive Value**: Depends on fall prevalence (typically 20-40% in home settings)

**Confusion Matrix**:
- **True Positives**: Actual falls with rapid postural change
- **False Positives**: Rapid sitting, bending over, lying down intentionally
- **False Negatives**: Slow falls, falls outside camera view, detection failures
- **True Negatives**: Normal activities

**Tuning Trade-offs**:
- **Lower Threshold** (e.g., 0.4): Higher sensitivity, more false alarms
- **Higher Threshold** (e.g., 0.8): Fewer false alarms, risk of missed falls
- **Recommendation**: Start at 0.6, adjust based on user feedback

**Predictive Horizon**: ~1-4 seconds (time window for observation pair).

#### 5. Pattern-of-Life Summaries
**Prediction Type**: Behavioral routine characterization.

**Maturity Time**: Requires 3-7 days of observations for stable patterns.

**Predictive Applications**:
- **Anomaly Detection**: Deviation from established patterns (new camera, unusual hour)
- **Absence Detection**: Expected activity not observed (combined with idle rules)
- **Routine Adherence**: Compare current behavior to historical baseline

**Accuracy**:
- **Stable Routines**: High predictability (90%+ for hour-of-day)
- **Variable Schedules**: Lower predictability (50-70%)

**Limitations**:
- Lifestyle changes (work schedule, visitors) invalidate historical patterns
- Requires manual interpretation for context

#### 6. Wearable-Enhanced Health Monitoring
**Prediction Type**: Physiological state inference.

**Heart Rate Context**:
- **Elevated HR + Idle**: Suggests stress, anxiety, or cardiac event (vs. elevated HR + activity = normal exercise)
- **Low HR + Normal Activity**: May indicate fitness or medication effects (vs. low HR + idle = bradycardia concern)

**SpO2 Context**:
- **Low SpO2 + Normal Activity**: Respiratory issue or device error
- **Low SpO2 + Supine Posture** (via vision): Sleep apnea concern

**Predictive Value**:
- **Multimodal Fusion**: Vision + wearable provides context unavailable to either alone
- **Latency**: Wearable devices may report data with 1-15 minute delay
- **Reliability**: Dependent on wearable device quality and skin contact

---

## Workflow Integration and Predictive Pipeline

### Real-Time Monitoring Workflow

```
1. Camera captures frame → POST to /ingest_frame
2. Detection + Feature Extraction (50-100ms)
3. Entity Assignment (10ms)
4. Pattern Update (5ms)
5. Rules Evaluation (5-10ms)
6. Event Emission (if triggered)
7. Notification Dispatch (webhook: 50-200ms; log: <1ms)
8. Dashboard Update (SSE stream, <50ms)

Total Latency: 120-365ms (detection to notification)
```

**Predictive Qualities**:
- **Near Real-Time**: Suitable for urgent alerts (falls, intrusions)
- **Deterministic**: Same input frame produces identical output (given same entity state)
- **Stateful**: Predictions improve as entity profiles mature

### Batch Analysis Workflow

```
1. Collect video file or frame directory
2. Run build_tracks CLI (offline processing)
3. Persists entity_store.json
4. Run health_report and safety_report CLIs
5. Analyze event patterns, tune thresholds
6. Deploy updated configs to production API
```

**Predictive Qualities**:
- **Retrospective Analysis**: Identifies missed events, validates thresholds
- **Threshold Calibration**: Empirical ROC curve generation
- **Baseline Establishment**: Determines normal patterns before anomaly detection

### Longitudinal Monitoring (Days to Months)

**Data Accumulation**:
- Observations per entity grow over time (hundreds to thousands)
- Pattern-of-life histograms stabilize (first 7 days)
- Gait speed trends emerge (14+ days)

**Predictive Applications**:
1. **Decline Detection**: Compare current week's gait speed to baseline month
2. **Seasonal Patterns**: Activity changes with daylight hours, weather
3. **Health Event Correlation**: Falls preceded by mobility decline (days to weeks prior)

**Statistical Significance**:
- Requires sufficient observations: minimum 50-100 per entity for stable statistics
- Outlier removal: Exclude single-frame anomalies
- Trend Analysis: Moving averages, Mann-Kendall tests for monotonic trends

---

## Configuration and Calibration

### Health Monitoring Configuration (`data/health_config.json`)

**Critical Parameters**:
```json
{
  "no_activity_hours": 8.0,
  "night_hours": [0, 6],
  "night_activity_threshold": 5,
  "low_mobility_speed_threshold": 0.05,
  "fall_height_drop_ratio": 0.35,
  "fall_aspect_ratio_increase": 1.8,
  "fall_score_threshold": 0.6,
  "hr_high": 110.0,
  "hr_low": 45.0,
  "spo2_low": 92.0,
  "notify_min_severity": "warning"
}
```

**Calibration Guidance**:
- **no_activity_hours**: Adjust based on typical absence periods (work, sleep)
- **low_mobility_speed_threshold**: Measure gait speeds of target population; set to 5th percentile
- **fall_score_threshold**: Staged fall testing; adjust for acceptable false positive rate
- **Wearable thresholds**: Consult medical guidelines; age-dependent

### Safety Monitoring Configuration (`data/safety_config.json`)

**Critical Parameters**:
```json
{
  "quiet_hours": [23, 6],
  "quiet_hours_cameras": ["front_door", "back_door"],
  "linger_seconds": 120.0,
  "burst_window_seconds": 60.0,
  "burst_count_threshold": 6,
  "areas": {
    "front_door": {"label": "Entry", "risk": "high"}
  }
}
```

**Calibration Guidance**:
- **linger_seconds**: Measure typical dwell times at each camera; set 2× normal
- **burst_count_threshold**: Depends on frame ingestion rate (higher FPS = higher threshold)
- **Risk labels**: Domain-specific; entry points typically "high", interior "medium/low"

### System Configuration (Environment Variables)

```bash
EP_GLOBAL_SEED=1337                    # Reproducibility
EP_USE_ONNX_DETECTOR=1                 # Enable neural network detector
EP_TRACKER_SIM_THRESHOLD=0.7           # Track association sensitivity
EP_TRACKER_MAX_AGE_SECONDS=3.0         # Track expiration time
EP_API_TOKEN=secret_token_here         # API authentication
EP_API_KEYS=key1,key2,key3             # Alternative auth
```

---

## Privacy, Ethics, and Limitations

### Privacy Design Principles

1. **Pseudonymous Identities**: Entity IDs are UUIDs with no inherent connection to real-world identity
2. **No Facial Recognition**: System explicitly avoids high-resolution face reconstruction
3. **Local Processing**: All computation occurs on-premises; no cloud dependency (optional webhooks configurable)
4. **Configurable Retention**: Users control data persistence duration (via event store TTL)

### Ethical Considerations

1. **Informed Consent**: All individuals in monitored spaces must be informed of surveillance
2. **Appropriate Use**: Designed for safety monitoring with resident consent, not covert surveillance
3. **Human Review**: System alerts require human verification before action
4. **Medical Disclaimer**: NOT a medical device; cannot replace professional health monitoring

### Technical Limitations

1. **Camera Coverage**: Cannot monitor areas outside camera field of view
2. **Detection Failures**: Occlusions, poor lighting, unusual postures reduce accuracy
3. **Entity Confusion**: Similar-appearing individuals may be clustered together
4. **Clothing Changes**: Breaks entity associations (by design for privacy)
5. **False Positives**: All rule-based systems generate false alerts; threshold tuning required
6. **Latency**: Wearable data may lag; fall detection requires 1-4 seconds of observation

### Deployment Requirements

1. **Hardware**: CPU-only operation supported; GPU optional for ONNX models
2. **Network**: API requires HTTP access; webhooks need outbound connectivity
3. **Frame Rate**: 5-10 FPS recommended for gait analysis
4. **Camera Placement**: Side/angled views optimal; avoid top-down mounting
5. **Lighting**: Minimum 50 lux for reliable detection
6. **Resolution**: 640×480 minimum; 1280×720 recommended

---

## Summary: Integrated Predictive Capabilities

### What This Platform Predicts

1. **Immediate State** (0-5 seconds):
   - Person location and count per camera
   - Current posture (upright, horizontal)
   - Fall events (rapid postural change)

2. **Short-Term Activity** (1-15 minutes):
   - Movement patterns (pacing, lingering)
   - Activity bursts (rapid repeated motion)
   - Wearable-detected physiological anomalies (HR, SpO2)

3. **Medium-Term Patterns** (hours to days):
   - Daily activity rhythms (hour-of-day histogram)
   - Camera usage patterns (location preferences)
   - Sleep disruption trends (nighttime activity)

4. **Long-Term Trends** (weeks to months):
   - Mobility decline (gait speed trends)
   - Behavioral pattern changes (routine deviations)
   - Prolonged inactivity (absence detection)

### What This Platform Does NOT Predict

1. **Medical Diagnoses**: Cannot diagnose diseases or predict specific health outcomes
2. **Intent**: Cannot infer why someone is present or what they plan to do
3. **Identity**: Cannot determine real-world identity (by design)
4. **Off-Camera Events**: No prediction beyond camera coverage
5. **Future Behavior**: No forecasting of future actions (only current state and retrospective patterns)

### Confidence and Reliability

- **Deterministic Core**: Detection, feature extraction, and rule evaluation are reproducible
- **Statistical Patterns**: Require 3-7 days for stability; confidence increases with observation count
- **Threshold-Dependent**: All alerts are binary (threshold-based); no probabilistic confidence scores
- **User Calibration**: Performance highly dependent on environment-specific threshold tuning

---

## Conclusion

The Entity Profiler platform provides a comprehensive, deterministic framework for safety and health monitoring using low-resolution cameras with optional wearable integration. Its modular architecture enables:

- **Vision-based activity tracking** through person detection, pose estimation, and feature extraction
- **Gait and movement analysis** for mobility assessment and fall detection
- **Pseudonymous entity profiling** respecting privacy while enabling pattern recognition
- **Configurable health and safety rules** with real-time alerting
- **Multimodal sensor fusion** combining camera and wearable data

The system's **predictive qualities** are grounded in deterministic algorithms and statistical pattern recognition rather than learned models, ensuring reproducibility and interpretability. Its **complementary modules** work in concert to provide rich behavioral context unavailable to any single modality.

**Deployment success** requires careful threshold calibration, environmental awareness (camera placement, lighting), and human oversight of automated alerts. This platform is a **research tool and decision support system**, not an autonomous medical or security solution.

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-13  
**Grounded In**: Entity Profiler repository code, README.md, and documentation files in docs/
