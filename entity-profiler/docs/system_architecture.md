# System Architecture

High-level flow:

1. Video frames arrive from cameras.
2. `PersonDetector` finds coarse person regions.
3. `PoseEstimator` produces simple skeletons.
4. Soft biometrics and clothing descriptors are extracted per detection.
5. Gait features are computed over pose sequences.
6. Features are fused into a single vector.
7. `EntityClusteringEngine` assigns observations to entities or creates new ones.
8. `EntityStore` accumulates observations.
9. Pattern-of-life summaries are generated on demand for each entity.
