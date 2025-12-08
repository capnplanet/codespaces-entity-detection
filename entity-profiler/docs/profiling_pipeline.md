# Profiling Pipeline

1. **Ingest frame** with camera ID and timestamp.
2. **Detect persons** to get bounding boxes.
3. **Estimate pose** and build a minimal pose sequence.
4. **Compute soft biometrics** from the bounding box.
5. **Extract clothing descriptors** from the cropped patch.
6. **Fuse features** into a unified vector.
7. **Cluster** against existing entity profiles to find the best match.
8. **Store observation** under that entity (or create a new entity).
9. **Summarize** pattern-of-life for that entity (cameras, times, counts).
