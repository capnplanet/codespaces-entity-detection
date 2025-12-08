# API Usage Example

Assuming the API is running on `http://localhost:8000`:

```bash
curl -X POST "http://localhost:8000/ingest_frame"   -F "camera_id=cam01"   -F "timestamp=1719945600.0"   -F "frame=@examples/sample_frame.jpg"
```
