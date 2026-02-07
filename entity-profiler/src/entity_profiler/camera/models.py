from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import uuid


@dataclass
class Site:
    site_id: str
    name: str
    timezone: str = "UTC"
    metadata: Dict[str, str] | None = None


@dataclass
class Camera:
    camera_id: str
    site_id: str
    name: str
    rtsp_url: Optional[str] = None
    location: Optional[str] = None
    risk_level: Optional[str] = None  # e.g. low|medium|high
    enabled: bool = True
    metadata: Dict[str, str] | None = None


class CameraRegistry:
    """Simple JSON-backed registry for sites and cameras.

    This is intentionally minimal and suitable for small deployments. For
    production-scale use, replace the persistence layer with a database-backed
    implementation.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sites: Dict[str, Site] = {}
        self._cameras: Dict[str, Camera] = {}
        self._load()

    # Persistence helpers

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for s in payload.get("sites", []):
            site = Site(
                site_id=s["site_id"],
                name=s["name"],
                timezone=s.get("timezone", "UTC"),
                metadata=s.get("metadata") or {},
            )
            self._sites[site.site_id] = site
        for c in payload.get("cameras", []):
            cam = Camera(
                camera_id=c["camera_id"],
                site_id=c["site_id"],
                name=c["name"],
                rtsp_url=c.get("rtsp_url"),
                location=c.get("location"),
                risk_level=c.get("risk_level"),
                enabled=bool(c.get("enabled", True)),
                metadata=c.get("metadata") or {},
            )
            self._cameras[cam.camera_id] = cam

    def _save(self) -> None:
        payload = {
            "sites": [asdict(s) for s in self._sites.values()],
            "cameras": [asdict(c) for c in self._cameras.values()],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # Site operations

    def list_sites(self) -> List[Site]:
        return list(self._sites.values())

    def get_site(self, site_id: str) -> Optional[Site]:
        return self._sites.get(site_id)

    def create_site(self, name: str, timezone: str = "UTC", metadata: Optional[Dict[str, str]] = None) -> Site:
        site_id = str(uuid.uuid4())
        site = Site(site_id=site_id, name=name, timezone=timezone, metadata=metadata or {})
        self._sites[site_id] = site
        self._save()
        return site

    def update_site(self, site_id: str, **fields) -> Site:
        site = self._sites.get(site_id)
        if not site:
            raise KeyError(site_id)
        data = asdict(site)
        data.update({k: v for k, v in fields.items() if v is not None})
        self._sites[site_id] = Site(**data)
        self._save()
        return self._sites[site_id]

    def delete_site(self, site_id: str) -> None:
        if site_id in self._sites:
            # Remove cameras belonging to this site as well
            self._cameras = {cid: cam for cid, cam in self._cameras.items() if cam.site_id != site_id}
            del self._sites[site_id]
            self._save()

    # Camera operations

    def list_cameras(self) -> List[Camera]:
        return list(self._cameras.values())

    def list_cameras_for_site(self, site_id: str) -> List[Camera]:
        return [c for c in self._cameras.values() if c.site_id == site_id]

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        return self._cameras.get(camera_id)

    def create_camera(
        self,
        site_id: str,
        name: str,
        rtsp_url: Optional[str] = None,
        location: Optional[str] = None,
        risk_level: Optional[str] = None,
        enabled: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Camera:
        if site_id not in self._sites:
            raise KeyError(site_id)
        camera_id = str(uuid.uuid4())
        cam = Camera(
            camera_id=camera_id,
            site_id=site_id,
            name=name,
            rtsp_url=rtsp_url,
            location=location,
            risk_level=risk_level,
            enabled=enabled,
            metadata=metadata or {},
        )
        self._cameras[camera_id] = cam
        self._save()
        return cam

    def update_camera(self, camera_id: str, **fields) -> Camera:
        cam = self._cameras.get(camera_id)
        if not cam:
            raise KeyError(camera_id)
        data = asdict(cam)
        data.update({k: v for k, v in fields.items() if v is not None})
        self._cameras[camera_id] = Camera(**data)
        self._save()
        return self._cameras[camera_id]

    def delete_camera(self, camera_id: str) -> None:
        if camera_id in self._cameras:
            del self._cameras[camera_id]
            self._save()
