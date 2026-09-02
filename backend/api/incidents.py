"""
VigilVoice — Read-only Incident Records API (React frontend support)
====================================================================
Exposes the incident evidence JSON files already persisted under
reports/incidents/ so the Incident Center and Reports pages can list and
inspect them. This is strictly additive and read-only — it never creates,
modifies, or deletes records (incident creation stays in backend/api/live.py).
"""

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import PROJECT_ROOT

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])

INCIDENTS_DIR = PROJECT_ROOT / "reports" / "incidents"

# Incident IDs are generated server-side as e.g. "INC-A1B2C3D4" or
# "CALL-0FFB952D". Restrict reads to that shape to prevent path traversal.
_SAFE_ID = re.compile(r"^[A-Za-z0-9-]+$")


def _read_incident(path: Path) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@router.get("")
async def list_incidents():
    """Lists all saved incident evidence records (newest first)."""
    incidents: list[dict] = []
    if INCIDENTS_DIR.is_dir():
        for path in INCIDENTS_DIR.glob("*.json"):
            record = _read_incident(path)
            if record and record.get("incident_id"):
                incidents.append(record)

    incidents.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return {"success": True, "count": len(incidents), "incidents": incidents}


@router.get("/{incident_id}")
async def get_incident(incident_id: str):
    """Returns a single incident evidence record by ID."""
    if not _SAFE_ID.match(incident_id) or ".." in incident_id or "/" in incident_id:
        raise HTTPException(status_code=400, detail="Invalid incident identifier.")

    path = (INCIDENTS_DIR / f"{incident_id}.json").resolve()
    # Ensure the resolved path stays inside the incidents directory.
    try:
        path.relative_to(INCIDENTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid incident identifier.")

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Incident not found.")

    record = _read_incident(path)
    if not record:
        raise HTTPException(status_code=500, detail="Incident record could not be read.")

    return {"success": True, "incident": record}
