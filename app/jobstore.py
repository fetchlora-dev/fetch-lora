"""Reading and writing the small JSON files Fetch LoRA keeps its state in.

Each training has a folder data/jobs/<job-id>/ with a job.json inside. App-wide settings
(the webhook address and the GPU speeds it has learned) live in data/jobs/.settings.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import HTTPException

from config import ID_RE, JOBS, SETTINGS_FILE, lock
from safepath import safe_join


# --------------------------------------------------------------------------- #
# Job storage helpers
# --------------------------------------------------------------------------- #
def job_dir(job_id: str) -> Path:
    if not ID_RE.fullmatch(job_id):
        raise HTTPException(404, "No such job")
    d = safe_join(JOBS, job_id)
    if not (d / "job.json").exists():
        raise HTTPException(404, "No such job")
    return d


def read_state(d: Path) -> dict:
    return json.loads((d / "job.json").read_text())


def write_state(d: Path, state: dict) -> None:
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(d / "job.json")


def update_state(d: Path, **changes) -> dict:
    with lock:
        s = read_state(d)
        s.update(changes)
        write_state(d, s)
        return s


def all_jobs() -> List[tuple]:
    out = []
    for d in JOBS.iterdir():
        if not d.is_dir() or not ID_RE.fullmatch(d.name):
            continue
        try:
            out.append((d, read_state(d)))
        except Exception:
            continue
    return out


def read_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def write_settings(s: dict) -> None:
    """Write to a temp file, then swap it in, so a reader never sees a half-written file
    (read_settings() would then return {} and the webhook address would look gone)."""
    tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(SETTINGS_FILE)
