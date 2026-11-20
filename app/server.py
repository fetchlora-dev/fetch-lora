"""Fetch LoRA - a one-button LoRA trainer you drive from an iPad or iPhone.

A thin web app on top of ostris/ai-toolkit (MIT). Pick photos, type a name,
press Start; the PC's GPU trains a LoRA and this page shows it learning.

This file wires the app together: the web pages, the PIN lock, photo uploads and settings.
The rest lives in small modules (see docs/HOW_IT_WORKS.md):

    config.py    settings, folders and shared constants
    jobstore.py  reading and writing job.json and .settings.json
    training.py  steps, time estimates and the AI Toolkit recipe
    photos.py    opening photos, tags and captions, drafts
    delivery.py  finished LoRAs, Draw Things, notifications
    worker.py    the training queue and reading progress from the log
    auth.py      the PIN lock
    jobs_api.py  the web API for trainings
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import models as mm
import photocheck
from config import BASE_MODEL, IMG_EXT, KINDS, MAX_PHOTOS, MAX_PHOTO_MB, PRESETS, STATIC, lock
from jobstore import all_jobs, read_settings, write_settings, write_state
from training import estimate_seconds, gpu_stats, plan_steps
from photos import add_draft_photo, cleanup_drafts, draft_dir, draft_view, new_draft
from delivery import dt_addon, send_webhook
from worker import queue_info, worker
import auth
import jobs_api


app = FastAPI(title="Fetch LoRA")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(auth.router)       # /login, /api/login, /api/logout
app.include_router(jobs_api.router)   # /api/jobs/...


@app.middleware("http")
async def upload_size_limit(request: Request, call_next):
    """FastAPI saves a whole upload to a temp file before upload_photo() even runs, so turn away
    photo uploads that say up front (Content-Length) that they're too big."""
    if request.method == "POST" and request.url.path.endswith("/photos"):
        try:
            size = int(request.headers.get("content-length", "0"))
        except ValueError:
            size = 0
        if size > (MAX_PHOTO_MB + 1) * 1024 * 1024:  # +1 MB for the form around the photo
            return JSONResponse({"detail": f"That photo is bigger than {MAX_PHOTO_MB} MB."}, status_code=413)
    return await call_next(request)


# The PIN check is added last so it runs first, before every other step.
app.middleware("http")(auth.pin_gate)


@app.on_event("startup")
def startup() -> None:
    for d, s in all_jobs():
        try:
            if s.get("status") == "running":
                s.update(status="interrupted", prep=None,
                         error="The PC or trainer restarted. Press Resume to continue where it left off.")
                write_state(d, s)
        except Exception:
            pass
    cleanup_drafts()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=photocheck._detector, daemon=True).start()  # warm up the photo checker


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/info")
def info() -> dict:
    g = gpu_stats()
    return {"base_model": BASE_MODEL, "presets": PRESETS, "kinds": {k: v[0] for k, v in KINDS.items()},
            "gpu": g, "models": mm.catalog(g), "drawthings_addon": bool(dt_addon())}


@app.get("/api/models")
def list_models() -> dict:
    """The model matrix for the picker: labels, notes, and whether THIS PC's GPU can train each one."""
    g = gpu_stats()
    return {"gpu": g, "default": mm.DEFAULT_MODEL, "models": mm.catalog(g)}


@app.get("/api/estimate")
def estimate(photos: int = 20, steps: int = 0, bg_remove: bool = False) -> dict:
    out = {"models": {}}
    for k in PRESETS:  # top level: Klein 4B, as before
        st = plan_steps(k, photos, steps or None)
        out[k] = {"steps": st, "seconds": estimate_seconds(k, photos, st, bg_remove)}
    for m, spec in mm.MODELS.items():  # every model x level, for the picker
        out["models"][m] = {}
        for k in spec["presets"]:
            st = plan_steps(k, photos, steps or None, m)
            out["models"][m][k] = {"steps": st, "seconds": estimate_seconds(k, photos, st, bg_remove, m)}
    running = [s for _, s in all_jobs() if s.get("status") in ("running", "queued")]
    waits = queue_info()
    out["wait_s"] = sum(v.get("eta_s", 0) for v in waits.values() if "eta_s" in v) + \
        sum(v.get("est_s", 0) for v in waits.values() if "est_s" in v)
    out["busy"] = len(running)
    return out


@app.get("/api/gpu")
def gpu() -> dict:
    return gpu_stats() or {}


# ------------------------------- photo drafts ------------------------------ #
@app.post("/api/drafts")
def create_draft() -> dict:
    return {"id": new_draft().name}


@app.post("/api/drafts/{draft_id}/photos")
def upload_photo(draft_id: str, photo: UploadFile = File(...), tags: str = Form("")) -> dict:
    dd = draft_dir(draft_id)
    ext = Path(photo.filename or "").suffix.lower() or ".jpg"
    if ext not in IMG_EXT:
        raise HTTPException(400, f"{photo.filename} isn't a photo.")
    if len(list((dd / "meta").glob("*.json"))) >= MAX_PHOTOS:
        raise HTTPException(400, f"That's plenty of photos: one training can use at most {MAX_PHOTOS}.")
    tmp = dd / f"upload_{uuid.uuid4().hex}{ext}"
    try:
        size, limit = 0, MAX_PHOTO_MB * 1024 * 1024
        with open(tmp, "wb") as fh:
            while chunk := photo.file.read(1024 * 1024):  # copy 1 MB at a time, stop if it's too big
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"{photo.filename} is bigger than {MAX_PHOTO_MB} MB.")
                fh.write(chunk)
        try:
            r = add_draft_photo(dd, tmp, photo.filename or "", tags)
        except Exception:
            raise HTTPException(400, f"Couldn't open {photo.filename}. Is it a photo?")
    finally:
        tmp.unlink(missing_ok=True)
    os.utime(dd, None)
    return r


@app.get("/api/drafts/{draft_id}")
def get_draft(draft_id: str, kind: str = "dog", keep: str = "") -> dict:
    """`keep` is an optional comma-separated list of the photo ids still chosen, e.g. "0,2,3"."""
    ids = [int(x) for x in keep.split(",") if x.strip().isdigit()] if keep else None
    return draft_view(draft_dir(draft_id), kind, ids)


@app.get("/api/drafts/{draft_id}/thumb/{pid}")
def draft_thumb(draft_id: str, pid: int):
    f = draft_dir(draft_id) / "thumbs" / f"{int(pid):04d}.jpg"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


# -------------------------------- settings --------------------------------- #
@app.get("/api/settings")
def get_settings() -> dict:
    s = read_settings()
    return {"webhook_url": s.get("webhook_url", "")}


@app.post("/api/settings")
def save_settings(body: dict = Body(...)) -> dict:
    url = str(body.get("webhook_url", "")).strip()
    if url and not re.match(r"^https?://\S+$", url):
        raise HTTPException(400, "That doesn't look like a web address (it should start with http:// or https://).")
    with lock:  # the trainer also writes this file (learned GPU speeds); don't lose either change
        s = read_settings()
        s["webhook_url"] = url
        write_settings(s)
    return {"ok": True}


@app.post("/api/settings/test")
def test_webhook(request: Request) -> dict:
    url = read_settings().get("webhook_url", "").strip()
    if not url:
        raise HTTPException(400, "Save a notification address first.")
    try:
        code = send_webhook(url, {"job": "test", "name": "test", "status": "test",
                                  "message": "Fetch LoRA can reach your phone. 🎉", "url": str(request.base_url)})
        return {"ok": True, "message": f"Sent (answer {code})."}
    except Exception as e:
        raise HTTPException(400, f"Couldn't reach that address: {e}")
