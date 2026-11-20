"""The web API for trainings ("jobs"): start one, watch it, cancel, resume, reorder the line,
pick a version, send it to Draw Things, train again, and download its files.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import time
import uuid

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import models as mm
from config import DT_DRY_RUN, DT_STATUS, EXPORTS, JOBS, KINDS, NAME_RE, PRESETS, current, lock, queue_event
from jobstore import all_jobs, job_dir, read_state, update_state, write_state
from training import gpu_stats, plan_steps
from safepath import safe_join, safe_name
from photos import add_draft_photo, draft_dir, draft_metas, draft_view, new_draft, parse_tags
from delivery import checkpoints, chosen_checkpoint, deliver, dt_addon
from worker import log_tail, progress, public, queue_info, queued_jobs


router = APIRouter()  # server.py adds these to the app


# ---------------------------------- jobs ----------------------------------- #
@router.get("/api/jobs")
def list_jobs() -> list:
    q = queue_info()
    out = []
    for d, s in all_jobs():
        try:
            item = {"id": d.name, "name": s["name"], "status": s["status"], "created": s["created"],
                    "preset_label": mm.PRESET_LABELS[s["preset"]] + ("" if mm.model_of(s) == mm.DEFAULT_MODEL
                                                                      else " · " + mm.MODELS[mm.model_of(s)]["short"]),
                    "photos": s.get("photos")}
            if d.name in q:
                item.update(q[d.name])
            if s["status"] == "running":
                p = progress(d, s)
                item["pct"] = round(100 * p["step"] / p["total"]) if p["total"] else 0
                item["phase"] = p.get("phase")
            out.append(item)
        except Exception:
            continue
    rank = {"running": 0, "queued": 1}
    return sorted(out, key=lambda x: (rank.get(x["status"], 2), x.get("position", 0), -x["created"]))


@router.post("/api/jobs")
def create_job(request: Request, body: dict = Body(...)) -> dict:
    name = str(body.get("name", "")).strip().lower().replace(" ", "_").replace("-", "_")
    kind, preset = str(body.get("kind") or "dog"), str(body.get("preset") or "good")
    if not NAME_RE.match(name):
        raise HTTPException(400, "Name must start with a letter and use only letters, numbers and _ (3-40 characters), like biscuit_dog")
    # The finished LoRA is saved as data/loras/<name>.safetensors, so two trainings with the same
    # name would overwrite each other. Refuse while the other one is still going, and otherwise
    # only replace it when the page says so (it asks the person first).
    same_name = [(od, st) for od, st in all_jobs() if st.get("name") == name and not st.get("replaced_by")]
    if any(st.get("status") in ("queued", "running") for _, st in same_name):
        raise HTTPException(400, f"{name} is already training or waiting its turn. Pick another name, or wait until it's done.")
    if same_name and not body.get("replace"):
        raise HTTPException(409, f"You already have a training called {name}. Starting this one will replace its LoRA file.")
    model = str(body.get("model") or mm.DEFAULT_MODEL)
    if kind not in KINDS or preset not in PRESETS or model not in mm.MODELS:
        raise HTTPException(400, "Unknown option")
    if preset == "quick":
        model = mm.DEFAULT_MODEL  # Quick test is always Klein 4B
    can = mm.availability(model, preset, gpu_stats())
    if not can["ok"]:
        raise HTTPException(400, f"{mm.MODELS[model]['label']}: {can['reason']}")
    dd = draft_dir(str(body.get("draft", "")))
    metas = {m["pid"]: m for m in draft_metas(dd)}
    try:
        keep = list(dict.fromkeys(int(p) for p in body.get("keep", list(metas))))  # each photo once, in order
        tags = {int(k): str(v) for k, v in (body.get("tags") or {}).items()}
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, "The list of chosen photos didn't make sense. Please pick them again.")
    keep = [p for p in keep if p in metas]
    if len(keep) < 5:
        raise HTTPException(400, f"Only {len(keep)} photos left. Please keep at least 5 (20-40 is best).")
    autocrop = body.get("autocrop")
    autocrop = (KINDS[kind][1] != "style") if autocrop is None else bool(autocrop)
    try:
        steps_override = int(body.get("steps") or 0) or None
    except (TypeError, ValueError):
        steps_override = None
    job_id = uuid.uuid4().hex[:12]
    d = JOBS / job_id
    (d / "originals").mkdir(parents=True)
    recipe = {"photos": []}
    for i, pid in enumerate(keep, 1):
        f = f"{i:03d}.jpg"
        shutil.copy2(dd / "photos" / f"{pid:04d}.jpg", d / "originals" / f)
        m = dict(metas[pid])
        t = tags.get(pid, m.get("tags", ""))
        recipe["photos"].append({"file": f, "tags": t, "name": m.pop("name", ""), "meta": m})
    (d / "recipe.json").write_text(json.dumps(recipe, indent=1))
    n = len(keep)
    tagged = sum(1 for p in recipe["photos"] if parse_tags(p["tags"], name))
    state = {"name": name, "kind": kind, "preset": preset, "photos": n, "status": "queued",
             "created": time.time(), "order": time.time(), "model": model,
             "base_model": mm.MODELS[model]["repo"], "tagged": tagged,
             "steps": plan_steps(preset, n, steps_override, model),
             "options": {"autocrop": autocrop, "bg_remove": bool(body.get("bg_remove")), "steps_override": steps_override},
             "origin": str(request.base_url)}
    write_state(d, state)
    for od, _ in same_name:  # the old ones stay downloadable, but no longer own data/loras/<name>
        update_state(od, replaced_by=job_id)
    shutil.rmtree(dd, ignore_errors=True)
    queue_event.set()
    return {"id": job_id, "photos": n, "queue": queue_info().get(job_id)}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return public(job_dir(job_id))


@router.get("/api/jobs/{job_id}/log")
def get_log(job_id: str) -> JSONResponse:
    return JSONResponse({"log": log_tail(job_dir(job_id), 80)})


@router.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    d = job_dir(job_id)
    s = read_state(d)
    if s["status"] in ("queued", "running"):
        update_state(d, status="cancelled", finished=time.time())
        proc = current.get("proc")
        if current.get("job") == job_id and proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    return {"ok": True}


@router.post("/api/jobs/{job_id}/resume")
def resume(job_id: str) -> dict:
    d = job_dir(job_id)
    s = read_state(d)
    if s.get("replaced_by"):
        raise HTTPException(409, "A newer training with the same name replaced this one, so it can't be resumed.")
    if s["status"] in ("failed", "cancelled", "interrupted"):
        # ai-toolkit picks up from the newest saved checkpoint automatically
        update_state(d, status="queued", error=None, train_started=None, finished=None, order=time.time())
        queue_event.set()
    return {"ok": True}


@router.post("/api/jobs/{job_id}/move")
def move(job_id: str, direction: str = Form(...)) -> dict:
    """Reorder the waiting line: direction is 'up' or 'down'."""
    d = job_dir(job_id)
    with lock:
        q = queued_jobs()
        ids = [x[0].name for x in q]
        if job_id not in ids:
            raise HTTPException(400, "Only waiting trainings can be moved.")
        i = ids.index(job_id)
        j = i - 1 if direction == "up" else i + 1
        if 0 <= j < len(q):
            a, b = q[i], q[j]
            oa, ob = a[1].get("order", a[1]["created"]), b[1].get("order", b[1]["created"])
            if oa == ob:
                ob += 0.001 if j > i else -0.001
            a[1]["order"], b[1]["order"] = ob, oa
            write_state(a[0], a[1])
            write_state(b[0], b[1])
    return {"ok": True, "queue": queue_info().get(d.name)}


@router.post("/api/jobs/{job_id}/choose")
def choose(job_id: str, step: int = Form(...)) -> dict:
    """Pick which saved version becomes 'the' LoRA (download, Draw Things)."""
    d = job_dir(job_id)
    s = read_state(d)
    if step not in [c["step"] for c in checkpoints(d, s)]:
        raise HTTPException(400, "That version isn't saved.")
    s = update_state(d, chosen_step=step, dt=None)
    if s.get("status") == "done" and s.get("delivered_step") != step and not s.get("replaced_by"):
        info = deliver(d, s)
        update_state(d, result=info, delivered_step=step)
    return {"ok": True}


@router.post("/api/jobs/{job_id}/drawthings")
def send_to_drawthings(job_id: str) -> dict:
    """Convert the chosen version (the add-on does this by itself) and restart the Draw Things server."""
    d = job_dir(job_id)
    s = read_state(d)
    if s.get("status") != "done":
        raise HTTPException(400, "This training isn't finished yet.")
    if s.get("replaced_by"):
        raise HTTPException(409, "A newer training with the same name replaced this one. Download it instead.")
    if not mm.MODELS[mm.model_of(s)]["dt_send"]:
        raise HTTPException(409, f"The Draw Things server on this PC only has Klein 4B. Download this LoRA and import it "
                                 f"in the Draw Things app, then use it with {mm.MODELS[mm.model_of(s)]['dt_model']}.")
    hb = dt_addon()
    if not hb:
        raise HTTPException(409, "The Draw Things helper isn't running on this PC, so use Download instead.")
    if not hb.get("can_restart"):
        raise HTTPException(409, "The Draw Things helper can't restart the server by itself. Restart the Draw Things server on the PC once, and the LoRA will show up.")
    ck = chosen_checkpoint(d, s)
    if ck and s.get("delivered_step") != ck["step"]:
        info = deliver(d, s)
        s = update_state(d, result=info, delivered_step=ck["step"])
    src = EXPORTS / f"{s['name']}.safetensors"
    if not src.exists():
        info = deliver(d, s)
        s = update_state(d, result=info, delivered_step=info["step"])
    rid = uuid.uuid4().hex[:12]
    req = {"id": rid, "name": s["name"], "source_mtime": src.stat().st_mtime, "dry_run": DT_DRY_RUN, "at": time.time()}
    DT_STATUS.mkdir(parents=True, exist_ok=True)
    tmp = DT_STATUS / f".restart-request-{rid}.tmp"
    tmp.write_text(json.dumps(req))
    tmp.replace(DT_STATUS / f"restart-request-{rid}.json")
    update_state(d, dt={"request": rid, "at": time.time()})
    return {"ok": True, "dry_run": DT_DRY_RUN}


@router.post("/api/jobs/{job_id}/again")
def train_again(job_id: str) -> dict:
    """Load a past training's photos, tags and settings into a new, editable set."""
    d = job_dir(job_id)
    s = read_state(d)
    dd = new_draft()
    if (d / "recipe.json").exists():
        recipe = json.loads((d / "recipe.json").read_text())
        for p in recipe["photos"]:
            src = d / "originals" / p["file"]
            if src.exists():
                add_draft_photo(dd, src, p.get("name", ""), p.get("tags", ""), dict(p.get("meta") or {}))
    else:  # trainings from before recipes: use the prepared photos and their tags
        tag_map = json.loads((d / "tags.json").read_text()) if (d / "tags.json").exists() else {}
        for src in sorted((d / "images").glob("*.jpg")):
            add_draft_photo(dd, src, src.name, ", ".join(tag_map.get(src.name, [])))
    o = s.get("options", {})
    settings = {"name": s["name"], "kind": s["kind"], "preset": s["preset"], "model": mm.model_of(s),
                "autocrop": o.get("autocrop", KINDS[s["kind"]][1] != "style"),
                "bg_remove": bool(o.get("bg_remove")), "steps": o.get("steps_override") or ""}
    return {"draft": draft_view(dd, s["kind"]), "settings": settings}


@router.get("/api/jobs/{job_id}/files/{kind}/{fname}")
def job_file(job_id: str, kind: str, fname: str):
    d = job_dir(job_id)
    s = read_state(d)
    safe_name(fname)
    if kind == "samples":
        f = safe_join(d / "output" / s["name"] / "samples", fname)
    elif kind == "images":
        f = safe_join(d / "images", fname)
    elif kind == "lora":
        ck = chosen_checkpoint(d, s)
        if not ck:
            raise HTTPException(404)
        return FileResponse(ck["file"], filename=f"{s['name']}.safetensors", media_type="application/octet-stream")
    else:
        raise HTTPException(404)
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)
