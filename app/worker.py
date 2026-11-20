"""The training worker (one training at a time, in queue order), and everything that reads a
training's progress from AI Toolkit's log: steps, loss, speed, test pictures and time left.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import models as mm
from config import AITK, LOSS_RE, SAMPLE_STEP_RE, STEP_RE, current, lock, queue_event
from jobstore import all_jobs, read_state, update_state, write_state
from training import build_config, job_estimate, remember_speed, save_every
from photos import prepare_job_images
from delivery import checkpoints, chosen_checkpoint, deliver, dt_addon, dt_view, notify


# --------------------------------------------------------------------------- #
# Training worker: one job at a time, in queue order
# --------------------------------------------------------------------------- #
def queued_jobs() -> List[tuple]:
    q = [(d, s) for d, s in all_jobs() if s.get("status") == "queued"]
    return sorted(q, key=lambda x: (x[1].get("order", x[1]["created"]), x[1]["created"]))


def next_queued() -> Path | None:
    q = queued_jobs()
    return q[0][0] if q else None


def claim(d: Path) -> Optional[dict]:
    """Mark a waiting job as running, unless it was cancelled a moment ago. Returns its state or None."""
    with lock:
        s = read_state(d)
        if s.get("status") != "queued":
            return None
        s.update(status="running", started=time.time(), error=None)
        write_state(d, s)
        return s


def run_job(d: Path) -> None:
    s = claim(d)
    if s is None:  # cancelled just as its turn came
        return
    with open(d / "train.log", "ab") as log:  # `with` closes the log even if something below fails
        _train(d, s, log)


def _train(d: Path, s: dict, log) -> None:
    def note(msg: str) -> None:
        log.write((time.strftime("%H:%M:%S ") + "[fetch] " + msg + "\n").encode())
        log.flush()

    has_images = (d / "images").exists() and any((d / "images").glob("*.txt"))
    if (d / "recipe.json").exists() and not has_images:
        update_state(d, prep="Getting your photos ready")
        try:
            n = prepare_job_images(d, s, note)
        finally:
            update_state(d, prep=None)
        if n < 5:
            msg = f"Only {n} photos could be used. Please pick at least 5 (20-40 is best) and try again."
            update_state(d, status="failed", finished=time.time(), error=msg)
            notify(d, "failed", msg)
            return
        s = read_state(d)
        if s.get("status") != "running":  # cancelled while the photos were being prepared
            return
    cfg = build_config(d, s)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        ["python", "run.py", str(cfg)], cwd=AITK, stdout=log, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    current.update(job=d.name, proc=proc)
    try:
        if read_state(d).get("status") == "cancelled":  # Cancel pressed while it was starting
            os.killpg(proc.pid, signal.SIGTERM)
        code = proc.wait()
    finally:
        current.update(job=None, proc=None)
    s = read_state(d)
    if s.get("status") != "running":
        # Cancelled (and maybe already resumed, so it's waiting again): leave its status alone.
        return
    if code == 0:
        try:
            info = deliver(d, s)
            update_state(d, status="done", finished=time.time(), result=info, chosen_step=info["step"],
                         delivered_step=info["step"])
            notify(d, "done", f"{s['name']} is trained and ready.")
        except Exception as e:
            update_state(d, status="failed", finished=time.time(), error=str(e))
            notify(d, "failed", str(e))
    else:
        tail = log_tail(d, 30)
        hint = "The GPU ran out of memory. Close other AI apps and press Resume." if "out of memory" in tail.lower() else "Training stopped unexpectedly. Press Resume to try again, or see the technical log."
        update_state(d, status="failed", finished=time.time(), error=hint)
        notify(d, "failed", hint)


def worker() -> None:
    while True:
        d = next_queued()
        if d is None:
            queue_event.wait(timeout=10)
            queue_event.clear()
            continue
        try:
            run_job(d)
        except Exception as e:
            try:
                if read_state(d).get("status") == "running":  # don't turn a cancelled job into "failed"
                    update_state(d, status="failed", finished=time.time(), error=f"Something went wrong: {e}")
                    notify(d, "failed", str(e))
            except Exception:
                time.sleep(5)  # the job folder itself is broken; don't spin on it


def log_tail(d: Path, lines: int = 40) -> str:
    f = d / "train.log"
    if not f.exists():
        return ""
    with open(f, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 64000))
        text = fh.read().decode("utf-8", "replace")
    text = text.replace("\r", "\n")
    return "\n".join([x for x in text.split("\n") if x.strip()][-lines:])


def progress(d: Path, s: dict) -> dict:
    model = mm.model_of(s)
    out = {"step": 0, "total": int(s.get("steps") or mm.preset_cfg(model, s["preset"])["steps"]), "loss": None, "phase": None}
    tail = log_tail(d, 200)
    for line in reversed(tail.split("\n")):
        m = STEP_RE.search(line)
        if m and s["name"] in line:
            out["step"], out["total"] = int(m.group(1)), int(m.group(2))
            lm = LOSS_RE.search(line)
            out["loss"] = float(lm.group(1)) if lm else None
            break
    if s["status"] == "running":
        if s.get("prep"):
            out["phase"] = s["prep"]
        elif out["step"] == 0:
            low = tail.lower()
            last = low.split("\n")[-1] if low else ""
            if "download" in last:
                m = re.search(r"([0-9.]+)\s*gb", last)
                out["phase"] = "First run only: downloading the base AI model" + (f" ({m.group(1)} GB so far)" if m else "")
            elif "generating images" in last or "sampl" in last:
                out["phase"] = "Drawing the 'before' pictures"
            elif "caching" in low:
                out["phase"] = "Studying your photos"
            elif "loading" in low or "quantiz" in low:
                out["phase"] = "Loading the AI model onto the GPU"
            else:
                out["phase"] = (f"Getting ready (the first {mm.MODELS[model]['short']} run also downloads "
                                f"the base model, ~{mm.MODELS[model]['download_gb']} GB)")
        else:
            out["phase"] = "Learning"
        sps = measure_speed(tail, s["name"], save_every(out["total"]))
        if sps:
            out["s_per_step"] = round(sps, 2)
            left_rounds = sum(1 for k in range(out["step"] + 1, out["total"] + 1)
                              if k % save_every(out["total"]) == 0 or k == out["total"])
            out["eta_s"] = int(sps * (out["total"] - out["step"]) + left_rounds * mm.MODELS[model]["sample_round_s"])
            if abs(round(sps, 2) - s.get("speed_saved", 0)) > 0.05:  # remember this GPU's real speed
                remember_speed(s["preset"], sps, model)
                update_state(d, speed_saved=round(sps, 2))
        if out.get("eta_s") is None:
            out["eta_s"] = max(60, job_estimate(s) - int(time.time() - s.get("started", time.time())))
    return out


TQDM_RE = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")


def measure_speed(tail: str, name: str, every: int) -> Optional[float]:
    """Seconds per learning step on this GPU, from ai-toolkit's progress lines.

    Only counts stretches between two saves, so the pauses for test pictures don't inflate it.
    Needs at least 10 steps of evidence (about the first 20 steps of a run).
    """
    pts: Dict[int, float] = {}
    for line in tail.split("\n"):
        if name not in line:
            continue
        m = TQDM_RE.search(line)
        if m:
            h = int(m.group(3) or 0)
            pts[int(m.group(1))] = h * 3600 + int(m.group(4)) * 60 + int(m.group(5))
    groups: Dict[int, list] = {}
    for st, t in sorted(pts.items()):
        if st > 0:
            groups.setdefault((st - 1) // every, []).append((st, t))
    for g in sorted(groups, reverse=True):
        pts_g = groups[g]
        (a, ta), (b, tb) = pts_g[0], pts_g[-1]
        if b - a >= 10 and tb > ta:
            return (tb - ta) / (b - a)
    return None


def samples(d: Path, s: dict) -> list:
    folder = d / "output" / s["name"] / "samples"
    if not folder.exists():
        return []
    items = {}
    for f in sorted(folder.iterdir()):  # names start with a timestamp, so later files win
        m = SAMPLE_STEP_RE.search(f.name)
        if m and f.suffix.lower() in {".jpg", ".png", ".webp"} and not f.name.endswith(".jpg.jpg"):
            # ai-toolkit draws the last step twice (regular + final pictures); keep one set
            items[(int(m.group(1)), int(m.group(2)))] = {"step": int(m.group(1)), "index": int(m.group(2)),
                                                         "url": f"/api/jobs/{d.name}/files/samples/{f.name}"}
    return [items[k] for k in sorted(items)]


def queue_info() -> Dict[str, dict]:
    """Position and 'starts in about' for every waiting job."""
    info, wait = {}, 0
    for d, s in all_jobs():
        if s.get("status") == "running":
            try:
                wait = progress(d, s).get("eta_s") or job_estimate(s)
            except Exception:
                wait = job_estimate(s)
            info[d.name] = {"eta_s": wait}
    for pos, (d, s) in enumerate(queued_jobs(), 1):
        est = job_estimate(s)
        info[d.name] = {"position": pos, "starts_in_s": wait, "est_s": est}
        wait += est
    return info


def public(d: Path) -> dict:
    s = read_state(d)
    s["id"] = d.name
    s["progress"] = progress(d, s)
    smp = samples(d, s)
    s["samples"] = smp
    s["preset_label"] = mm.PRESET_LABELS[s["preset"]]
    model = mm.model_of(s)
    s["model"] = model
    s["model_label"] = mm.MODELS[model]["label"]
    s["dt_model"] = mm.MODELS[model]["dt_model"]
    s["estimate_s"] = job_estimate(s)
    s["queue"] = queue_info().get(d.name)
    thumbs = sorted((d / "images").glob("*.jpg"))[:12]
    s["thumbs"] = [f"/api/jobs/{d.name}/files/images/{t.name}" for t in thumbs]
    # versions to pick from: every saved checkpoint, with the test pictures drawn at that step
    by_step: Dict[int, list] = {}
    for x in smp:
        by_step.setdefault(x["step"], []).append(x)
    cks = checkpoints(d, s)
    total = int(s.get("steps") or mm.preset_cfg(model, s["preset"])["steps"])
    chosen = chosen_checkpoint(d, s)
    s["versions"] = []
    for i, c in enumerate(cks, 1):
        pics = by_step.get(c["step"], [])
        if not pics and c["step"] == total and by_step:  # final pictures may carry the last step number
            pics = by_step.get(max(by_step), [])
        s["versions"].append({"step": c["step"], "number": i, "samples": sorted(pics, key=lambda x: x["index"]),
                              "last": c is cks[-1]})
    s["chosen_step"] = chosen["step"] if chosen else None
    s["before"] = sorted(by_step.get(0, []), key=lambda x: x["index"])  # step 0 = the base model, LoRA still blank
    s["drawthings"] = dt_view(d, s) if s.get("status") == "done" else {"addon": bool(dt_addon())}
    s.pop("origin", None)
    return s
