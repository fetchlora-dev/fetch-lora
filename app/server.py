"""Fetch LoRA - a one-button LoRA trainer you drive from an iPad or iPhone.

A thin web app on top of ostris/ai-toolkit (MIT). Pick photos, type a name,
press Start; the PC's GPU trains a LoRA and this page shows it learning.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

import models as mm
import photocheck

try:  # iPhone/iPad photos from the Files app may still be HEIC
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------- #
# Settings (all overridable with environment variables)
# --------------------------------------------------------------------------- #
DATA = Path(os.environ.get("DRAGON_DATA", "/data"))
JOBS = DATA / "jobs"
DRAFTS = JOBS / ".drafts"          # photos picked but not started yet (checked, waiting for Start)
SETTINGS_FILE = JOBS / ".settings.json"
EXPORTS = Path(os.environ.get("DRAGON_EXPORTS", "/exports"))
AITK = Path(os.environ.get("AITK_DIR", "/app/ai-toolkit"))
# Base models (Klein 4B, Z-Image, Qwen-Image) live in models.py; Klein 4B is the default.
BASE_MODEL = mm.MODELS[mm.DEFAULT_MODEL]["repo"]
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# Optional: extra folders the finished LoRA is copied into (comma separated),
# e.g. a folder that a Draw Things gRPC server can see.
DELIVER_TO = [Path(p) for p in os.environ.get("DRAGON_DELIVER_TO", "").split(",") if p.strip()]
# The optional fetch-drawthings container (converter/, GPL-3.0) converts finished LoRAs for
# Draw Things, leaves a status file per LoRA here, and restarts the Draw Things server on request.
DT_STATUS = EXPORTS / ".drawthings"
# Test mode for "Send to Draw Things": ask the add-on to check everything but not restart.
DT_DRY_RUN = os.environ.get("DRAGON_DT_DRY_RUN", "0") == "1"
HERE = Path(__file__).parent

JOBS.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)
DRAFTS.mkdir(parents=True, exist_ok=True)

KINDS = {
    "dog": ("a dog", "subject"),
    "cat": ("a cat", "subject"),
    "pet": ("an animal", "subject"),
    "person": ("a person", "subject"),
    "thing": ("an object", "subject"),
    "style": ("an art style", "style"),
}

# Steps, speeds and per-model settings live in models.py (the model matrix). PRESETS keeps the
# Klein 4B numbers under the old names for anything that still reads them.
PRESETS = {k: dict(v, label=mm.PRESET_LABELS[k]) for k, v in mm.MODELS[mm.DEFAULT_MODEL]["presets"].items()}

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
ID_RE = re.compile(r"[0-9a-f]{12}")
STEP_RE = re.compile(r"(\d+)/(\d+) \[")
LOSS_RE = re.compile(r"loss:\s*([0-9.eE+-]+)")
SAMPLE_STEP_RE = re.compile(r"__(\d{9})_(\d+)\.")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}

lock = threading.Lock()
draft_lock = threading.Lock()
queue_event = threading.Event()
current = {"job": None, "proc": None}


# --------------------------------------------------------------------------- #
# Job storage helpers
# --------------------------------------------------------------------------- #
def job_dir(job_id: str) -> Path:
    if not ID_RE.fullmatch(job_id):
        raise HTTPException(404, "No such job")
    d = JOBS / job_id
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


# --------------------------------------------------------------------------- #
# Planning: steps, checkpoints, time estimates
# --------------------------------------------------------------------------- #
def plan_steps(preset: str, photos: int, override: Optional[int] = None, model: str = mm.DEFAULT_MODEL) -> int:
    if override:
        return max(50, min(8000, int(override)))
    p = mm.preset_cfg(model, preset)
    if not p["per_photo"]:
        return p["steps"]
    return max(p["min"], min(p["max"], int(photos) * p["per_photo"]))


def save_every(steps: int) -> int:
    """About 5 saved versions over the run (3 for short runs), each with its own test pictures."""
    n = 3 if steps <= 400 else 5
    return max(10, int(math.ceil(steps / n / 10.0)) * 10)


def measured_speed(preset: str, model: str = mm.DEFAULT_MODEL) -> float:
    """Seconds per step on THIS PC's GPU, learned from earlier trainings (default: our RTX 5070 Ti numbers)."""
    try:
        v = float(read_settings().get("speed", {}).get(mm.speed_key(model, preset), 0))
        if 0.05 < v < 60:
            return v
    except Exception:
        pass
    return mm.preset_cfg(model, preset)["s_per_step"]


def remember_speed(preset: str, sps: float, model: str = mm.DEFAULT_MODEL) -> None:
    try:
        with lock:
            st = read_settings()
            st.setdefault("speed", {})[mm.speed_key(model, preset)] = round(sps, 3)
            SETTINGS_FILE.write_text(json.dumps(st, indent=2))
    except Exception:
        pass


def estimate_seconds(preset: str, photos: int, steps: int, bg_remove: bool = False,
                     model: str = mm.DEFAULT_MODEL) -> int:
    m = mm.MODELS[model]
    p = mm.preset_cfg(model, preset)
    every = save_every(steps)
    # step 0 + each saved version (+ ai-toolkit draws the last step twice when it lands on a save)
    rounds = 1 + int(math.ceil(steps / every)) + (1 if steps % every == 0 else 0)
    prep = photos * (2.0 if bg_remove else 0.3) + (60 if bg_remove else 0)
    caching = photos * len(p["res"]) * 0.5
    return int(m["load_s"] + steps * measured_speed(preset, model) + rounds * m["sample_round_s"] + prep + caching)


def job_estimate(s: dict) -> int:
    o = s.get("options", {})
    model = mm.model_of(s)
    steps = s.get("steps") or mm.preset_cfg(model, s["preset"])["steps"]
    return estimate_seconds(s["preset"], s.get("photos", 20), steps, bool(o.get("bg_remove")), model)


def sample_prompts(trigger: str, kind: str) -> List[str]:
    cls, mode = KINDS[kind]
    if mode == "style":
        return [
            f"a lighthouse on a cliff at sunset, in the style of {trigger}",
            f"a portrait of an old fisherman, in the style of {trigger}",
            f"a cozy kitchen with a cat on the table, in the style of {trigger}",
        ]
    return [
        f"a photo of {trigger}, {cls}, sitting in a sunny meadow",
        f"{trigger}, {cls}, as an astronaut on the moon, cinematic lighting",
        f"a renaissance oil painting of {trigger}, {cls}, in a golden frame",
    ]


def build_config(d: Path, s: dict) -> Path:
    model = mm.model_of(s)
    p = mm.preset_cfg(model, s["preset"])
    steps = int(s.get("steps") or p["steps"])
    gpu_total = (gpu_stats() or {}).get("mem_total_gb")
    low_vram = os.environ.get("DRAGON_LOW_VRAM", "0") == "1"
    every = save_every(steps)
    trigger, kind = s["name"], s["kind"]
    cls, mode = KINDS[kind]
    cfg = {
        "job": "extension",
        "config": {
            "name": trigger,
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": str(d / "output"),
                    "device": "cuda",
                    "trigger_word": trigger,
                    "performance_log_every": 10,
                    "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                    "save": {
                        "dtype": "bf16",
                        "save_every": every,  # ~5 versions to pick from on the job page
                        "max_step_saves_to_keep": 8,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": str(d / "images"),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "cache_latents_to_disk": True,
                            "resolution": mm.resolutions(model, s["preset"], gpu_total),
                            "flip_x": False,
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "timestep_type": "weighted",
                        "content_or_style": "balanced",
                        "optimizer_params": {"weight_decay": 1e-4},
                        "unload_text_encoder": False,
                        "cache_text_embeddings": False,  # True crashes with an 8-bit text encoder in ai-toolkit 0.13.42 + torch 2.13
                        "lr": 1e-4,
                        "dtype": "bf16",
                        "skip_first_sample": False,
                    },
                    "model": mm.model_block(model, gpu_total, low_vram),
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": every,  # test pictures for every saved version
                        "width": 768,
                        "height": 768,
                        "samples": [{"prompt": x} for x in sample_prompts(trigger, kind)],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": False,
                        "guidance_scale": mm.MODELS[model]["sample"]["guidance_scale"],
                        "sample_steps": mm.MODELS[model]["sample"]["sample_steps"],
                    },
                }
            ],
        },
        "meta": {"name": trigger, "version": "1.0"},
    }
    cfg["config"]["process"][0]["train"].update(mm.train_overrides(model, gpu_total))
    path = d / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


MAX_TAGS = 12
TAG_MAX_LEN = 40
_WORD_RE = re.compile(r"[a-z0-9]+")


def parse_tags(raw, trigger: str = "") -> List[str]:
    """Turn 'frisbee, mid-air , Left side' into ['frisbee', 'mid-air', 'left side'].

    Lowercased, trimmed, no duplicates, no trigger word, at most MAX_TAGS short tags.
    """
    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(x) for x in raw)
    out: List[str] = []
    for part in str(raw or "").split(","):
        t = " ".join("".join(c if c.isprintable() else " " for c in part).split()).lower()
        t = t[:TAG_MAX_LEN].strip()
        if not t or t == trigger.lower() or t in out:
            continue
        out.append(t)
        if len(out) >= MAX_TAGS:
            break
    return out


def merge_caption(trigger: str, tags: List[str], auto: str, auto_without_trigger: str = "") -> str:
    """Caption for one photo.

    No tags: the auto caption, exactly as before.
    With tags: "<trigger>, <tags...>, <auto caption>", skipping tags whose words the
    auto caption already has (e.g. "dog") so nothing is said twice.
    """
    tags = parse_tags(tags, trigger)
    rest = auto_without_trigger or auto
    have = set(_WORD_RE.findall(rest.lower())) | {trigger.lower()}
    tags = [t for t in tags if not set(_WORD_RE.findall(t)) <= have]
    if not tags:
        return auto
    return ", ".join([trigger, *tags, rest])


def caption_parts(trigger: str, kind: str) -> tuple:
    cls, mode = KINDS[kind]
    if mode == "style":
        return f"an image in the style of {trigger}", "an image in this style"
    return f"a photo of {trigger}, {cls}", f"a photo of {cls}"


def open_photo(src: Path, max_side: int) -> Image.Image:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def prepare_job_images(d: Path, s: dict, log) -> int:
    """originals/ (+ recipe.json) -> images/ with captions: auto-crop, optional background removal."""
    recipe = json.loads((d / "recipe.json").read_text())
    trigger, kind = s["name"], s["kind"]
    opts = s.get("options", {})
    caption, caption_rest = caption_parts(trigger, kind)
    out = d / "images"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    tag_map, n, cropped, notes = {}, 0, 0, []
    for item in recipe["photos"]:
        try:
            im = open_photo(d / "originals" / item["file"], 4096)
            box = None
            if opts.get("autocrop"):
                try:
                    box = photocheck.crop_box(item.get("meta", {}), kind)
                except Exception as e:
                    log(f"auto-crop skipped for {item['file']}: {e}")
            if box:
                im = im.crop(tuple(box))
                cropped += 1
            im.thumbnail((1536, 1536))
            if min(im.size) < photocheck.SMALL_MIN:
                continue
            n += 1
            im.save(out / f"{n:03d}.jpg", quality=95)
            my_tags = parse_tags(item.get("tags", ""), trigger)
            (out / f"{n:03d}.txt").write_text(merge_caption(trigger, my_tags, caption, caption_rest))
            if my_tags:
                tag_map[f"{n:03d}.jpg"] = my_tags
        except Exception as e:
            log(f"skipped {item.get('file')}: {e}")
    if tag_map:  # kept in the job folder, outside images/ so ai-toolkit never sees it
        (d / "tags.json").write_text(json.dumps(tag_map, indent=2))
    log(f"prepared {n} photos ({cropped} cropped around the subject)")
    if opts.get("bg_remove") and n:
        update_state(d, prep="Removing backgrounds")
        try:
            r = subprocess.run([sys.executable, str(HERE / "bgremove.py"), str(out)], capture_output=True,
                               text=True, timeout=3600, cwd=HERE)
            log((r.stdout + r.stderr)[-3000:])
            if r.returncode != 0:
                raise RuntimeError(f"exit {r.returncode}")
        except Exception as e:  # graceful: train on the photos as they are
            log(f"background removal failed: {e}")
            notes.append("Background removal didn't work this time, so your photos were used as they were.")
    update_state(d, photos=n, cropped=cropped, tagged=len(tag_map), notes=notes)
    return n


# --------------------------------------------------------------------------- #
# Finished LoRA: versions (checkpoints), delivery, Draw Things, notifications
# --------------------------------------------------------------------------- #
def checkpoints(d: Path, s: dict) -> List[dict]:
    folder = d / "output" / s["name"]
    out = {}
    if folder.exists():
        for f in folder.glob(f"{s['name']}*.safetensors"):
            m = re.fullmatch(re.escape(s["name"]) + r"_(\d{9})\.safetensors", f.name)
            if m:
                out[int(m.group(1))] = f
            elif f.name == f"{s['name']}.safetensors":
                out[int(s.get("steps") or mm.preset_cfg(mm.model_of(s), s["preset"])["steps"])] = f
    return [{"step": k, "file": v} for k, v in sorted(out.items())]


def chosen_checkpoint(d: Path, s: dict) -> Optional[dict]:
    cks = checkpoints(d, s)
    if not cks:
        return None
    for c in cks:
        if c["step"] == s.get("chosen_step"):
            return c
    return cks[-1]  # default: the last (most practiced) version


def copy_atomic(src: Path, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")  # not *.safetensors, so watchers ignore it mid-copy
    shutil.copy2(src, part)
    os.utime(part, None)  # fresh mtime: a different version always looks new to the converter
    part.replace(dest)


def deliver(d: Path, s: dict) -> dict:
    """Copy the chosen version where people (and Draw Things) can find it."""
    ck = chosen_checkpoint(d, s)
    if ck is None:
        raise RuntimeError("Training finished but the LoRA file is missing")
    name = f"{s['name']}.safetensors"
    sub = lora_subfolder(s)
    (EXPORTS / sub).mkdir(parents=True, exist_ok=True)
    copy_atomic(ck["file"], EXPORTS / sub / name)
    copied = []
    for target in DELIVER_TO:
        try:
            (target / sub).mkdir(parents=True, exist_ok=True)
            copy_atomic(ck["file"], target / sub / name)
            copied.append(str(target))
        except Exception as e:  # keep going, report it
            copied.append(f"{target} (failed: {e})")
    return {"file": name, "size_mb": round(ck["file"].stat().st_size / 1e6, 1), "copied_to": copied,
            "step": ck["step"]}


def lora_subfolder(s: dict) -> str:
    """'' for Klein 4B (the Draw Things add-on converts those); e.g. 'zimage' for other models.

    The add-on only looks at *.safetensors directly in the export folder and converts them as
    Klein 4B, so LoRAs for other base models go one folder down where it never looks.
    """
    model = mm.model_of(s)
    return "" if mm.MODELS[model]["dt_send"] else model


def dt_addon() -> Optional[dict]:
    """The Draw Things add-on's heartbeat, if it's running (it writes one every few seconds)."""
    try:
        hb = json.loads((DT_STATUS / ".alive").read_text())
        if time.time() - hb.get("at", 0) < 90:
            return hb
    except Exception:
        pass
    return None


def dt_view(d: Path, s: dict) -> dict:
    hb = dt_addon()
    view = {"addon": bool(hb), "can_restart": bool(hb and hb.get("can_restart")), "state": "idle"}
    if not mm.MODELS[mm.model_of(s)]["dt_send"]:  # the PC's Draw Things server only has Klein 4B
        view.update(unsupported=True, can_restart=False)
        return view
    src = EXPORTS / f"{s['name']}.safetensors"
    try:
        st = json.loads((DT_STATUS / f"{s['name']}.json").read_text())
        converted = bool(st.get("ok")) and src.exists() and st.get("source_mtime") == src.stat().st_mtime
        view["converted"] = converted
        if st.get("ok") is False and src.exists() and st.get("source_mtime") == src.stat().st_mtime:
            view["convert_error"] = True
    except Exception:
        view["converted"] = False
    req = (s.get("dt") or {}).get("request")
    if req:
        res = DT_STATUS / f"restart-result-{req}.json"
        if res.exists():
            try:
                r = json.loads(res.read_text())
                view["state"] = "ready" if r.get("ok") else "failed"
                view["message"] = r.get("message", "")
                view["dry_run"] = bool(r.get("dry_run"))
            except Exception:
                view["state"] = "failed"
        elif view.get("convert_error"):
            view["state"] = "failed"
            view["message"] = "Draw Things couldn't convert this LoRA. You can still download it below."
        else:
            view["state"] = "restarting" if view["converted"] else "converting"
            if time.time() - (s.get("dt") or {}).get("at", 0) > 900:
                view["state"] = "failed"
                view["message"] = "The Draw Things add-on didn't answer. Is the fetch-drawthings container running?"
    return view


def notify(d: Path, status: str, message: str) -> None:
    """POST {job, name, status, message, url} to the webhook in Settings, if there is one. Never fails a job."""
    url = read_settings().get("webhook_url", "").strip()
    if not url:
        return
    try:
        s = read_state(d)
        payload = {"job": d.name, "name": s.get("name"), "status": status, "message": message,
                   "url": (s.get("origin") or "").rstrip("/") + f"/#job={d.name}"}
        send_webhook(url, payload)
        update_state(d, webhook="sent")
    except Exception as e:
        try:
            update_state(d, webhook=f"failed: {e}")
        except Exception:
            pass


def send_webhook(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "FetchLoRA"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


# --------------------------------------------------------------------------- #
# Training worker: one job at a time, in queue order
# --------------------------------------------------------------------------- #
def queued_jobs() -> List[tuple]:
    q = [(d, s) for d, s in all_jobs() if s.get("status") == "queued"]
    return sorted(q, key=lambda x: (x[1].get("order", x[1]["created"]), x[1]["created"]))


def next_queued() -> Path | None:
    q = queued_jobs()
    return q[0][0] if q else None


def run_job(d: Path) -> None:
    s = update_state(d, status="running", started=time.time(), error=None)
    log = open(d / "train.log", "ab")

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
            log.close()
            msg = f"Only {n} photos could be used. Please pick at least 5 (20-40 is best) and try again."
            update_state(d, status="failed", finished=time.time(), error=msg)
            notify(d, "failed", msg)
            return
        s = read_state(d)
        if s.get("status") == "cancelled":
            log.close()
            return
    cfg = build_config(d, s)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        ["python", "run.py", str(cfg)], cwd=AITK, stdout=log, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    current.update(job=d.name, proc=proc)
    code = proc.wait()
    log.close()
    current.update(job=None, proc=None)
    s = read_state(d)
    if s.get("status") == "cancelled":
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
            update_state(d, status="failed", error=f"Something went wrong: {e}")
            notify(d, "failed", str(e))


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


# --------------------------------------------------------------------------- #
# Web API
# --------------------------------------------------------------------------- #
app = FastAPI(title="Fetch LoRA")
STATIC = HERE / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# --------------------------------------------------------------------------- #
# PIN gate: every page and API call needs a cookie that proves the PIN was
# entered once on this device. Change the PIN with TRAINER_PIN in .env.
# --------------------------------------------------------------------------- #
PIN = os.environ.get("TRAINER_PIN", "1234").strip()
COOKIE = "dragon_auth"
_secret_file = JOBS / ".auth_secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32))
SECRET = _secret_file.read_text().strip().encode()
TOKEN = hmac.new(SECRET, PIN.encode(), hashlib.sha256).hexdigest()  # changes when the PIN changes
OPEN_PATHS = {"/login", "/api/login", "/static/icon.svg", "/static/icon.png"}
failures: dict = {}


@app.middleware("http")
async def pin_gate(request: Request, call_next):
    if request.url.path in OPEN_PATHS or not PIN:
        return await call_next(request)
    if hmac.compare_digest(request.cookies.get(COOKIE, ""), TOKEN):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "PIN required"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page() -> str:
    return (STATIC / "login.html").read_text()


@app.post("/api/login")
def login(request: Request, pin: str = Form(...)):
    ip = request.client.host if request.client else "?"
    count, until = failures.get(ip, (0, 0))
    if time.time() < until:
        return JSONResponse({"detail": "Too many tries. Wait a minute and try again."}, status_code=429)
    if not hmac.compare_digest(pin.strip(), PIN):
        count += 1
        failures[ip] = (count, time.time() + 60 if count >= 5 else 0)
        time.sleep(1)
        return JSONResponse({"detail": "Wrong PIN"}, status_code=401)
    failures.pop(ip, None)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, TOKEN, max_age=365 * 24 * 3600, httponly=True, samesite="lax")
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


def cleanup_drafts(max_age_h: float = 48) -> None:
    for d in DRAFTS.iterdir():
        try:
            if time.time() - d.stat().st_mtime > max_age_h * 3600:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


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


def gpu_stats() -> dict | None:
    try:
        args = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"]
        if GPU_ID:
            args.insert(1, f"--id={GPU_ID.split(',')[0]}")
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        name, util, used, total, temp, power = [x.strip() for x in r.stdout.strip().split("\n")[0].split(",")]
        return {"name": name, "util": round(float(util)), "mem_used_gb": round(float(used) / 1024, 1),
                "mem_total_gb": round(float(total) / 1024, 1), "temp_c": round(float(temp)),
                "power_w": round(float(power))}
    except Exception:
        return None


@app.get("/api/gpu")
def gpu() -> dict:
    return gpu_stats() or {}




# ------------------------------- photo drafts ------------------------------ #
def draft_dir(draft_id: str) -> Path:
    if not ID_RE.fullmatch(draft_id) or not (DRAFTS / draft_id).is_dir():
        raise HTTPException(404, "Those photos expired. Please pick them again.")
    return DRAFTS / draft_id


def draft_metas(dd: Path) -> List[dict]:
    out = []
    for f in sorted((dd / "meta").glob("*.json")):
        try:
            m = json.loads(f.read_text())
            m["pid"] = int(f.stem)
            out.append(m)
        except Exception:
            continue
    return out


def draft_view(dd: Path, kind: str = "dog") -> dict:
    metas = draft_metas(dd)
    fl = photocheck.flags(metas, kind if kind in KINDS else "dog")
    photos = [{"pid": m["pid"], "name": m.get("name", ""), "tags": m.get("tags", ""), "w": m["w"], "h": m["h"],
               "thumb": f"/api/drafts/{dd.name}/thumb/{m['pid']}", "flags": f}
              for m, f in zip(metas, fl)]
    return {"id": dd.name, "photos": photos, "detector": any(m.get("detector") for m in metas)}


def add_draft_photo(dd: Path, src: Path, name: str, tags: str = "", meta: Optional[dict] = None) -> dict:
    im = open_photo(src, 2560)  # room to crop, still small enough to keep things fast
    with draft_lock:
        pid = len(list((dd / "meta").glob("*.json"))) + len(list((dd / "photos").glob("*.part")))
        while (dd / "photos" / f"{pid:04d}.jpg").exists():
            pid += 1
        (dd / "photos" / f"{pid:04d}.part").write_bytes(b"")
    try:
        im.save(dd / "photos" / f"{pid:04d}.jpg", quality=95)
        th = im.copy()
        th.thumbnail((360, 360))
        th.save(dd / "thumbs" / f"{pid:04d}.jpg", quality=82)
        if meta is None or meta.get("w") != im.width:
            meta = photocheck.analyze(im)
        meta.update(name=name, tags=tags)
        (dd / "meta" / f"{pid:04d}.json").write_text(json.dumps(meta))
    finally:
        (dd / "photos" / f"{pid:04d}.part").unlink(missing_ok=True)
    return {"pid": pid}


def new_draft() -> Path:
    cleanup_drafts()
    did = uuid.uuid4().hex[:12]
    dd = DRAFTS / did
    for sub in ("photos", "thumbs", "meta"):
        (dd / sub).mkdir(parents=True)
    return dd


@app.post("/api/drafts")
def create_draft() -> dict:
    return {"id": new_draft().name}


@app.post("/api/drafts/{draft_id}/photos")
def upload_photo(draft_id: str, photo: UploadFile = File(...), tags: str = Form("")) -> dict:
    dd = draft_dir(draft_id)
    ext = Path(photo.filename or "").suffix.lower() or ".jpg"
    if ext not in IMG_EXT:
        raise HTTPException(400, f"{photo.filename} isn't a photo.")
    tmp = dd / f"upload_{uuid.uuid4().hex}{ext}"
    try:
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(photo.file, fh, length=1024 * 1024)
        try:
            r = add_draft_photo(dd, tmp, photo.filename or "", tags)
        except Exception:
            raise HTTPException(400, f"Couldn't open {photo.filename}. Is it a photo?")
    finally:
        tmp.unlink(missing_ok=True)
    os.utime(dd, None)
    return r


@app.get("/api/drafts/{draft_id}")
def get_draft(draft_id: str, kind: str = "dog") -> dict:
    return draft_view(draft_dir(draft_id), kind)


@app.get("/api/drafts/{draft_id}/thumb/{pid}")
def draft_thumb(draft_id: str, pid: int):
    f = draft_dir(draft_id) / "thumbs" / f"{int(pid):04d}.jpg"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


# ---------------------------------- jobs ----------------------------------- #
@app.get("/api/jobs")
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


@app.post("/api/jobs")
def create_job(request: Request, body: dict = Body(...)) -> dict:
    name = str(body.get("name", "")).strip().lower().replace(" ", "_").replace("-", "_")
    kind, preset = body.get("kind", "dog"), body.get("preset", "good")
    if not NAME_RE.match(name):
        raise HTTPException(400, "Name must start with a letter and use only letters, numbers and _ (3-40 characters), like biscuit_dog")
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
    keep = [int(p) for p in body.get("keep", list(metas)) if int(p) in metas]
    tags = {int(k): str(v) for k, v in (body.get("tags") or {}).items()}
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
    shutil.rmtree(dd, ignore_errors=True)
    queue_event.set()
    return {"id": job_id, "photos": n, "queue": queue_info().get(job_id)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return public(job_dir(job_id))


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str) -> JSONResponse:
    return JSONResponse({"log": log_tail(job_dir(job_id), 80)})


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    d = job_dir(job_id)
    s = read_state(d)
    if s["status"] in ("queued", "running"):
        update_state(d, status="cancelled", finished=time.time())
        proc = current.get("proc")
        if current.get("job") == job_id and proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
def resume(job_id: str) -> dict:
    d = job_dir(job_id)
    s = read_state(d)
    if s["status"] in ("failed", "cancelled", "interrupted"):
        # ai-toolkit picks up from the newest saved checkpoint automatically
        update_state(d, status="queued", error=None, train_started=None, finished=None, order=time.time())
        queue_event.set()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/move")
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


@app.post("/api/jobs/{job_id}/choose")
def choose(job_id: str, step: int = Form(...)) -> dict:
    """Pick which saved version becomes 'the' LoRA (download, Draw Things)."""
    d = job_dir(job_id)
    s = read_state(d)
    if step not in [c["step"] for c in checkpoints(d, s)]:
        raise HTTPException(400, "That version isn't saved.")
    s = update_state(d, chosen_step=step, dt=None)
    if s.get("status") == "done" and s.get("delivered_step") != step:
        info = deliver(d, s)
        update_state(d, result=info, delivered_step=step)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/drawthings")
def send_to_drawthings(job_id: str) -> dict:
    """Convert the chosen version (the add-on does this by itself) and restart the Draw Things server."""
    d = job_dir(job_id)
    s = read_state(d)
    if s.get("status") != "done":
        raise HTTPException(400, "This training isn't finished yet.")
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


@app.post("/api/jobs/{job_id}/again")
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


@app.get("/api/jobs/{job_id}/files/{kind}/{fname}")
def job_file(job_id: str, kind: str, fname: str):
    d = job_dir(job_id)
    s = read_state(d)
    if "/" in fname or ".." in fname:
        raise HTTPException(404)
    if kind == "samples":
        f = d / "output" / s["name"] / "samples" / fname
    elif kind == "images":
        f = d / "images" / fname
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
    s = read_settings()
    s["webhook_url"] = url
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))
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
