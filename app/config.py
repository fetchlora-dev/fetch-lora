"""Settings, folders and shared constants for Fetch LoRA.

Every setting can be changed with an environment variable (see docker-compose.yml and .env).
The locks and the `current` dict are shared by the web server and the training worker.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import models as mm


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
# Upload limits, so one bad (or huge) upload can't fill the PC's disk or memory.
MAX_PHOTO_MB = int(os.environ.get("DRAGON_MAX_PHOTO_MB", "50"))      # per photo file
MAX_PHOTOS = int(os.environ.get("DRAGON_MAX_PHOTOS", "300"))         # per set of photos
MAX_PIXELS = 100_000_000                                             # 100 megapixels per photo

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


STATIC = HERE / "static"  # the web page, login page and icons
