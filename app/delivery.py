"""After training: the saved versions (checkpoints), copying the chosen one to data/loras,
the optional Draw Things add-on, and the "it's done" notification (webhook).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import models as mm
from config import DELIVER_TO, DT_STATUS, EXPORTS
from jobstore import read_settings, read_state, update_state


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
