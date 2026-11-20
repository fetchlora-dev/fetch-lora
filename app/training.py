"""Planning a training: how many steps, how long it will take, and the recipe (config.yaml)
handed to AI Toolkit. Also asks nvidia-smi about the GPU.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import yaml

import models as mm
from config import GPU_ID, KINDS, lock
from jobstore import read_settings, write_settings


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
            write_settings(st)
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
                    "network": mm.network_block(model),
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
