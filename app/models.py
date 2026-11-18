"""The model matrix: which base model a LoRA is trained on.

    Quick test  -> FLUX.2 Klein 4B only (smallest, fastest).
    Good        -> Klein 4B or Z-Image (6B).
    Best        -> Klein 4B, Z-Image or Qwen-Image (20B, needs a 24 GB+ GPU to train).

Everything here is plain data plus small helpers; server.py uses it to plan steps,
estimate time, build the ai-toolkit config and decide what Draw Things can do.
All three models are Apache 2.0 and ungated on Hugging Face (checked Nov 2026).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

DEFAULT_MODEL = "klein4b"

# Steps grow with the number of photos. Best is quality first; if the last version over-learns,
# the version picker can fall back to an earlier one. s_per_step is only a first guess for an
# RTX 5070 Ti-class card: once this PC has trained a model+preset, its measured speed is used.
MODELS: Dict[str, dict] = {
    "klein4b": {
        "label": "FLUX.2 Klein 4B",
        "short": "Klein 4B",
        "rank": 1,  # realism: higher is more realistic (used to pre-pick the best allowed model)
        "repo": os.environ.get("DRAGON_BASE_MODEL", "black-forest-labs/FLUX.2-klein-base-4B"),
        "arch": os.environ.get("DRAGON_ARCH", "flux2_klein_4b"),
        "params": "4B",
        "license": "Apache 2.0",
        "maker": "Black Forest Labs (Germany)",
        "download_gb": 16,
        "min_vram_gb": 12,
        "presets": {
            "quick": {"steps": 300, "per_photo": 0, "min": 300, "max": 300, "res": [512], "s_per_step": 0.94},
            "good": {"steps": 1500, "per_photo": 25, "min": 800, "max": 2000, "res": [512, 768, 1024], "s_per_step": 1.9},
            "best": {"steps": 3000, "per_photo": 60, "min": 2000, "max": 6000, "res": [512, 768, 1024], "s_per_step": 2.0},
        },
        "load_s": 150, "sample_round_s": 80,
        "sample": {"guidance_scale": 4, "sample_steps": 30},
        "blurb": "Fastest and lightest. Good likeness; can wobble on unusual scenes (like the astronaut).",
        "dt_model": "FLUX.2 [klein] 4B",
        "dt_send": True,   # the PC's Draw Things server has this base model, so one-tap send works
        "dt_note": "Works in Draw Things on iPhone 15 Pro or later and any M1-or-later iPad; "
                   "older devices can use Draw Things' cloud offload.",
    },
    "zimage": {
        "label": "Z-Image 6B",
        "short": "Z-Image",
        "rank": 2,
        "repo": os.environ.get("DRAGON_ZIMAGE_MODEL", "Tongyi-MAI/Z-Image"),  # Base, not Turbo
        "arch": "zimage",
        "params": "6B",
        "license": "Apache 2.0",
        "maker": "Alibaba Tongyi Lab (China)",
        "download_gb": 21,
        "min_vram_gb": 15,  # a "16 GB" card reports ~15.9 GB
        "presets": {
            # Z-Image Base learns a bit slower per step than Klein and has more to learn (6B).
            # Good: 60 per photo, 1500-2500.  Best: 120 per photo, 3600-4800 (Eric's ~4800 ceiling).
            "good": {"steps": 2000, "per_photo": 60, "min": 1500, "max": 2500, "res": [512, 768, 1024], "s_per_step": 2.9},
            "best": {"steps": 4200, "per_photo": 120, "min": 3600, "max": 4800, "res": [512, 768, 1024], "s_per_step": 3.0},
        },
        "load_s": 210, "sample_round_s": 110,
        # ai-toolkit subtracts 1 from guidance for Z-Image, so 5 here = CFG 4 (model card: 3-5, 28-50 steps)
        "sample": {"guidance_scale": 5, "sample_steps": 30},
        "blurb": "Bigger and about 1.5x slower. Usually better at putting your pet into new scenes and styles.",
        "dt_model": "Z Image Base 1.0",
        "dt_send": False,  # the PC's Draw Things server only has Klein 4B; import the file instead
        "dt_note": "Works in Draw Things on iPhone 15 Pro or later and any M1-or-later iPad; "
                   "older devices can use Draw Things' cloud offload.",
    },
    "qwen": {
        "label": "Qwen-Image 20B",
        "short": "Qwen-Image",
        "rank": 3,
        "repo": os.environ.get("DRAGON_QWEN_MODEL", "Qwen/Qwen-Image"),
        "arch": "qwen_image",
        "params": "20B",
        "license": "Apache 2.0",
        "maker": "Alibaba Qwen team (China)",
        "download_gb": 58,
        "min_vram_gb": 23,  # a "24 GB" card reports ~23.6 GB; 3-bit + accuracy recovery adapter
        "presets": {
            "best": {"steps": 3000, "per_photo": 60, "min": 2000, "max": 4000, "res": [512, 768, 1024], "s_per_step": 4.0},
        },
        "load_s": 420, "sample_round_s": 150,
        "sample": {"guidance_scale": 3, "sample_steps": 25},  # ai-toolkit's own Qwen-Image example
        "blurb": "The most realistic and by far the slowest. Needs a big GPU to train (24 GB+, 48 GB is comfortable).",
        "dt_model": "Qwen Image 1.0",
        "dt_send": False,
        "dt_note": "Needs about 11 GB in Draw Things: a Mac or a 16 GB iPad Pro (1 TB/2 TB models); not iPhone.",
    },
}

PRESET_LABELS = {"quick": "Quick test", "good": "Good", "best": "Best"}

# Qwen-Image on 24-39 GB: 3-bit weights + Ostris' accuracy recovery adapter (ai-toolkit's own 24 GB recipe).
QWEN_ARA = "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors"
QWEN_FULL_VRAM_GB = 40


def model_of(s: dict) -> str:
    """The model a job uses (jobs from before the matrix are all Klein 4B)."""
    m = s.get("model") or DEFAULT_MODEL
    return m if m in MODELS else DEFAULT_MODEL


def preset_cfg(model: str, preset: str) -> dict:
    return MODELS[model]["presets"][preset]


def speed_key(model: str, preset: str) -> str:
    # Klein keeps the old keys so speeds this PC already learned still count.
    return preset if model == DEFAULT_MODEL else f"{model}:{preset}"


def vram_gb(total_gb: Optional[float]) -> Optional[float]:
    """Training GPU memory, or DRAGON_VRAM_GB if set (testing, or when nvidia-smi isn't visible)."""
    try:
        o = float(os.environ.get("DRAGON_VRAM_GB", "") or 0)
        if o > 0:
            return o
    except ValueError:
        pass
    return total_gb


def availability(model: str, preset: str, gpu: Optional[dict]) -> dict:
    """Can THIS PC train this model at this level? (Phone/iPad checks happen in the page.)"""
    m = MODELS[model]
    if preset not in m["presets"]:
        return {"ok": False, "reason": f"Not offered for {PRESET_LABELS[preset]}."}
    total = vram_gb((gpu or {}).get("mem_total_gb"))
    name = (gpu or {}).get("name") or "your GPU"
    name = name.replace("NVIDIA GeForce ", "")
    if total is None:  # can't tell: allow it, the run itself will say if memory runs out
        return {"ok": True, "vram_gb": None}
    if total < m["min_vram_gb"]:
        need = int(round(m["min_vram_gb"] + 0.5))
        return {"ok": False, "vram_gb": total,
                "reason": f"Needs a GPU with about {need} GB to train. {name} has {total:.0f} GB."}
    return {"ok": True, "vram_gb": total}


def catalog(gpu: Optional[dict]) -> dict:
    """What the page needs to draw the model picker."""
    out = {}
    for k, m in MODELS.items():
        out[k] = {key: m[key] for key in ("label", "short", "rank", "params", "license", "maker", "download_gb",
                                          "min_vram_gb", "blurb", "dt_model", "dt_send", "dt_note")}
        out[k]["presets"] = list(m["presets"])
        out[k]["available"] = {p: availability(k, p, gpu) for p in m["presets"]}
    return out


def model_block(model: str, gpu_total_gb: Optional[float], low_vram_env: bool) -> dict:
    """The ai-toolkit `model:` section for one base model."""
    m = MODELS[model]
    total = vram_gb(gpu_total_gb)
    block = {"name_or_path": m["repo"], "arch": m["arch"], "quantize": True, "qtype": "qfloat8",
             "quantize_te": True, "qtype_te": "qfloat8", "low_vram": low_vram_env}
    if model == "klein4b":
        block["model_kwargs"] = {"match_target_res": False}
    elif model == "zimage":
        # quantize on the CPU first so loading never spikes past 16 GB
        block["low_vram"] = low_vram_env or (total is not None and total < 20)
    elif model == "qwen":
        if total is None or total < QWEN_FULL_VRAM_GB:
            block["qtype"] = QWEN_ARA
            block["low_vram"] = True
    return block


def train_overrides(model: str, gpu_total_gb: Optional[float]) -> dict:
    """Per-model changes to the `train:` section."""
    total = vram_gb(gpu_total_gb)
    if model == "qwen" and (total is None or total < QWEN_FULL_VRAM_GB):
        # ai-toolkit: "caching text embeddings is required for 24GB". Our captions already
        # contain the trigger word, so the trigger still works with cached embeddings.
        return {"cache_text_embeddings": True}
    return {}


def resolutions(model: str, preset: str, gpu_total_gb: Optional[float]) -> List[int]:
    res = list(preset_cfg(model, preset)["res"])
    total = vram_gb(gpu_total_gb)
    if model == "qwen" and (total is None or total < QWEN_FULL_VRAM_GB):
        res = [r for r in res if r <= 768]  # 1024 px is tight on a 24 GB card
    return res
