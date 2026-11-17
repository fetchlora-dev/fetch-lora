"""Photo check and auto-crop for Fetch LoRA.

Everything here runs on the CPU and fails soft: if a check can't run, the photo is simply
not flagged (and not cropped), and training continues the old way.

- Blur: variance of the Laplacian (OpenCV), measured on the subject when one was found.
- Near-duplicates: a 64-bit perceptual hash (DCT of a 32x32 grey thumbnail).
- Subject: torchvision's Faster R-CNN MobileNetV3 detector, COCO classes (BSD-3-Clause),
  used to flag "no dog found" / "several dogs" and to crop around the subject.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
from PIL import Image

BLUR_MIN = 60.0        # Laplacian variance below this looks blurry (tuned on real pet photos)
SMALL_WARN = 512       # shortest side below this: low resolution warning
SMALL_MIN = 256        # shortest side below this: unusable
DUP_MAX_BITS = 6       # perceptual hashes this close are near-duplicates
DET_SCORE = 0.5

ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
# The COCO detector often calls a black curly dog a "bear", "teddy bear" or even "cat" (seen on real
# photos), so for pets any animal counts as the subject.
PETS = ANIMALS | {"teddy bear"}
TARGETS = {"dog": PETS, "cat": PETS, "pet": PETS, "person": {"person"}, "thing": None, "style": None}
NOUN = {"dog": "dog", "cat": "cat", "pet": "animal", "person": "person"}

_model = None
_categories: List[str] = []
_model_lock = threading.Lock()
_model_failed = False


def _detector():
    global _model, _categories, _model_failed
    if _model is not None or _model_failed:
        return _model
    with _model_lock:
        if _model is None and not _model_failed:
            try:
                import torch
                from torchvision.models.detection import (
                    FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn)
                torch.set_num_threads(4)
                w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=0.3)
                m.eval()
                _categories = list(w.meta["categories"])
                _model = m
            except Exception as e:  # no detector: no subject flags, no crop
                print("photocheck: detector unavailable:", e, flush=True)
                _model_failed = True
    return _model


def detect(im: Image.Image) -> List[dict]:
    """Objects in the photo: [{'label': 'dog', 'score': 0.97, 'box': [x0, y0, x1, y1]}] in image pixels."""
    m = _detector()
    if m is None:
        return []
    try:
        import torch
        from torchvision.transforms.functional import to_tensor
        scale = 640 / max(im.size)
        small = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale)))) if scale < 1 else im
        s = scale if scale < 1 else 1.0
        with _model_lock, torch.inference_mode():
            out = m([to_tensor(small.convert("RGB"))])[0]
        dets = []
        for box, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
            if score >= 0.3:
                dets.append({"label": _categories[label], "score": round(score, 3),
                             "box": [round(v / s) for v in box]})
        return dets
    except Exception as e:
        print("photocheck: detect failed:", e, flush=True)
        return []


def blur_score(im: Image.Image, box: Optional[List[int]] = None) -> Optional[float]:
    try:
        import cv2
        g = im.convert("L")
        if box:
            x0, y0, x1, y1 = box
            if (x1 - x0) > 64 and (y1 - y0) > 64:
                g = g.crop((x0, y0, x1, y1))
        scale = 1024 / max(g.size)
        if scale < 1:
            g = g.resize((max(1, round(g.width * scale)), max(1, round(g.height * scale))))
        return round(float(cv2.Laplacian(np.asarray(g, dtype=np.uint8), cv2.CV_64F).var()), 1)
    except Exception:
        return None


def phash(im: Image.Image) -> Optional[str]:
    try:
        from scipy.fft import dct
        a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
        d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()
        bits = d[1:] > np.median(d[1:])
        return "%016x" % int("".join("1" if b else "0" for b in np.concatenate([[False], bits])), 2)
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def analyze(im: Image.Image) -> dict:
    """Everything we need to know about one photo, computed once at upload."""
    dets = detect(im)
    for d in dets:
        if d["score"] >= DET_SCORE:
            d["blur"] = blur_score(im, d["box"])
    return {"w": im.width, "h": im.height, "dets": dets, "phash": phash(im),
            "blur_full": blur_score(im), "detector": _model is not None}


def subject(meta: dict, kind: str) -> Optional[dict]:
    targets = TARGETS.get(kind)
    if targets is None:
        return None
    found = [d for d in meta.get("dets", []) if d["label"] in targets and d["score"] >= DET_SCORE]
    if not found:
        return None
    area = lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
    return max(found, key=area)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def distinct(dets: List[dict]) -> List[dict]:
    """One entry per thing: the same dog labelled 'dog' and 'bear' counts once."""
    out: List[dict] = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        if all(_iou(d["box"], o["box"]) < 0.5 for o in out):
            out.append(d)
    return out


def flags(metas: List[dict], kind: str, detector_ok: bool = True) -> List[List[dict]]:
    """Friendly flags per photo (same order as metas). Each flag: {'code', 'text', 'serious'}."""
    out: List[List[dict]] = []
    seen: List[tuple] = []  # (index, phash)
    targets = TARGETS.get(kind)
    noun = NOUN.get(kind, "subject")
    for i, m in enumerate(metas):
        f: List[dict] = []
        short = min(m.get("w", 0), m.get("h", 0))
        if short < SMALL_MIN:
            f.append({"code": "too_small", "text": "Too small to use", "serious": True})
        elif short < SMALL_WARN:
            f.append({"code": "small", "text": "Low resolution", "serious": True})
        subj = subject(m, kind)  # sharp dog + blurry background is fine, so judge the dog
        blur = subj.get("blur") if subj else None
        if blur is None:
            blur = m.get("blur_full")
        if blur is not None and blur < BLUR_MIN:
            f.append({"code": "blurry", "text": "Looks blurry", "serious": True})
        if targets is not None and detector_ok and m.get("detector"):
            found = [d for d in m.get("dets", []) if d["label"] in targets and d["score"] >= DET_SCORE]
            if not found:
                f.append({"code": "no_subject", "text": f"Can't spot a {noun}", "serious": True})
            else:
                big = max((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) for d in found)
                others = [d for d in distinct(found) if d["score"] >= 0.7 and
                          (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) >= 0.3 * big]
                if len(others) >= 2:
                    f.append({"code": "several", "text": f"More than one {noun}", "serious": False})
        ph = m.get("phash")
        if ph:
            for j, other in seen:
                if hamming(ph, other) <= DUP_MAX_BITS:
                    f.append({"code": "duplicate", "text": f"Almost the same as photo {j + 1}", "serious": True})
                    break
            seen.append((i, ph))
        out.append(f)
    return out


def crop_box(meta: dict, kind: str, pad: float = 0.25, min_side: int = 640) -> Optional[List[int]]:
    """A crop around the main subject with some room around it, or None to keep the whole photo."""
    subj = subject(meta, kind)
    if subj is None:
        return None
    W, H = meta["w"], meta["h"]
    x0, y0, x1, y1 = subj["box"]
    bw, bh = x1 - x0, y1 - y0
    p = pad * max(bw, bh)
    x0, y0, x1, y1 = x0 - p, y0 - p, x1 + p, y1 + p
    # grow small crops toward min_side (when the photo is big enough)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, min(min_side, W)), max(y1 - y0, min(min_side, H))
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    # shift inside the photo, then clamp
    if x0 < 0: x1, x0 = x1 - x0, 0
    if y0 < 0: y1, y0 = y1 - y0, 0
    if x1 > W: x0, x1 = x0 - (x1 - W), W
    if y1 > H: y0, y1 = y0 - (y1 - H), H
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if (x1 - x0) * (y1 - y0) > 0.85 * W * H:  # subject already fills the photo
        return None
    if min(x1 - x0, y1 - y0) < SMALL_MIN:
        return None
    return [x0, y0, x1, y1]
