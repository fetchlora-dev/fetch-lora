"""Photos: opening them safely, tags and captions, getting them ready for training, and
"drafts" (photos someone picked on the page but hasn't pressed Start for yet).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from PIL import Image, ImageOps

import photocheck
from config import DRAFTS, HERE, ID_RE, KINDS, MAX_PIXELS, draft_lock
from jobstore import update_state


try:  # iPhone/iPad photos from the Files app may still be HEIC
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass


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
        # Check the size before decoding: a tiny file can claim to be a gigantic picture.
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"{im.width}x{im.height} is too many pixels")
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


# ------------------------------- photo drafts ------------------------------ #
def cleanup_drafts(max_age_h: float = 48) -> None:
    for d in DRAFTS.iterdir():
        try:
            if time.time() - d.stat().st_mtime > max_age_h * 3600:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


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


def draft_view(dd: Path, kind: str = "dog", keep: Optional[List[int]] = None) -> dict:
    """Photos and their flags. With `keep`, only the photos still chosen are checked, so a photo
    isn't called a duplicate of one that was already removed, and "photo 3" is the third tile."""
    metas = draft_metas(dd)
    if keep is not None:
        metas = [m for m in metas if m["pid"] in keep]
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
