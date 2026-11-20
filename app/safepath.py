"""One place that turns names from a web request into file paths safely.

Job ids, draft ids and file names arrive in URLs, so a crafted value like "../../etc" must never
reach the disk. Every path built from such a value goes through safe_join(), which normalizes the
result and refuses anything that would land outside the folder it belongs in.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import HTTPException

# A single file name: letters, digits, dot, dash, underscore; no separators, not "." or "..".
NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def safe_join(base: Path, *parts: str, status: int = 404) -> Path:
    """base/parts..., but only if the result stays inside base; otherwise an HTTP error."""
    root = os.path.normpath(os.path.abspath(str(base)))
    full = os.path.normpath(os.path.join(root, *[str(p) for p in parts]))
    if not full.startswith(root + os.sep):
        raise HTTPException(status)
    return Path(full)


def safe_name(value: str, status: int = 404) -> str:
    """A plain file name (no folders, no '..', no empty string), or an HTTP error."""
    if not isinstance(value, str) or not NAME_RE.fullmatch(value) or value in (".", ".."):
        raise HTTPException(status)
    return value
