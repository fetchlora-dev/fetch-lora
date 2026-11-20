"""The PIN lock: every page and API call needs a cookie that proves the PIN was entered once
on this device. Change the PIN with TRAINER_PIN in .env.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from config import JOBS, STATIC


router = APIRouter()  # server.py adds these pages to the app


def choose_pin(from_env: str) -> str:
    """The PIN for the page: TRAINER_PIN from .env, or, if that's empty, a random 6-digit PIN
    made once, kept in data/jobs/.pin and shown in the log (`docker logs dragon-trainer`).

    There used to be a built-in 1234 (and an empty PIN turned the lock off), which anyone on
    the network could guess.
    """
    pin = from_env.strip()
    if pin:
        if pin == "1234":
            print("WARNING: TRAINER_PIN is 1234, which anyone can guess. Pick your own in .env.", flush=True)
        return pin
    pin_file = JOBS / ".pin"
    if not pin_file.exists():
        pin_file.write_text(f"{secrets.randbelow(1_000_000):06d}")
    pin = pin_file.read_text().strip()
    print(f"No TRAINER_PIN in .env, so Fetch LoRA made one: {pin}  (set TRAINER_PIN in .env to choose your own)",
          flush=True)
    return pin


PIN = choose_pin(os.environ.get("TRAINER_PIN", ""))
COOKIE = "dragon_auth"
_secret_file = JOBS / ".auth_secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32))
SECRET = _secret_file.read_text().strip().encode()
TOKEN = hmac.new(SECRET, PIN.encode(), hashlib.sha256).hexdigest()  # changes when the PIN changes
# Pages and files anyone may load without the PIN. The manifest and icons are here because
# "Add to Home Screen" fetches them from the PIN page, before the device has the cookie.
OPEN_PATHS = {"/login", "/api/login", "/static/icon.svg", "/static/icon.png", "/static/icon-512.png",
              "/static/manifest.webmanifest"}
failures: dict = {}


async def pin_gate(request: Request, call_next):
    """Runs before every request (server.py installs it): no PIN cookie, no entry."""
    if request.url.path in OPEN_PATHS:
        return await call_next(request)
    if hmac.compare_digest(request.cookies.get(COOKIE, ""), TOKEN):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "PIN required"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page() -> str:
    return (STATIC / "login.html").read_text()


@router.post("/api/login")
def login(request: Request, pin: str = Form(...)):
    ip = request.client.host if request.client else "?"
    count, until = failures.get(ip, (0, 0))
    if time.time() < until:
        mins = max(1, round((until - time.time()) / 60))
        return JSONResponse({"detail": f"Too many tries. Wait {mins} minute{'s' if mins > 1 else ''} and try again."},
                            status_code=429)
    # Compare as bytes: comparing str with non-ASCII characters (like "é") raises an error.
    if not hmac.compare_digest(pin.strip().encode(), PIN.encode()):
        count += 1
        # After 5 wrong PINs, wait 1 minute, then 2, 4, 8 ... up to an hour per try, so a 4-digit
        # PIN can't be guessed by trying them all. (It used to stay at 1 minute: ~1 day for all 10,000.)
        wait = min(3600, 60 * 2 ** (count - 5)) if count >= 5 else 0
        if len(failures) > 10000:  # don't let a flood of addresses fill the memory
            failures.clear()
        failures[ip] = (count, time.time() + wait)
        time.sleep(1)
        return JSONResponse({"detail": "Wrong PIN"}, status_code=401)
    failures.pop(ip, None)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, TOKEN, max_age=365 * 24 * 3600, httponly=True, samesite="lax")
    return resp


@router.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp
