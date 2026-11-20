"""Tests for the web API, using FastAPI's test client (no GPU, no training is started)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server


@pytest.fixture()
def client():
    # Not used as a context manager, so the startup hook (the training worker) doesn't run.
    c = TestClient(server.app)
    r = c.post("/api/login", data={"pin": server.PIN})
    assert r.status_code == 200
    return c


def jpeg_bytes(seed: int, size=(640, 640)) -> bytes:
    im = Image.effect_noise(size, 40 + seed).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


def new_draft_with_photos(c, n=5) -> str:
    draft = c.post("/api/drafts").json()["id"]
    for i in range(n):
        r = c.post(f"/api/drafts/{draft}/photos", files={"photo": (f"p{i}.jpg", jpeg_bytes(i), "image/jpeg")})
        assert r.status_code == 200, r.text
    return draft


def start(c, name, **extra):
    draft = new_draft_with_photos(c)
    return c.post("/api/jobs", json={"draft": draft, "name": name, "kind": "dog", "preset": "quick", **extra})


def test_pages_need_the_pin():
    c = TestClient(server.app)
    assert c.get("/api/jobs").status_code == 401
    assert c.post("/api/login", data={"pin": "wrong-pin"}).status_code == 401


def test_same_name_asks_before_replacing(client):
    r = start(client, "same_name_dog")
    assert r.status_code == 200, r.text
    first = r.json()["id"]
    # still waiting its turn: a second one with that name is refused
    assert start(client, "same_name_dog").status_code == 400
    d = server.JOBS / first
    server.update_state(d, status="done")
    # finished: the page has to ask first (409), then send replace=true
    assert start(client, "same_name_dog").status_code == 409
    r = start(client, "same_name_dog", replace=True)
    assert r.status_code == 200, r.text
    assert server.read_state(d)["replaced_by"] == r.json()["id"]
    assert client.post(f"/api/jobs/{first}/resume").status_code == 409


def test_home_screen_files_load_before_login():
    # "Add to Home Screen" on the PIN page fetches these before anyone has logged in
    c = TestClient(server.app, follow_redirects=False)
    for path in ("/static/manifest.webmanifest", "/static/icon.png", "/static/icon-512.png"):
        assert c.get(path).status_code == 200, path
