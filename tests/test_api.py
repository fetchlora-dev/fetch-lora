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


def test_upload_limits(client, monkeypatch):
    draft = client.post("/api/drafts").json()["id"]
    monkeypatch.setattr(server, "MAX_PHOTO_MB", 0)  # every real file is "too big" now
    r = client.post(f"/api/drafts/{draft}/photos", files={"photo": ("big.jpg", jpeg_bytes(1), "image/jpeg")})
    assert r.status_code == 413
    assert not list((server.DRAFTS / draft).glob("upload_*"))  # the partial file was cleaned up
    monkeypatch.setattr(server, "MAX_PHOTO_MB", 50)
    monkeypatch.setattr(server, "MAX_PHOTOS", 1)
    ok = client.post(f"/api/drafts/{draft}/photos", files={"photo": ("a.jpg", jpeg_bytes(2), "image/jpeg")})
    assert ok.status_code == 200
    r = client.post(f"/api/drafts/{draft}/photos", files={"photo": ("b.jpg", jpeg_bytes(3), "image/jpeg")})
    assert r.status_code == 400


def test_huge_pixel_count_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "wide.png"
    Image.new("L", (400, 300)).save(f)
    monkeypatch.setattr(server, "MAX_PIXELS", 400 * 300 - 1)
    with pytest.raises(ValueError):
        server.open_photo(f, 1024)


def test_oversized_upload_is_refused_before_it_is_read(client):
    draft = client.post("/api/drafts").json()["id"]
    big = b"x" * ((server.MAX_PHOTO_MB + 2) * 1024 * 1024)
    r = client.post(f"/api/drafts/{draft}/photos", files={"photo": ("huge.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_empty_pin_makes_a_random_one():
    pin = server.choose_pin("")
    assert len(pin) == 6 and pin.isdigit()
    assert server.choose_pin("   ") == pin          # the same one every start, until you set your own
    assert server.choose_pin("8642") == "8642"


def test_bad_job_requests_get_a_clear_answer_not_a_crash(client):
    draft = new_draft_with_photos(client)
    base = {"draft": draft, "name": "odd_input_dog", "preset": "quick"}
    assert client.post("/api/jobs", json={**base, "kind": ["dog"]}).status_code == 400
    assert client.post("/api/jobs", json={**base, "keep": ["one", "two"]}).status_code == 400
    assert client.post("/api/jobs", json={**base, "tags": {"x": "ball"}}).status_code == 400
    # the same photo five times is still only one photo
    assert client.post("/api/jobs", json={**base, "keep": [0, 0, 0, 0, 0]}).status_code == 400


def test_wrong_pins_wait_longer_and_longer(monkeypatch):
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    server.failures.clear()
    c = TestClient(server.app)
    for _ in range(5):
        assert c.post("/api/login", data={"pin": "0000"}).status_code == 401
    assert c.post("/api/login", data={"pin": server.PIN}).status_code == 429  # locked, even with the right PIN
    count, until = server.failures["testclient"]
    assert count == 5 and 50 < until - server.time.time() <= 60
    server.failures["testclient"] = (7, 0)  # pretend the wait is over, after 7 misses
    c.post("/api/login", data={"pin": "0000"})
    assert server.failures["testclient"][1] - server.time.time() > 60 * 2 ** 2  # 8 minutes now
    server.failures.clear()


def test_non_ascii_pin_is_just_wrong():
    server.failures.clear()
    r = TestClient(server.app).post("/api/login", data={"pin": "é1234"})
    assert r.status_code == 401
    server.failures.clear()


def test_saving_the_webhook_keeps_learned_speeds(client):
    server.remember_speed("good", 1.234)
    assert client.post("/api/settings", json={"webhook_url": "http://example.invalid/hook"}).status_code == 200
    s = server.read_settings()
    assert s["webhook_url"] == "http://example.invalid/hook" and s["speed"]["good"] == 1.234


def test_duplicate_flag_ignores_removed_photos(client):
    draft = client.post("/api/drafts").json()["id"]
    same = jpeg_bytes(7)
    for i in range(3):  # photo 0 and 1 are identical, photo 2 is different
        data = same if i < 2 else jpeg_bytes(8)
        client.post(f"/api/drafts/{draft}/photos", files={"photo": (f"p{i}.jpg", data, "image/jpeg")})
    codes = lambda view: {p["pid"]: [f["code"] for f in p["flags"]] for p in view["photos"]}
    if server.photocheck.phash(Image.open(io.BytesIO(same))) is None:
        pytest.skip("scipy isn't installed here, so there are no duplicate checks")
    assert "duplicate" in codes(client.get(f"/api/drafts/{draft}").json())[1]
    # after removing photo 0, photo 1 is no longer anyone's duplicate
    assert "duplicate" not in codes(client.get(f"/api/drafts/{draft}?keep=1,2").json())[1]
