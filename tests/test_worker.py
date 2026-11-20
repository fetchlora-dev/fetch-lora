"""Tests for the training worker, with a stand-in for ai-toolkit (a tiny run.py), so no GPU is needed."""
import os
import sys
import threading
import time

import server
from test_api import client, start  # noqa: F401  (pytest fixture and helper)


def fake_toolkit(tmp_path, monkeypatch, seconds=5, exit_code=1):
    # Like ai-toolkit, it takes a moment to shut down after Cancel (SIGTERM).
    (tmp_path / "run.py").write_text(
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: (time.sleep(1), sys.exit(1)))\n"
        f"time.sleep({seconds})\nsys.exit({exit_code})\n")
    monkeypatch.setattr(server, "AITK", tmp_path)
    # run_job starts `python run.py`; make sure `python` is this test's Python
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])


def wait_for(cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_cancelled_job_is_not_started(client, tmp_path, monkeypatch):
    fake_toolkit(tmp_path, monkeypatch, seconds=0, exit_code=0)
    job = start(client, "cancel_first_dog").json()["id"]
    d = server.JOBS / job
    client.post(f"/api/jobs/{job}/cancel")
    server.run_job(d)
    assert server.read_state(d)["status"] == "cancelled"
    assert not (d / "train.log").exists()


def test_cancel_then_resume_while_stopping_stays_queued(client, tmp_path, monkeypatch):
    fake_toolkit(tmp_path, monkeypatch, seconds=30)
    job = start(client, "resume_me_dog").json()["id"]
    d = server.JOBS / job
    t = threading.Thread(target=server.run_job, args=(d,))
    t.start()
    assert wait_for(lambda: server.current.get("job") == job)
    client.post(f"/api/jobs/{job}/cancel")
    client.post(f"/api/jobs/{job}/resume")  # pressed before the old run has fully stopped
    t.join(timeout=20)
    assert not t.is_alive()
    # it used to be marked "failed" here when the cancelled run exited
    assert server.read_state(d)["status"] == "queued"
    server.update_state(d, status="cancelled")  # tidy up for other tests


def test_failed_run_says_so(client, tmp_path, monkeypatch):
    fake_toolkit(tmp_path, monkeypatch, seconds=0, exit_code=1)
    job = start(client, "fails_dog").json()["id"]
    d = server.JOBS / job
    server.run_job(d)
    s = server.read_state(d)
    assert s["status"] == "failed" and s["finished"]
