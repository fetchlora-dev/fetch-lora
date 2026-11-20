# SPDX-License-Identifier: GPL-3.0-only  (tests the GPL-3.0 converter/ add-on, so it shares its license)
"""Tests for the Draw Things add-on's request handling (converter/watch.py). No Docker needed."""
import json

import watch


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "IN", tmp_path / "in")
    monkeypatch.setattr(watch, "OUT", tmp_path / "out")
    monkeypatch.setattr(watch, "STATUS", tmp_path / "in" / ".drawthings")
    watch.STATUS.mkdir(parents=True)
    watch.OUT.mkdir()


def test_request_with_a_sneaky_name_is_dropped(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    req = watch.STATUS / "restart-request-abc.json"
    req.write_text(json.dumps({"id": "abc", "name": "../../etc/passwd", "at": 0}))
    watch.handle_requests()
    assert not req.exists()
    assert not list(watch.STATUS.glob("restart-result-*"))


def test_broken_status_file_does_not_block_requests(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    (watch.STATUS / "biscuit_dog.json").write_text("{not json")
    req = watch.STATUS / "restart-request-abc.json"
    req.write_text(json.dumps({"id": "abc", "name": "biscuit_dog", "at": 0}))  # very old: times out
    watch.handle_requests()
    result = json.loads((watch.STATUS / "restart-result-abc.json").read_text())
    assert result["ok"] is False and not req.exists()
