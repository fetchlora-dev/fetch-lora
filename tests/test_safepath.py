"""Names from a URL must never reach a file outside the folder they belong in."""
from pathlib import Path

import pytest
from fastapi import HTTPException

from safepath import safe_join, safe_name

BAD = ["", ".", "..", "../x", "../../etc/passwd", "/etc/passwd", "a/b", "a\\b", "..%2Fx", "%2e%2e", "x\x00y"]


def test_safe_join_keeps_normal_names(tmp_path):
    assert safe_join(tmp_path, "abc123def456") == tmp_path / "abc123def456"
    assert safe_join(tmp_path, "images", "001.jpg") == tmp_path / "images" / "001.jpg"


@pytest.mark.parametrize("part", ["..", "../x", "../../etc/passwd", "/etc/passwd", "a/../../x", "", "."])
def test_safe_join_refuses_escapes(tmp_path, part):
    with pytest.raises(HTTPException):
        safe_join(tmp_path / "base", part)


def test_safe_name_accepts_real_file_names():
    for n in ["001.jpg", "1700000000000__000000250_0.jpg", "my_dog.safetensors"]:
        assert safe_name(n) == n


@pytest.mark.parametrize("name", BAD)
def test_safe_name_refuses_tricks(name):
    with pytest.raises(HTTPException):
        safe_name(name)
