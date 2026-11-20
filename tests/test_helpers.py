"""Tests for the small helper functions (app/training.py and app/photos.py)."""
import photos
import training


def test_plan_steps_grows_with_photos_and_stays_in_range():
    assert training.plan_steps("quick", 30) == 300            # Quick test is a fixed length
    assert training.plan_steps("good", 1) == 800              # never below the minimum
    assert training.plan_steps("good", 1000) == 2000          # never above the maximum
    assert training.plan_steps("good", 40) == 40 * 25


def test_plan_steps_override_is_clamped():
    assert training.plan_steps("good", 20, override=10) == 50
    assert training.plan_steps("good", 20, override=99999) == 8000


def test_save_every_gives_a_few_versions():
    assert training.save_every(300) == 100
    assert training.save_every(1500) == 300


def test_parse_tags_cleans_up():
    assert photos.parse_tags("Frisbee, mid-air , frisbee,, Left  side", "biscuit") == ["frisbee", "mid-air", "left side"]
    assert photos.parse_tags("biscuit, ball", "biscuit") == ["ball"]   # the trigger word is never a tag


def test_merge_caption_skips_words_already_there():
    cap = photos.merge_caption("biscuit", ["dog", "frisbee"], "a photo of biscuit, a dog", "a photo of a dog")
    assert cap == "biscuit, frisbee, a photo of a dog"
    assert photos.merge_caption("biscuit", [], "a photo of biscuit, a dog") == "a photo of biscuit, a dog"
