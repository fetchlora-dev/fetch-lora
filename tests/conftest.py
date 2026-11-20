"""Test setup: point Fetch LoRA at a throwaway data folder before server.py is imported.

server.py creates its folders when it's imported, so the environment variables have to be set first.
No GPU, Docker or AI models are needed for these tests.
"""
import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="fetchlora-test-"))
os.environ["DRAGON_DATA"] = str(_tmp / "data")
os.environ["DRAGON_EXPORTS"] = str(_tmp / "exports")
os.environ.setdefault("TRAINER_PIN", "246810")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "converter"))
