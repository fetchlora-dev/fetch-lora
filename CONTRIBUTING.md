# Contributing to Fetch LoRA

Thanks for looking! Fetch LoRA is a free, open-source hobby project. It is never sold, and there
are no support promises, so contributions are what make it better. Beginners are very welcome:
small, careful changes are worth more than big ones.

> Idea by Eric. Built by Eric's AI assistant, GraphicGuru. Free and open source — open to contributors.

## Before you start

1. Read [README.md](README.md), then [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the map of the code.
2. You **don't need a GPU** for most code changes. Much of `app/server.py` and `app/photocheck.py`
   is plain Python you can read, run and test on any computer. You only need an NVIDIA GPU to run a
   real training.
3. Never commit secrets or personal data: no `.env`, no `data/` folder, no photos of real people,
   no IP addresses or hostnames from your own network. `.gitignore` already blocks the usual ones.

## How to send a change

1. Fork the repository and create a branch: `git checkout -b short-description`.
2. Make one focused change. Keep the plain, friendly wording the app uses for its messages.
3. Check that the Python still compiles: `python -m py_compile app/*.py converter/watch.py scripts/*.py`.
   If you added tests, run them (`pytest`).
4. Commit with a clear message ("Add tests for parse_tags") and open a pull request explaining
   **what** you changed and **why**. Screenshots help for anything on the page.

Found a problem but don't want to fix it yourself? Opening an issue that says what you did, what you
expected, and what happened instead is a real contribution too.

## Running the tests

The tests need no GPU, Docker or AI models. From the project folder:

```
python -m venv .venv
.venv/bin/pip install fastapi==0.115.* python-multipart pillow numpy scipy pyyaml httpx pytest
.venv/bin/python -m pytest
```

(On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\python`.) `tests/conftest.py` points the
app at a temporary data folder, so your real trainings are never touched.

## Code style

- Python 3.10+, standard library first, 4-space indents, lines up to about 120 characters.
- Prefer small functions with a docstring that says what they do in plain English.
- Anything optional (photo check, cropping, background removal, webhooks) must **fail soft**: if it
  can't run, the training still goes ahead the plain way.
- Messages shown to users should be short and friendly, with no jargon.

## Good first issues

Small, real improvements that are a good size for a first pull request. None of them needs a GPU.

1. **Add more tests.** `tests/` has a first set (helpers, the web API, the training worker with a
   stand-in for AI Toolkit, and the Draw Things add-on's requests). Pick something not covered yet,
   for example `estimate_seconds()`, `measure_speed()` with a few made-up log lines, or `checkpoints()`
   with empty `.safetensors` files in a temporary folder.

2. **Write unit tests for the photo check rules.** `photocheck.flags()` and `photocheck.crop_box()`
   take plain dictionaries (width, height, detected boxes, blur score, hash), so you can test them with
   made-up data and no real photos or detector. Cover at least: too small, blurry, near-duplicate,
   "can't spot a dog", and a crop that stays inside the photo.

3. **Show a gentle note when the PIN is weak.** The server already prints a warning in its log if
   `TRAINER_PIN` is `1234`. Add a friendly note on the home page too (for example when the PIN is
   1234 or shorter than 6 digits), telling the owner how to change it in `.env`.

4. **Move the test-picture prompts into a settings file.** The three prompts used for the test pictures
   are hard-coded in `sample_prompts()`. Let people change them without editing Python, for example from
   an optional `prompts.json` in the data folder, falling back to today's prompts if the file is missing
   or broken. Document it in the README.

5. **Tidy up the converter folder's docs.** `converter/Dockerfile.build` isn't referenced anywhere
   else, and `converter/watch.py`'s functions have few docstrings or type hints. Find out (by reading the
   Dockerfiles and build.sh) whether `Dockerfile.build` is still needed, then either document how to
   use it in `converter/README.md` or propose removing it, and add docstrings and type hints to
   `watch.py`'s functions.

## License of your contribution

By contributing, you agree that your changes are released under the same license as the files you
change: MIT for the trainer, GPL-3.0 for anything in `converter/`.
