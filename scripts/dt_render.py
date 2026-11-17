"""Render through a Draw Things gRPC server with LoRA metadata attached.

The Draw Things app tells the server what each LoRA is (name, model version) inside the request
("metadata override"). dts-utils doesn't send that yet, so this wrapper adds it and then runs the
normal `dts-utils generate` command. Run it with dts-utils' Python:

  python dt_render.py --lora-spec bo_dog_lora_f16.ckpt:flux2_4b -- generate --host 127.0.0.1 ...
"""
import json
import sys

import dts_utils.generate_api as g

specs = []
args = sys.argv[1:]
while args and args[0] == "--lora-spec":
    file, version = args[1].split(":", 1)
    specs.append({"name": file.replace("_lora_f16.ckpt", ""), "file": file, "prefix": "", "version": version})
    args = args[2:]
if args and args[0] == "--":
    args = args[1:]

_orig = g.prepare_image_generation_request


def _patched(gen):
    req, p, n = _orig(gen)
    if specs:
        req.override.loras = json.dumps(specs).encode()
    return req, p, n


g.prepare_image_generation_request = _patched

from dts_utils.cli_router import main  # noqa: E402

sys.argv = ["dts-utils"] + args
sys.exit(main())
