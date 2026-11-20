"""Fetch LoRA → Draw Things: convert every new LoRA into Draw Things' own format.

Watches IN (the trainer's export folder) for *.safetensors. For each one without an up-to-date
<name>_lora_f16.ckpt in OUT (the Draw Things gRPC server's models folder), it runs Draw Things'
LoRAConverter, copies the result in, and records the LoRA in OUT/custom_lora.json so the server
can tell the iPad what it is. A small status file goes to IN/.drawthings/<name>.json for the
trainer page.

"Send to Draw Things": when the trainer page asks (IN/.drawthings/restart-request-<id>.json),
and the LoRA is converted, this restarts the Draw Things server container so it lists the new
LoRA. That needs the Docker socket mounted (see docker-compose.drawthings.yml); it only ever
calls "restart" on the one container named in DT_SERVER_CONTAINER. DT_RESTART_DRY_RUN=1 checks
everything but doesn't restart.
"""
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

IN = Path(os.environ.get("DT_IN", "/exports"))
OUT = Path(os.environ.get("DT_OUT", "/grpc-models"))
VERSION = os.environ.get("DT_VERSION", "flux2_4b")
INTERVAL = float(os.environ.get("DT_INTERVAL", "10"))
CONVERTER = os.environ.get("DT_CONVERTER", "/opt/dt/LoRAConverter")
STATUS = IN / ".drawthings"
DOCKER_SOCK = os.environ.get("DT_DOCKER_SOCK", "/var/run/docker.sock")
SERVER = os.environ.get("DT_SERVER_CONTAINER", "drawthings-grpc")
DRY_RUN = os.environ.get("DT_RESTART_DRY_RUN", "0") == "1"
REQUEST_TIMEOUT = 900
NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,39}")  # the same names the trainer allows


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def merge_spec(spec):
    path = OUT / "custom_lora.json"
    try:
        specs = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        specs = []
    specs = [s for s in specs if s.get("file") != spec["file"]] + [spec]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(specs, indent=1))
    tmp.replace(path)


def convert(src: Path):
    name = src.stem
    target = OUT / f"{name}_lora_f16.ckpt"
    status = STATUS / f"{name}.json"
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([CONVERTER, "-f", str(src), "-n", name, "--version", VERSION, "-o", tmp],
                           capture_output=True, text=True, timeout=1800)
        made = Path(tmp) / target.name
        if r.returncode != 0 or not made.exists():
            err = (r.stderr or r.stdout)[-500:]
            status.write_text(json.dumps({"ok": False, "source_mtime": src.stat().st_mtime,
                                          "error": err, "at": time.time()}))
            log(f"FAILED {src.name}: {err}")
            return
        out = r.stdout
        spec = json.loads(out[out.index("{"):out.rindex("}") + 1])  # printed as pretty JSON
        part = OUT / (target.name + ".part")
        shutil.copyfile(made, part)
        part.replace(target)
    spec.setdefault("prefix", "")
    merge_spec(spec)
    status.write_text(json.dumps({"ok": True, "file": target.name, "spec": spec,
                                  "source_mtime": src.stat().st_mtime, "at": time.time()}))
    log(f"converted {src.name} -> {target.name} ({target.stat().st_size / 1e6:.0f} MB)")


# ------------------------- Docker (restart only) ---------------------------- #
class _UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=60)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(60)
        self.sock.connect(self._path)


def docker(method, path):
    c = _UnixHTTP(DOCKER_SOCK)
    try:
        c.request(method, path)
        r = c.getresponse()
        body = r.read()
        return r.status, body
    finally:
        c.close()


def can_restart():
    if not os.path.exists(DOCKER_SOCK):
        return False
    try:
        code, _ = docker("GET", f"/containers/{SERVER}/json")
        return code == 200
    except Exception:
        return False


def restart_server(dry_run):
    code, body = docker("GET", f"/containers/{SERVER}/json")
    if code != 200:
        return False, f"Couldn't find the Draw Things server ({SERVER})."
    if dry_run:
        state = json.loads(body).get("State", {}).get("Status", "?")
        return True, f"Test mode: the Draw Things server is {state}; it was NOT restarted."
    code, body = docker("POST", f"/containers/{SERVER}/restart?t=10")
    if code not in (200, 204):
        return False, f"Docker said no ({code}): {body[:200].decode('utf-8', 'replace')}"
    for _ in range(30):  # wait until it's running again
        time.sleep(1)
        code, body = docker("GET", f"/containers/{SERVER}/json")
        if code == 200 and json.loads(body).get("State", {}).get("Running"):
            break
    time.sleep(8)  # give the server a moment to load its model list
    return True, "Done! Open Draw Things and pick your LoRA."


def read_json(path: Path) -> dict:
    """A small JSON status file, or {} if it's missing or broken."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def handle_requests():
    for req_file in sorted(STATUS.glob("restart-request-*.json")):
        req = read_json(req_file)
        rid, name = str(req.get("id", "")), str(req.get("name", ""))
        # Only plain LoRA names and ids: never let a request point at files outside these folders.
        if not NAME_RE.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{1,32}", rid):
            req_file.unlink(missing_ok=True)
            continue
        result = STATUS / f"restart-result-{rid}.json"
        src = IN / f"{name}.safetensors"
        status_file = STATUS / f"{name}.json"
        done = read_json(status_file)  # a broken status file used to stop every request, forever
        target = OUT / f"{name}_lora_f16.ckpt"
        res = None
        if src.exists() and done.get("source_mtime") == src.stat().st_mtime:
            if done.get("ok") and target.exists():
                try:
                    ok, msg = restart_server(bool(req.get("dry_run")) or DRY_RUN)
                except Exception as e:
                    ok, msg = False, f"Couldn't restart the Draw Things server: {e}"
                res = {"ok": ok, "message": msg, "dry_run": bool(req.get("dry_run")) or DRY_RUN}
            else:
                res = {"ok": False, "message": "Draw Things couldn't convert this LoRA."}
        elif time.time() - req.get("at", 0) > REQUEST_TIMEOUT:
            res = {"ok": False, "message": "Timed out waiting for the conversion."}
        if res is not None:
            res.update(id=rid, name=name, at=time.time())
            result.write_text(json.dumps(res))
            req_file.unlink(missing_ok=True)
            log(f"send-to-Draw-Things {name}: {res['message']}")


def heartbeat(restart_ok):
    try:
        tmp = STATUS / ".alive.tmp"
        tmp.write_text(json.dumps({"at": time.time(), "can_restart": restart_ok, "dry_run": DRY_RUN}))
        tmp.replace(STATUS / ".alive")
    except Exception:
        pass


def main():
    STATUS.mkdir(parents=True, exist_ok=True)
    restart_ok = can_restart()
    log(f"watching {IN} -> {OUT} (version {VERSION}); restart button: "
        f"{'on' if restart_ok else 'off (no Docker socket)'}{' [dry run]' if DRY_RUN else ''}")
    last_check = time.time()
    while True:
        heartbeat(restart_ok)
        for src in sorted(IN.glob("*.safetensors")):
            try:
                st = src.stat()
                if time.time() - st.st_mtime < 15:  # still being copied
                    continue
                done = read_json(STATUS / f"{src.stem}.json")
                target = OUT / f"{src.stem}_lora_f16.ckpt"
                if done.get("source_mtime") == st.st_mtime and (target.exists() or not done.get("ok")):
                    continue  # up to date (or failed for this exact file; don't loop)
                convert(src)
            except Exception as e:
                log(f"error on {src.name}: {e}")
                try:  # remember the failure so we don't retry this exact file every few seconds
                    (STATUS / f"{src.stem}.json").write_text(json.dumps(
                        {"ok": False, "source_mtime": src.stat().st_mtime, "error": str(e), "at": time.time()}))
                except Exception:
                    pass
        try:
            handle_requests()
        except Exception as e:
            log(f"error handling requests: {e}")
        if time.time() - last_check > 300:
            restart_ok, last_check = can_restart(), time.time()
        time.sleep(min(INTERVAL, 3) if any(STATUS.glob("restart-request-*.json")) else INTERVAL)


if __name__ == "__main__":
    main()
