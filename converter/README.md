# Fetch LoRA → Draw Things converter (optional, GPL-3.0)

This folder is a separate program with a separate license. It is **GPL-3.0** (see `LICENSE` here)
because it builds and runs `LoRAConverter` from Draw Things' open-source code,
[draw-things-community](https://github.com/drawthingsai/draw-things-community) (GPL-3.0).
The rest of Fetch LoRA is MIT and doesn't link to it; the two only share a folder of files.

What it does: whenever Fetch LoRA finishes a LoRA, this container converts it into Draw Things' own
`<name>_lora_f16.ckpt` format and puts it straight into your Draw Things gRPC server's models
folder, and records it in `custom_lora.json` there. No import step on the iPad.

**Send to Draw Things:** when you tap that button on a finished training, the trainer leaves a small
request file next to the LoRAs. Once the LoRA is converted, this container restarts the Draw Things
server so it picks the LoRA up (the server only reads its LoRA list at startup). For that it needs the
Docker socket (`/var/run/docker.sock`, mounted in `docker-compose.drawthings.yml`). The code makes exactly
two kinds of Docker calls: *inspect* and *restart* of the one container named in `DT_SERVER_CONTAINER`
(default `drawthings-grpc`). Remove the socket line to turn the button off; set `DT_RESTART_DRY_RUN=1` to
test it without restarting anything. It also writes a heartbeat (`.drawthings/.alive`) so the trainer page
knows the add-on is running.

Turn it on by adding this line to `.env` (Windows uses `;`, Linux/macOS `:`):

    COMPOSE_FILE=docker-compose.yml;docker-compose.drawthings.yml

and set `DT_MODELS_VOLUME` to the Docker volume your gRPC server keeps its models in
(`docker volume ls`), then `docker compose up -d --build`. The first build compiles the converter
(about 15–25 minutes, CPU only); after that it's instant.

Source: built from the commit in `DT_COMMIT` (see `Dockerfile`); the exact commit is also saved
inside the image at `/opt/dt/SOURCE_COMMIT.txt`. If you share the image, share this folder too
(GPL-3.0 requires offering the source).
