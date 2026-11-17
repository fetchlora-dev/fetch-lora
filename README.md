# 🐕 Fetch LoRA

**Teach an AI image model a new word, from your iPad or phone, on your own GPU.**

> Idea by Eric. Built by Eric's AI assistant, GraphicGuru. Free and open source — open to contributors.

Fetch LoRA is free and open source. It is not for sale and never will be, and it comes with no
support promises: it's a hobby project shared as-is (see [LICENSE](LICENSE)). Issues and pull
requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## What it is

A **LoRA** is a small add-on file (about 100 MB) that teaches a big image model one new thing:
your dog, your cat, an object you own, or an art style. After training you type a made-up word
like `biscuit_dog` in a prompt, and the model draws *your* dog:
*"a photo of biscuit_dog walking through a sunlit meadow."*

Training a LoRA normally means Python scripts, config files and a lot of settings. Fetch LoRA
hides all of that behind one web page:

1. Pick 20–40 photos on your iPad or phone.
2. Type a name.
3. Press **Start**.

Your PC's graphics card does the learning (often overnight), and the page shows the AI's practice
pictures getting better as it goes. When it's done, you pick the version you like best and use it in
[Draw Things](https://drawthings.ai) (iPad/iPhone) or [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

- **Private:** photos go from your device to your own PC over your own network (or a private VPN
  like [Tailscale](https://tailscale.com)). Nothing is uploaded to a cloud service.
- **Checks your photos first:** flags blurry, tiny, near-duplicate and "can't find the dog" photos.
- **Watch it learn, then pick the best version:** about five saved versions, each with test pictures.
- **A waiting line:** add several trainings and they run one after another.
- **Optional extras:** auto-crop around the subject, background removal, a phone notification when
  it's done, and a one-tap "Send to Draw Things" add-on.

## How it works (plain English)

Think of it as a small restaurant:

- **The menu (the web page)** is `app/static/index.html`. It runs in Safari on your iPad and only
  ever talks to your PC.
- **The waiter (the web server)** is `app/server.py`, a [FastAPI](https://fastapi.tiangolo.com) app
  in Python. It takes your photos, checks them, writes down the order and puts it in line.
- **The kitchen (the trainer)** is [AI Toolkit](https://github.com/ostris/ai-toolkit) by Ostris.
  The server writes a recipe (`config.yaml`) and AI Toolkit cooks: it trains the LoRA on the
  [FLUX.2 [klein] base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) image model
  on your GPU, saving a version and drawing three test pictures every so often.
- **The page checks back** every few seconds, reads AI Toolkit's progress from its log, and shows
  you the step count, the time left and the newest test pictures.
- **Takeout:** the version you pick is copied to `data/loras/` and can be downloaded from the page.

Everything runs inside [Docker](https://www.docker.com) containers, so you don't have to install
Python, PyTorch or CUDA yourself. For a file-by-file tour and a diagram, read
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## What you need (hardware)

| | Minimum | Tested on |
|---|---|---|
| GPU | NVIDIA with 16 GB VRAM (12 GB works, a bit slower) | RTX 5070 Ti 16 GB, RTX 4070 Super 12 GB |
| NVIDIA driver | 580 or newer (CUDA 13) | |
| RAM | 32 GB | 64 GB |
| Disk | 40 GB free (the base model alone is ~16 GB) | |
| OS | Windows 10/11 with Docker Desktop (WSL 2), or Linux with Docker + NVIDIA Container Toolkit | Windows 11 |
| Phone / tablet | Any modern browser (Safari is fine) | iPad, iPhone |

AMD, Intel and Apple Silicon GPUs are **not** supported (the trainer needs CUDA).

Rough times on an RTX 5070 Ti: a *Quick test* (300 steps) takes about 12–15 minutes end to end;
*Good* about 30 minutes to 1¼ hours; *Best* about 1¼ to 3½ hours, depending on how many photos you pick.
The very first run also downloads the base model once (~16 GB).

## Quick start

1. **Update your NVIDIA driver** ([nvidia.com/drivers](https://www.nvidia.com/drivers)).
2. **Install Docker Desktop** ([docker.com](https://www.docker.com/products/docker-desktop/)) with WSL 2 on
   Windows, then check Docker can see your GPU:
   ```
   docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
   ```
3. **Get the code and set it up:**
   ```
   git clone https://github.com/<you>/fetch-lora.git
   cd fetch-lora
   copy .env.example .env          # Linux/macOS: cp .env.example .env
   notepad .env                    # set TRAINER_PIN= to a PIN you'll remember
   ```
   With more than one GPU, run `nvidia-smi -L` and put the one you want in `DRAGON_GPU`.
4. **Start it:**
   ```
   docker compose up -d --build
   ```
5. On your iPad or phone, open `http://<your-pc-address>:8677`, type the PIN once, and in Safari tap
   Share → **Add to Home Screen** for an app icon. (Same Wi-Fi: use the PC's local IP. Away from home:
   use a private VPN such as Tailscale on both devices.)

To change the PIN later, edit `TRAINER_PIN=` in `.env` and run `docker compose up -d`; every device
is asked for the new PIN. Logs: `docker logs dragon-trainer`, or the *Technical log* on each training page.

## Your first training

1. **Pick photos.** Different angles, lighting and places; a few close-ups and a few full-body shots.
   Avoid other animals or people in the frame, heavy filters and near-duplicates. The page flags
   photos that might hurt the result; tap **Remove those** or ✕ on a single photo. You can also type
   optional **tags** under a photo (`frisbee, mid-air`) for things you want to control in prompts later.
2. **Name it**, lowercase with underscores, like `biscuit_dog`. That's the word you'll type later.
3. **Choose what it is** (dog, cat, person, object, style) and **how long to practice**
   (*Quick test*, *Good*, or *Best*). Steps grow with the number of photos.
4. **Press Start.** You can close the page; training keeps going.
5. **When it's done**, pick your favorite version, then **Download** it or tap **Send to Draw Things**.

**Using it in Draw Things:** download the `.safetensors` file, then in Draw Things open the model menu →
**LoRA** → **Manage** → **Import**. Choose **FLUX.2 [klein] 4B** as the model, switch your LoRA on, and
use your word in the prompt. The optional `converter/` add-on can do this for a Draw Things gRPC server
automatically; see [converter/README.md](converter/README.md).

**Using it in ComfyUI:** put the file in `models/loras`, load FLUX.2 klein 4B with the Qwen3-4B text
encoder, and add a *LoraLoaderModelOnly* node. `scripts/comfy_test_render.py` is an example.

**Phone notification (optional):** under *More options → Tell my phone when it's done*, paste a webhook
address (for example a [Home Assistant](https://www.home-assistant.io) webhook). When a training
finishes or stops, Fetch LoRA sends it a JSON POST like
`{"job": "…", "name": "biscuit_dog", "status": "done", "message": "…", "url": "…"}`.

## Please be kind

- **Ask before you train on a person.** Anyone with the file can make pictures of that face.
- **Keep LoRAs of real people private**, especially children. Don't upload them to public sites.
- Pets, art you made yourself and objects you own are perfect for learning.
- Follow the base model's usage rules (see the FLUX.2 [klein] model card's *Out-of-Scope Use* section).

## Licenses

**Fetch LoRA itself** (everything except `converter/`) is under the [MIT License](LICENSE),
copyright "Fetch LoRA contributors". You can use, copy, change and share it, including commercially,
as long as you keep the license notice. It comes with no warranty.

**`converter/`** (the optional Draw Things add-on) is a separate program under **GPL-3.0**
([converter/LICENSE](converter/LICENSE)), because it builds Draw Things' open-source
`LoRAConverter` from [draw-things-community](https://github.com/drawthingsai/draw-things-community)
(GPL-3.0). The MIT trainer doesn't link to it; the two only share a folder of files. If you share a
built converter image, you must also offer its source.

Fetch LoRA downloads and uses these third-party models and tools. They are **not** included in this
repository and keep their own licenses:

| Component | Used for | License | Notes |
|---|---|---|---|
| [FLUX.2 [klein] base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) by Black Forest Labs | The image model the LoRA is trained on (includes its Qwen3-4B text encoder and VAE) | Apache 2.0 | Not gated. The model card says outputs can be used commercially under Apache 2.0, and lists out-of-scope uses. The **9B** klein variants use a non-commercial license and are **not** used here. |
| [AI Toolkit](https://github.com/ostris/ai-toolkit) by Ostris (Docker image `ostris/aitoolkit:0.13.42`) | The training engine | MIT | The Docker image also bundles PyTorch, CUDA and other packages under their own licenses. |
| [BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) by ZhengPeng7 | Optional background removal | MIT | Pinned revision, downloaded on first use (~180 MB). Loaded with `trust_remote_code`, so the pin matters. |
| [torchvision](https://github.com/pytorch/vision) Faster R-CNN MobileNetV3-Large FPN | Photo check and auto-crop (finds the dog/person) | Code: BSD-3-Clause | The weights are trained on the [COCO](https://cocodataset.org) dataset. torchvision notes that pre-trained weights "may have their own licenses or terms" from their training data (COCO annotations are CC BY 4.0; the images come from Flickr under their own terms). |
| [OpenCV](https://opencv.org), [SciPy](https://scipy.org), [FastAPI](https://fastapi.tiangolo.com), [Pillow](https://python-pillow.org) | Blur check, duplicate check, web server, image handling | Apache 2.0 / BSD-3-Clause / MIT / MIT-CMU | |
| [pillow-heif](https://github.com/bigcat88/pillow_heif) | Opening iPhone HEIC photos | BSD-3-Clause | Its prebuilt wheels bundle libheif and codec libraries that carry their own (LGPL/GPL-family) licenses. |
| [draw-things-community](https://github.com/drawthingsai/draw-things-community) | `LoRAConverter`, only in the optional `converter/` add-on | GPL-3.0 | Built from a pinned commit. |

Draw Things, ComfyUI, Home Assistant and Tailscale are separate projects by their own developers.
Fetch LoRA is not affiliated with any of them.

*License details were checked against each project's own license file or model card in November 2026.
Always check the current terms yourself before relying on them.*
