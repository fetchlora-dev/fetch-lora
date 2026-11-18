# Which AI model learns your pet? (the model matrix)

Fetch LoRA can train your LoRA on four different base models. Bigger and newer models usually put
your pet into new scenes more convincingly (the astronaut test), but they train slower, download
more, and need more memory, both on the PC that trains and on the phone, iPad or Mac that uses it.

| Level | Models offered | Picked for you (most realistic that passes both checks) |
|---|---|---|
| **Quick test** | FLUX.2 Klein 4B only | Klein 4B |
| **Good** | Klein 4B, Z-Image 6B, MOSS-Image 5B | MOSS-Image, then Z-Image, then Klein 4B |
| **Best** | Klein 4B, Z-Image 6B, MOSS-Image 5B, Qwen-Image-2512 20B | Qwen-Image-2512, then MOSS-Image, then Z-Image, then Klein 4B |

The page shows a one-line **"Why this model"** under the picker. You can always tap a different
(not greyed-out) model.

## The four models

| | FLUX.2 Klein 4B | Z-Image 6B | MOSS-Image 5B | Qwen-Image-2512 20B |
|---|---|---|---|---|
| Repo trained on | [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Base, not Turbo) | [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) |
| Made by | Black Forest Labs (Germany) | Alibaba Tongyi Lab (China) | OpenMOSS team, Fudan University (China) | Alibaba Qwen team (China) |
| On Hugging Face since | Jan 2026 | Jan 2026 | Oct 2026 | Dec 2025 |
| License | Apache 2.0, not gated | Apache 2.0, not gated | Apache 2.0, not gated | Apache 2.0, not gated |
| First-use download | ~16 GB | ~21 GB | ~27 GB (9B text encoder) | ~58 GB (+0.3 GB adapter on 24 GB cards) |
| GPU memory to train | 12 GB+ | 16 GB card | 12 GB+ (8-bit + offloading) | 24 GB+ (48 GB is comfortable) |
| ai-toolkit `arch` | `flux2_klein_4b` | `zimage` | `moss_image` | `qwen_image` |
| Speed, RTX 5070 Ti (first guess) | ~2.0 s/step (measured) | ~3 s/step | ~3.5 s/step | n/a on 16 GB |
| Use it in Draw Things with | FLUX.2 [klein] 4B | Z Image Base 1.0 | MOSS Image SFT (or Base) | Qwen Image 2512 |
| One-tap "Send to Draw Things" | Yes | No: import the file | No: import the file | No: import the file |

**Why this order (auto-pick ranking):** Qwen-Image-2512 is by far the largest. MOSS-Image is ranked
above Z-Image because it is the newest (Oct 2026), pairs a 5B image model with a much bigger 9B text
encoder, and early testers report strong character likeness from varied photos (ImageLoRA and
ModelTrends guides, a note.com 12 GB walkthrough). One tester also notes it can lack fine skin
detail. This ranking is from community reports, not our own side-by-side test yet.

### Steps (how long it practices)

| Model | Good | Best |
|---|---|---|
| Klein 4B | photos x 25, 800 to 2,000 (unchanged) | photos x 60, 2,000 to 6,000 (unchanged) |
| Z-Image | photos x 60, 1,500 to 2,500 | photos x 120, 3,600 to 4,800 |
| MOSS-Image | photos x 60, 1,500 to 2,500 | photos x 100, 3,000 to 4,000 |
| Qwen-Image-2512 | not offered | photos x 90, 2,000 to 4,500 |

Why: Z-Image is 1.5x bigger than Klein and its Base version learns a new subject more slowly per
step, so it gets more steps; Best tops out at 4,800. MOSS-Image character LoRAs "typically need
2,000-3,000" steps (community guides), so Best gives 3,000-4,000. Qwen-Image-2512 follows the
community guidance of ~80-100 repeats per photo. A version is saved about every fifth of the run,
so if the last one looks overdone you can pick an earlier one. These are starting points, not
measured results. Change them in `app/models.py`.

Per-model settings (from ai-toolkit's own presets unless noted): Klein unchanged. Z-Image: CFG 4 for
test pictures (written as 5 because ai-toolkit subtracts 1), 30 steps. MOSS-Image: 8-bit ConvRot
weights, block 22 skipped on the CFG negative branch and left out of the LoRA, `shift` timesteps,
text-encoder output cached, on cards under 20 GB the text encoder is offloaded to system RAM;
test pictures CFG 3, 30 steps. Qwen-Image-2512: `sigmoid` timesteps (community), rank 32; on 24-39 GB
cards 3-bit weights + Ostris' accuracy recovery adapter, low-VRAM mode, cached text embeddings and
pictures up to 768 px; test pictures guidance 3, 25 steps.

### Grey-outs: two separate checks

1. **Training GPU (on the PC).** Fetch reads the training GPU's memory with `nvidia-smi`. A model
   that needs more than the card has is greyed out with the reason (for example "Needs a GPU with
   about 24 GB to train. RTX 5070 Ti has 16 GB."). `DRAGON_VRAM_GB` overrides the detected number.
   The server also refuses such a job, so the check can't be skipped.
2. **Your device (Draw Things).** A web page can't see how much memory an iPhone or iPad has, so
   the page asks *"Where will you use it in Draw Things?"* and remembers the answer on that device.
   - Any iPhone: Qwen-Image-2512 greyed out. iPhone 15 Pro+: MOSS-Image warning (untested).
   - Older iPhone (15 / 14 or earlier): Klein 4B, Z-Image and MOSS-Image get a cloud-offload warning.
   - iPad with M1 or later (8 GB): Qwen-Image-2512 greyed out.
   - 16 GB iPad Pro, Mac (16 GB+), Not sure: nothing greyed out; the notes are shown.

Device notes shown in the page:
- Klein 4B and Z-Image: *Works in Draw Things on iPhone 15 Pro or later and any M1-or-later iPad;
  older devices can use Draw Things' cloud offload.*
- MOSS-Image: *Runs in Draw Things on iPad (confirmed) and Mac. Its 9B text encoder makes it heavier
  than Z-Image; iPhone is untested, so use Draw Things' cloud offload there if it struggles.*
- Qwen-Image-2512: *Needs about 11 GB in Draw Things: a Mac or a 16 GB iPad Pro (1 TB/2 TB models); not iPhone.*

Sources: Draw Things posts from Dec 9 and 10, 2025 (Z-Image officially supported, including Z-Image
LoRAs; Z-Image at 6-bit about 4 GiB; Qwen-Image about 11 GiB) and the Draw Things wiki (8 GB is the
practical floor; older devices can use Server Offload). Draw Things' open-source code
([draw-things-community](https://github.com/drawthingsai/draw-things-community)) lists "Z Image Base
1.0", "MOSS Image SFT", "MOSS Image Base" and "Qwen Image 2512" models, and its LoRA importer has
Z-Image, MOSS-Image and Qwen-Image cases.

**Base vs Turbo / SFT:** Fetch trains Z-Image and MOSS-Image on their **Base** versions (made for
fine-tuning). Use a Z-Image LoRA with **Z Image Base 1.0** (Turbo may work but is untested). For
MOSS-Image, testers report Base-trained LoRAs work on **MOSS Image SFT** (the everyday version).

### Why only Klein 4B can be one-tap sent

The optional Draw Things add-on (`converter/`) converts every `*.safetensors` in `data/loras/` as a
Klein 4B LoRA, and the Draw Things server on the PC only has Klein 4B. So other LoRAs are copied to
`data/loras/<model>/` (for example `data/loras/zimage/`), where the add-on never looks, and the job
page says to **Download** and **Import** them in the Draw Things app.

## Qwen-Image-2512 on a rented GPU (optional, untested)

No 24 GB card at home? Rent one by the hour (for example on RunPod) and train there with AI Toolkit
directly. Very roughly $5 to $10 for one Best run on a 48 GB card.

1. In Fetch, run a training on the same photos (any model) so the photos get cropped and captioned.
   They end up in `data/jobs/<job id>/images/` (pairs of `001.jpg` + `001.txt`).
2. On RunPod, deploy a 48 GB GPU (A6000, L40S, or bigger) with Ostris' official **AI Toolkit**
   template, with at least 100 GB of disk (the model downloads ~58 GB).
3. Upload the `images` folder to `/workspace/<your word>/images` and
   [`docs/runpod_qwen_48gb.yaml`](runpod_qwen_48gb.yaml) (made by Fetch's own config builder).
   Change `biscuit_dog` to your word and the steps to photos x 90 (2,000 to 4,500).
4. In the pod's terminal, run `python run.py /workspace/runpod_qwen_48gb.yaml` from the AI Toolkit
   folder (or paste the config into the AI Toolkit web UI's advanced editor).
5. Download the `.safetensors` from `/workspace/output/<your word>/`, then **stop the pod** so it
   stops billing.
