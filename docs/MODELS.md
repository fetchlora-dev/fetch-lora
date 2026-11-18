# Which AI model learns your pet? (the model matrix)

Fetch LoRA can train your LoRA on three different base models. Bigger models usually put your pet
into new scenes more convincingly (the astronaut test), but they train slower, download more, and
need more memory, both on the PC that trains and on the phone, iPad or Mac that uses the LoRA.

| Level | Models offered | Picked for you |
|---|---|---|
| **Quick test** | FLUX.2 Klein 4B only | Klein 4B |
| **Good** | Klein 4B, Z-Image 6B | Z-Image if your GPU can train it and your device can run it, else Klein 4B |
| **Best** | Klein 4B, Z-Image 6B, Qwen-Image 20B | The most realistic one that passes both checks: Qwen-Image, then Z-Image, then Klein 4B |

The page shows a one-line **"Why this model"** under the picker. You can always tap a different
(not greyed-out) model.

## The three models

| | FLUX.2 Klein 4B | Z-Image 6B | Qwen-Image 20B |
|---|---|---|---|
| Hugging Face repo trained on | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Base, **not** Turbo) | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) |
| Made by | Black Forest Labs (Germany) | Alibaba Tongyi Lab (China) | Alibaba Qwen team (China) |
| License | Apache 2.0, not gated | Apache 2.0, not gated | Apache 2.0, not gated |
| First-use download | ~16 GB | ~21 GB | ~58 GB (+0.3 GB adapter on 24 GB cards) |
| GPU memory to train | 12 GB+ | 16 GB card | 24 GB+ (48 GB is comfortable) |
| ai-toolkit `arch` | `flux2_klein_4b` | `zimage` | `qwen_image` |
| Speed (RTX 5070 Ti, first guess) | ~2.0 s/step | ~3 s/step (estimate) | n/a on 16 GB |
| Use it in Draw Things with | FLUX.2 [klein] 4B | Z Image Base 1.0 | Qwen Image 1.0 |
| One-tap "Send to Draw Things" | Yes | No: import the file | No: import the file |

### Steps (how long it practices)

| Model | Good | Best |
|---|---|---|
| Klein 4B | photos x 25, 800 to 2,000 (unchanged) | photos x 60, 2,000 to 6,000 (unchanged) |
| Z-Image | photos x 60, 1,500 to 2,500 | photos x 120, 3,600 to 4,800 |
| Qwen-Image | not offered | photos x 60, 2,000 to 4,000 |

Why these numbers: Z-Image is 1.5x bigger than Klein and its Base version learns a new subject
more slowly per step, so it gets more steps. Best tops out at 4,800 (a common ceiling for
character LoRAs at learning rate 1e-4, rank 32); a version is saved about every fifth of the run,
so if the last one looks overdone you can pick an earlier one. Qwen-Image's range follows
ai-toolkit's own example ("500 - 4000 is a good range", default 2,000). These are starting points,
not measured results. Change them in `app/models.py`.

Test pictures: the same three prompts for every model (meadow, astronaut, renaissance portrait).
Klein: guidance 4, 30 steps. Z-Image: CFG 4 (written as 5 because ai-toolkit subtracts 1 for
Z-Image), 30 steps (model card: CFG 3 to 5, 28 to 50 steps). Qwen-Image: guidance 3, 25 steps
(ai-toolkit's example).

### Grey-outs: two separate checks

1. **Training GPU (on the PC).** Fetch reads the training GPU's memory with `nvidia-smi`. A model
   that needs more than the card has is greyed out with the reason (for example "Needs a GPU with
   about 24 GB to train. RTX 5070 Ti has 16 GB."). Set `DRAGON_VRAM_GB` to override the detected
   number. The server also refuses such a job, so the check can't be skipped.
2. **Your device (Draw Things).** A web page can't see how much memory an iPhone or iPad has, so
   the page asks *"Where will you use it in Draw Things?"* and remembers your answer on that device.
   - iPhone (any): Qwen-Image is greyed out.
   - Older iPhone (15 / 14 or earlier): Klein 4B and Z-Image get a warning to use Draw Things'
     cloud offload.
   - iPad with M1 or later (8 GB): Qwen-Image is greyed out.
   - 16 GB iPad Pro, Mac (16 GB+), Not sure: nothing greyed out; the notes are shown.

Device notes shown in the page:
- Klein 4B and Z-Image: *Works in Draw Things on iPhone 15 Pro or later and any M1-or-later iPad;
  older devices can use Draw Things' cloud offload.*
- Qwen-Image: *Needs about 11 GB in Draw Things: a Mac or a 16 GB iPad Pro (1 TB/2 TB models); not iPhone.*

Sources: Draw Things posts from Dec 9 and 10, 2025 (Z-Image officially supported, including
Z-Image LoRAs; Z-Image at 6-bit uses about 4 GiB; Qwen-Image about 11 GiB) and the Draw Things wiki
(8 GB is the practical floor; older devices can use Server Offload). Draw Things' open-source model
list includes "Z Image Base 1.0", "Z Image Turbo 1.0" and "Qwen Image 1.0", and its LoRA importer
handles Z-Image and Qwen-Image LoRAs.

**Z-Image Base vs Turbo:** Fetch trains on **Base** (the version made for fine-tuning). Use the LoRA
with **Z Image Base 1.0** in Draw Things. It *may* also work on Z Image Turbo (Draw Things reported a
Base-trained LoRA working well on Turbo), but likeness LoRAs on Turbo are untested here.

### Why only Klein 4B can be one-tap sent

The optional Draw Things add-on (`converter/`) converts every `*.safetensors` in `data/loras/` as a
Klein 4B LoRA, and the Draw Things server on the PC only has Klein 4B. So Z-Image and Qwen-Image
LoRAs are copied to `data/loras/zimage/` and `data/loras/qwen/` instead, where the add-on never
looks, and the job page tells you to **Download** and **Import** them in the Draw Things app.

## Qwen-Image on a rented GPU (optional, untested)

No 24 GB card at home? You can rent one by the hour (for example on RunPod) and train there with
AI Toolkit directly. Roughly $5 to $10 for one Best run on a 48 GB card.

1. In Fetch, run a training on the same photos (any model) so the photos get cropped and captioned.
   They end up in `data/jobs/<job id>/images/` (pairs of `001.jpg` + `001.txt`).
2. On RunPod, deploy a 48 GB GPU (A6000, L40S, or bigger) with Ostris' official **AI Toolkit**
   template, and give it at least 100 GB of disk (Qwen-Image downloads ~58 GB).
3. Upload the `images` folder to `/workspace/<your word>/images` and
   [`docs/runpod_qwen_48gb.yaml`](runpod_qwen_48gb.yaml) (made by Fetch's own config builder).
   Change `biscuit_dog` to your word and the steps to photos x 60 (2,000 to 4,000).
4. In the pod's terminal: `cd /app/ai-toolkit && python run.py /workspace/runpod_qwen_48gb.yaml`
   (or paste the config into the AI Toolkit web UI's advanced editor).
5. Download the `.safetensors` from `/workspace/output/<your word>/`, then **stop the pod** so it
   stops billing.

On a 24 GB card, Fetch itself switches Qwen-Image to 3-bit weights with Ostris' accuracy recovery
adapter, low-VRAM mode, cached text embeddings and pictures up to 768 px (ai-toolkit's 24 GB recipe).
