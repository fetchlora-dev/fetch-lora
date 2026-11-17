"""Optional background removal for Fetch LoRA (off by default).

Runs as its own process so the GPU memory it uses is fully released before training starts.
Model: BiRefNet_lite by ZhengPeng7 (MIT), pinned to one revision. Downloaded once (~180 MB)
into the Hugging Face cache on first use.

Usage: python bgremove.py <folder>   -> every *.jpg in the folder gets a plain white background.
Prints one line per photo; exits non-zero if the model can't run (the trainer then keeps the
photos as they were).
"""
import sys
from pathlib import Path

REPO = "ZhengPeng7/BiRefNet_lite"
REVISION = "aa62cd87eafb9cc43056d08ef3615a14628b831d"


def main(folder: str) -> int:
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageSegmentation.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model.to(dev).eval()
    if dev == "cuda":
        model.half()
    prep = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    files = sorted(Path(folder).glob("*.jpg"))
    for f in files:
        im = Image.open(f).convert("RGB")
        x = prep(im).unsqueeze(0).to(dev)
        if dev == "cuda":
            x = x.half()
        with torch.inference_mode():
            pred = model(x)[-1].sigmoid().float().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(im.size)
        white = Image.new("RGB", im.size, (255, 255, 255))
        out = Image.composite(im, white, mask)
        tmp = f.with_suffix(".tmp.jpg")
        out.save(tmp, quality=95)
        tmp.replace(f)
        print("background removed:", f.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
