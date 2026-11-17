"""Render a test image with a trained LoRA through a ComfyUI server (FLUX.2 klein 4B).

Usage: python comfy_test_render.py --lora dragon_bo_dog.safetensors --prompt "..." [--no-lora]
"""
import argparse, json, time, urllib.request, uuid

p = argparse.ArgumentParser()
p.add_argument("--server", default="http://127.0.0.1:8189")
p.add_argument("--unet", default="flux-2-klein-base-4b.safetensors")
p.add_argument("--te", default="qwen_3_4b.safetensors")
p.add_argument("--vae", default="flux\\flux2-vae.safetensors")
p.add_argument("--lora", default="")
p.add_argument("--strength", type=float, default=1.0)
p.add_argument("--prompt", required=True)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--cfg", type=float, default=4.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--size", type=int, default=1024)
p.add_argument("--prefix", default="dragon/test")
a = p.parse_args()

model = ["1", 0]
g = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.te, "type": "flux2"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
}
if a.lora:
    g["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": a.lora, "strength_model": a.strength}}
    model = ["4", 0]
g.update({
    "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": a.prompt}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
    "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": a.size, "height": a.size, "batch_size": 1}},
    "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
    "9": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["5", 0], "negative": ["6", 0], "cfg": a.cfg}},
    "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "11": {"class_type": "Flux2Scheduler", "inputs": {"steps": a.steps, "width": a.size, "height": a.size}},
    "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["9", 0], "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["7", 0]}},
    "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
    "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": a.prefix}},
})
body = json.dumps({"prompt": g, "client_id": uuid.uuid4().hex}).encode()
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(a.server + "/prompt", body, {"Content-Type": "application/json"})))
pid = r["prompt_id"]
while True:
    time.sleep(2)
    h = json.load(urllib.request.urlopen(f"{a.server}/history/{pid}"))
    if pid in h:
        st = h[pid].get("status", {})
        if st.get("status_str") == "error":
            print("ERROR", json.dumps(st)[:2000]); break
        for node in h[pid]["outputs"].values():
            for im in node.get("images", []):
                print("OUTPUT", im["subfolder"], im["filename"])
        break
print(f"seconds={time.time()-t0:.1f}")
