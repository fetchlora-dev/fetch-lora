# Fetch LoRA - one-button LoRA trainer
# Training engine: ostris/ai-toolkit (MIT), pinned to a tested release.
FROM ostris/aitoolkit:0.13.42

# Exact versions, as tested on top of ostris/aitoolkit:0.13.42 (Nov 2026), so a rebuild never
# quietly picks up a newer release. The image's own pillow, numpy, scipy and opencv are used as-is.
RUN pip install --no-cache-dir --break-system-packages \
      fastapi==0.115.14 "uvicorn[standard]==0.32.1" python-multipart==0.0.32 pillow-heif==0.22.0 pyyaml==6.0.3

# Photo check / auto-crop: torchvision's small COCO detector (BSD-3-Clause), baked in (~75 MB)
# so it works offline and on the first photo. Runs on the CPU.
RUN python -c "from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as f, FasterRCNN_MobileNet_V3_Large_FPN_Weights as W; f(weights=W.DEFAULT)"

COPY app /app/dragon
WORKDIR /app/dragon

ENV DRAGON_DATA=/data \
    DRAGON_EXPORTS=/exports \
    AITK_DIR=/app/ai-toolkit \
    HF_HOME=/hf-cache \
    PYTHONUNBUFFERED=1

EXPOSE 8675
# Replace ai-toolkit's own UI with our single page.
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8675"]
