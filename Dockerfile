# Fetch LoRA - one-button LoRA trainer
# Training engine: ostris/ai-toolkit (MIT), pinned to a tested release.
FROM ostris/aitoolkit:0.13.42

RUN pip install --no-cache-dir --break-system-packages \
      fastapi==0.115.* "uvicorn[standard]==0.32.*" python-multipart==0.0.* pillow-heif==0.* pyyaml

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
