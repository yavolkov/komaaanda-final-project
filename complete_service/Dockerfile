FROM python:3.11-slim

WORKDIR /service

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ARTIFACTS_DIR=/artifacts
ENV IMAGES_DIR=/artifacts/listing_images
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/cache_siglip.py scripts/cache_siglip.py
RUN python scripts/cache_siglip.py

COPY app/ ./app/
COPY templates/ ./templates/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
