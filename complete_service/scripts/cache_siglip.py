"""
Запускается один раз во время `docker build`.
Скачивает и кэширует SigLIP внутри image (~400 MB).
Без этого каждый cold start будет скачивать модель.
"""
from transformers import AutoProcessor, AutoModel

MODEL = "google/siglip-base-patch16-224"
print(f"Caching {MODEL}...")
AutoProcessor.from_pretrained(MODEL)
AutoModel.from_pretrained(MODEL)
print("Done.")