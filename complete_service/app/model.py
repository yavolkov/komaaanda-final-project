"""
model.py — инференс V8_TUNED
Интерфейс совместим с fake_model.py:
    result = predictor.predict(pil_image)
    result.point_price / .min_price / .max_price
    result.microcategory / .confidence
    result.similar_ads[i]["image"] / ["price"] / ["title"]
"""
from dataclasses import dataclass
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from PIL import Image
from transformers import AutoProcessor, AutoModel


@dataclass(slots=True)
class SimilarAd:
    image: Image.Image | None
    price: float
    title: str


@dataclass(slots=True)
class PricePrediction:
    point_price: float
    min_price: float
    max_price: float
    confidence: float
    microcategory: str
    similar_ads: list[SimilarAd]


class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class V8Tuned(nn.Module):
    def __init__(self, in_features: int = 768):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 1024), SwiGLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.4))
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512), SwiGLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3))
        self.skip       = nn.Linear(in_features, 256)
        self.price_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h) + self.skip(x)
        return self.price_head(h)


# ── Контракт ответа ───────────────────────────────

class PredictionResult:
    def __init__(self, point_price, min_price, max_price,
                 microcategory, confidence, similar_ads):
        self.point_price   = point_price    # float, рублей
        self.min_price     = min_price      # float, рублей
        self.max_price     = max_price      # float, рублей
        self.microcategory = microcategory  # str
        self.confidence    = confidence     # float 0..1
        self.similar_ads   = similar_ads    # list[dict]: image, price, title

    def __repr__(self):
        return (f"PredictionResult(point={self.point_price:,.0f} руб, "
                f"range=[{self.min_price:,.0f}, {self.max_price:,.0f}], "
                f"cat={self.microcategory}, conf={self.confidence:.3f}, "
                f"similar={len(self.similar_ads)})")


# ── Главный класс ─────────────────────────────────────────────────────────────

class Predictor:
    SIGLIP_MODEL = "google/siglip-base-patch16-224"

    def __init__(self):
        self.artifacts_dir = os.getenv("ARTIFACTS_DIR", "/artifacts")
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Predictor] device={self.device}")

        # 1. qhat + siglip_dim
        with open(os.path.join(self.artifacts_dir, "v8tuned_meta.json")) as f:
            meta = json.load(f)
        self.qhat       = float(meta["qhat"])
        self.siglip_dim = int(meta.get("siglip_dim", 768))
        print(f"[Predictor] qhat={self.qhat:.4f}  siglip_dim={self.siglip_dim}")

        # 2. SigLIP (веса закэшированы при build)
        self.processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL)
        self.siglip    = AutoModel.from_pretrained(self.SIGLIP_MODEL).to(self.device)
        self.siglip.eval()
        print("[Predictor] SigLIP loaded")

        # 3. V8Tuned
        self.model = V8Tuned(in_features=self.siglip_dim).to(self.device)
        self.model.load_state_dict(torch.load(
            os.path.join(self.artifacts_dir, "v8tuned_weights.pt"),
            map_location=self.device))
        self.model.eval()
        print("[Predictor] V8Tuned loaded")

        # 4. FAISS
        self.index = faiss.read_index(
            os.path.join(self.artifacts_dir, "siglip_train.index"))
        print(f"[Predictor] FAISS: {self.index.ntotal} vectors")

        # 5. Метаданные объявлений
        self.listings = pd.read_parquet(
            os.path.join(self.artifacts_dir, "train_listings_meta.parquet"))
        print(f"[Predictor] Listings: {len(self.listings)} rows  ready.")

    def predict(self, image: Image.Image, top_k: int = 5) -> PredictionResult:
        emb = self._embed(image)
        point_price, min_price, max_price = self._price(emb)
        similar_ads, best_idx, confidence = self._similar(emb, top_k)

        best_row      = self.listings.iloc[best_idx]
        microcategory = str(best_row.get("microcat_name",
                             best_row.get("learning_cat", "unknown")))

        return PredictionResult(
            point_price=point_price, min_price=min_price, max_price=max_price,
            microcategory=microcategory, confidence=confidence,
            similar_ads=similar_ads)

    def _embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(
            images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.siglip.get_image_features(**inputs)
        emb_np = emb.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)
        return emb_np[0]

    def _price(self, emb: np.ndarray):
        """log1p-масштаб → рубли через expm1"""
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_pred = self.model(x).item()
        return (
            float(np.expm1(log_pred)),
            float(np.expm1(log_pred - self.qhat)),
            float(np.expm1(log_pred + self.qhat)),
        )

    def _similar(self, emb: np.ndarray, top_k: int):
        q = emb.reshape(1, -1).astype(np.float32)
        sims, idxs = self.index.search(q, top_k)
        ads = []
        for sim, idx in zip(sims[0], idxs[0]):
            row     = self.listings.iloc[idx]
            img_path = row.get("image_docker_path", None)
            pil_img = None
            if img_path and os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                except Exception:
                    pass
            ads.append(SimilarAd(
                image=pil_img,
                price=float(row["price"]),
                title=str(row.get("title", "")),
            ))
        return ads, int(idxs[0][0]), float(sims[0][0]) if len(sims[0]) else 0.0