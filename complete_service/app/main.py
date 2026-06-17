from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import base64

from app.model import Predictor
from app.images import image_to_img_src


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Smart Camera Price Service", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/", response_class=HTMLResponse)
async def infer_html(request: Request, file: UploadFile = File(...)):
    ctx = {}

    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")

        prediction = app.state.predictor.predict(image)

        ctx.update(
            image=image_to_img_src(image),
            point_price=prediction.point_price,
            min_price=prediction.min_price,
            max_price=prediction.max_price,
            confidence=prediction.confidence,
            microcategory=prediction.microcategory,
            similar_ads=[
                {
                    "image": image_to_img_src(ad.image) if ad.image is not None else None,
                    "price": ad.price,
                    "title": ad.title,
                }
                for ad in prediction.similar_ads
            ],
        )
    except Exception as err:
        ctx.update(error=str(err))

    return templates.TemplateResponse(request, "index.html", ctx)


@app.post("/predict")
async def predict_api(file: UploadFile = File(...), top_k: int = 5):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть изображение: {err}")

    result = app.state.predictor.predict(image, top_k=top_k)

    similar_ads = []
    for ad in result.similar_ads:
        image_base64 = None

        if ad.image is not None:
            buf = io.BytesIO()
            ad.image.save(buf, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        similar_ads.append({
            "price": round(ad.price, 2),
            "title": ad.title,
            "image_base64": image_base64,
        })

    return JSONResponse({
        "point_price": round(result.point_price, 2),
        "min_price": round(result.min_price, 2),
        "max_price": round(result.max_price, 2),
        "microcategory": result.microcategory,
        "confidence": round(result.confidence, 4),
        "similar_ads": similar_ads,
    })