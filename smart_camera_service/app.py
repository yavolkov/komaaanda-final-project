from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from lib.fake_model import FakePriceModel
from lib.fake_model import get_model
from lib.images import image_to_img_src
from lib.images import open_image

app = FastAPI(title="Smart Camera Price Service")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request) -> Response:
    ctx: dict = {}
    return templates.TemplateResponse(request, "index.html", ctx)


@app.post("/", response_class=HTMLResponse)
def infer_model(
    file: UploadFile,
    request: Request,
    model: FakePriceModel = Depends(get_model, use_cache=True),
) -> Response:
    ctx: dict = {}
    try:
        image = open_image(file.file)
        prediction = model.predict(image)

        ctx.update(
            image=image_to_img_src(image),
            point_price=prediction.point_price,
            min_price=prediction.min_price,
            max_price=prediction.max_price,
            confidence=prediction.confidence,
            microcategory=prediction.microcategory,
            similar_ads=[
                {
                    "image": image_to_img_src(ad.image),
                    "price": ad.price,
                    "title": ad.title,
                }
                for ad in prediction.similar_ads
            ],
        )
    except Exception as err:
        ctx.update(error=str(err))
    return templates.TemplateResponse(request, "index.html", ctx)
