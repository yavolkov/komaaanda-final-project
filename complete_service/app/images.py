from base64 import b64encode
from io import BytesIO
from typing import IO

from PIL.Image import Image
from PIL.Image import open as _open_image


def open_image(image_fp: IO[bytes]) -> Image:
    image = _open_image(image_fp)
    image.load()
    return image.convert("RGB")


def _image_b64encode(image: Image) -> str:
    with BytesIO() as io:
        image.save(io, format="png", quality=100)
        io.seek(0)
        return b64encode(io.read()).decode()


def image_to_img_src(image: Image) -> str:
    return f"data:image/png;base64,{_image_b64encode(image)}"
