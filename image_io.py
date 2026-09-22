import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OCR_MAX_SIDE = 2048


def read_image(path: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                rgb = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                rgb = oriented.convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot decode image: {path}") from error


def write_image(path: str, image: np.ndarray) -> None:
    suffix = Path(path).suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a non-empty uint8 image")
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or three-channel BGR image")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def encode_image(image: Image.Image, max_side: int | None = None) -> str:
    image = image.convert("RGB")
    if max_side is not None:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def clip_box(
    x: int, y: int, width: int, height: int, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x0 = min(image_width, max(0, x))
    y0 = min(image_height, max(0, y))
    x1 = min(image_width, max(0, x + max(0, width)))
    y1 = min(image_height, max(0, y + max(0, height)))
    return x0, y0, x1, y1


def prepare_ocr_crop(
    image: np.ndarray, x: int, y: int, width: int, height: int, enhanced: bool = True
) -> Image.Image | None:
    image_height, image_width = image.shape[:2]
    x0, y0, x1, y1 = clip_box(x, y, width, height, image_width, image_height)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    border = 64 if enhanced else 32
    scale = min(3 if enhanced else 2, (OCR_MAX_SIDE - border * 2) / max(crop.shape[:2]))
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    if enhanced:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    crop = cv2.copyMakeBorder(
        crop,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255 if enhanced else (255, 255, 255),
    )
    return to_pil(crop)
