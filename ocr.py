import os

import numpy as np
import torch
from PIL import Image

from image_io import clip_box, prepare_ocr_crop, to_pil
from models import get_ocr_model
from settings import settings
from text_utils import clean_text


OCR_MAX_PATCHES = 512
OCR_MAX_NEW_TOKENS = 256


def _collapse_repeats(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    previous = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == previous:
            continue
        out.append(line)
        if stripped:
            previous = stripped
    return "\n".join(out).strip()


def _ocr_infer(crop: Image.Image) -> str:
    processor, model, tokenizer = get_ocr_model()
    inputs = processor(
        images=[crop.convert("RGB")],
        max_num_patches=OCR_MAX_PATCHES,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        texts = model.generate(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
            tokenizer=tokenizer,
            max_new_tokens=OCR_MAX_NEW_TOKENS,
            repetition_penalty=1.0,
        )
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("Hayai OCR returned an invalid response")
    return _collapse_repeats(clean_text(texts[0]))


def _ocr_crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> Image.Image | None:
    image_height, image_width = image.shape[:2]
    x0, y0, x1, y1 = clip_box(x, y, width, height, image_width, image_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return to_pil(image[y0:y1, x0:x1])


def ocr_region(
    img_cv: np.ndarray, x: int, y: int, w: int, h: int, idx: int, page_idx: int
) -> str:
    crop = _ocr_crop(img_cv, x, y, w, h)
    if crop is None:
        return ""
    if settings().crops_dir:
        os.makedirs(settings().crops_dir, exist_ok=True)
        crop.save(os.path.join(settings().crops_dir, f"p{page_idx:03d}_bubble_{idx:02d}.png"))
    text = _ocr_infer(crop)
    if text:
        return text

    enhanced = prepare_ocr_crop(img_cv, x, y, w, h, enhanced=True)
    if enhanced is None:
        return ""
    if settings().crops_dir:
        enhanced.save(
            os.path.join(settings().crops_dir, f"p{page_idx:03d}_bubble_{idx:02d}_retry.png")
        )
    return _ocr_infer(enhanced)
