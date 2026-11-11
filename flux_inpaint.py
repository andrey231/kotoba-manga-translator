import logging
import math

import cv2
import numpy as np
import torch
from PIL import Image

from models import DEVICE, get_flux_inpainter

logger = logging.getLogger(__name__)

FLUX_STEPS = 4
FLUX_GUIDANCE = 1.0
FLUX_SEED = 0
FLUX_TARGET_PIXELS = 1024 * 1024
FLUX_MAX_UPSCALE = 4.0
FLUX_MULTIPLE = 16
FLUX_MIN_SIDE = 64
REGION_MERGE_PX = 25
CONTEXT_FRAC = 0.5
MIN_CONTEXT_PX = 48
FEATHER_PX = 3
TONE_RING_PX = 12


def _regions(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REGION_MERGE_PX, REGION_MERGE_PX))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, kernel), connectivity=8)
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, area in stats[1:] if area > 0]


def _crop_bounds(box: tuple, shape: tuple) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad = max(MIN_CONTEXT_PX, int(CONTEXT_FRAC * max(x1 - x0, y1 - y0)))
    h, w = shape[:2]
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _model_size(width: int, height: int) -> tuple[int, int]:
    scale = min(FLUX_MAX_UPSCALE, math.sqrt(FLUX_TARGET_PIXELS / (width * height)))
    model_w = max(FLUX_MIN_SIDE, round(width * scale / FLUX_MULTIPLE) * FLUX_MULTIPLE)
    model_h = max(FLUX_MIN_SIDE, round(height * scale / FLUX_MULTIPLE) * FLUX_MULTIPLE)
    return model_w, model_h


def _generate(pipeline, prompt_embeds: torch.Tensor, crop_bgr: np.ndarray) -> np.ndarray:
    height, width = crop_bgr.shape[:2]
    model_w, model_h = _model_size(width, height)
    source = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    source = source.resize((model_w, model_h), Image.Resampling.LANCZOS)
    generator = torch.Generator(device=DEVICE).manual_seed(FLUX_SEED)
    with torch.inference_mode():
        result = pipeline(
            image=source,
            prompt_embeds=prompt_embeds.to(DEVICE),
            height=model_h,
            width=model_w,
            num_inference_steps=FLUX_STEPS,
            guidance_scale=FLUX_GUIDANCE,
            generator=generator,
        ).images[0]
    result = result.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)


def _match_tone(generated: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * TONE_RING_PX + 1, 2 * TONE_RING_PX + 1)
    )
    ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
    if not ring.any():
        return generated
    offset = np.median(original[ring].astype(np.float32), axis=0) - np.median(
        generated[ring].astype(np.float32), axis=0
    )
    return np.clip(generated.astype(np.float32) + offset, 0, 255).astype(np.uint8)


def inpaint_with_flux(img_cv: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    pipeline, prompt_embeds = get_flux_inpainter()
    if pipeline is None:
        return None

    out = img_cv.copy()
    try:
        for box in _regions(mask):
            x0, y0, x1, y1 = _crop_bounds(box, img_cv.shape)
            crop = out[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1]
            generated = _match_tone(_generate(pipeline, prompt_embeds, crop), crop, crop_mask)
            alpha = cv2.GaussianBlur(
                cv2.dilate(crop_mask, None).astype(np.float32) / 255,
                (2 * FEATHER_PX + 1, 2 * FEATHER_PX + 1),
                0,
            )[..., None]
            blended = alpha * generated.astype(np.float32) + (1 - alpha) * crop.astype(np.float32)
            out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    except Exception as error:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        logger.warning(f"  [flux] inpainting failed, using LaMa: {error}")
        return None
    return out
