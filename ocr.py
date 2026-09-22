import json
import logging
import os

import numpy as np
import requests
import torch
from PIL import Image

from image_io import encode_image, prepare_ocr_crop
from models import get_ocr_model
from settings import settings
from text_utils import clean_text

logger = logging.getLogger(__name__)

OCR_NUM_PREDICT = 256


OCR_PROMPT = "Text Recognition:"


OCR_STOP = ["```"]


def _collapse_repeats(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    prev = None
    for ln in lines:
        s = ln.strip()
        if s and s == prev:
            continue
        out.append(ln)
        if s:
            prev = s
    collapsed = "\n".join(out)

    stripped = [s for s in (ln.strip() for ln in collapsed.splitlines()) if s]
    if len(stripped) > 4:
        from collections import Counter

        most_common = Counter(stripped).most_common(1)[0][1]
        if most_common > len(stripped) * 0.5 or len(set(stripped)) <= 2:
            seen: list[str] = []
            for s in stripped:
                if s not in seen:
                    seen.append(s)
            collapsed = "\n".join(seen)
    return collapsed.strip()


_OCR_PROMPT_ECHO_MARKERS = (
    "read and return",
    "read any text",
    "return only what is written",
    "no explanation",
    "visible in this image",
    "letters, or characters",
    "line breaks and layout",
    "output only the text",
)


def _is_ocr_prompt_echo(s: str) -> bool:
    low = s.lower()
    return sum(1 for m in _OCR_PROMPT_ECHO_MARKERS if m in low) >= 2


def _clean_ocr(raw: str) -> str:
    cleaned = _collapse_repeats(clean_text(raw))
    kept = [ln for ln in cleaned.splitlines() if not _is_ocr_prompt_echo(ln)]
    cleaned = _collapse_repeats("\n".join(kept))
    if _is_ocr_prompt_echo(cleaned):
        return ""
    return cleaned


def _ocr_stream(crop: Image.Image) -> str:
    payload = {
        "model": "glm-ocr:latest",
        "prompt": OCR_PROMPT,
        "images": [encode_image(crop)],
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": OCR_NUM_PREDICT, "stop": OCR_STOP},
    }
    kept: list[str] = []
    seen: set[str] = set()
    buf = ""

    def _take(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        if s.startswith("```") or s in seen:
            return False
        seen.add(s)
        kept.append(line)
        return True

    try:
        with requests.post(settings().ollama_url, json=payload, timeout=60, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_lines():
                if not chunk:
                    continue
                event = json.loads(chunk)
                if event.get("error"):
                    raise ValueError(f"OCR request failed: {event['error']}")
                piece = event.get("response", "")
                buf += piece
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not _take(line):
                        return "\n".join(kept)
    except (requests.exceptions.RequestException, ValueError) as error:
        logger.warning(f"[ocr] Ollama inference failed: {error}")
        return ""

    _take(buf)
    return "\n".join(kept)


def _ocr_infer(crop: Image.Image) -> str | None:
    processor, model = get_ocr_model()
    if model is None:
        return None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=OCR_NUM_PREDICT)
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def _ocr_call(crop: Image.Image) -> str:
    try:
        text = _ocr_infer(crop)
    except (RuntimeError, ValueError, OSError) as error:
        logger.warning(f"[ocr] Local inference failed: {error}")
        text = None
    if text is None:
        text = _ocr_stream(crop)
    return _clean_ocr(text)


def ocr_region(img_cv: np.ndarray, x: int, y: int, w: int, h: int, idx: int, page_idx: int) -> str:
    best = ""
    for enhanced in (True, False):
        crop = prepare_ocr_crop(img_cv, x, y, w, h, enhanced=enhanced)
        if crop is None:
            return ""
        if settings().crops_dir:
            os.makedirs(settings().crops_dir, exist_ok=True)
            suffix = "" if enhanced else "_retry"
            crop.save(
                os.path.join(settings().crops_dir, f"p{page_idx:03d}_bubble_{idx:02d}{suffix}.png")
            )
        text = _ocr_call(crop)
        if len(text) > len(best):
            best = text
        if len(best) >= 3:
            return best
    return best
