"""
manga_translator.py
Kotoba — manga translator with character memory.

Automatic manga translation using local LLMs via Ollama. The key difference
from other open-source solutions is the accumulation of a character archive
and scene context across pages, which yields a more consistent translation
(correct names, grammatical gender, tone of speech).

Per-page pipeline:
  1. Bubble detection      — RT-DETRv2
  2. OCR                   — glm-ocr via Ollama
  3. Character analysis    — vision LLM + CharacterArchive
  4. Scene analysis        — vision LLM + MangaContext
  5. Speaker attribution   — vision LLM (who speaks + gender)
  6. Translation           — text LLM with context
  7. Text segmentation     — precise text-pixel masks (CTD)
  8. Render and write      — PIL, OpenCV (inpainting via LaMa)
"""

import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))

import re
import io
import json
import time
import base64
import warnings
import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

warnings.filterwarnings("ignore")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_URL = "http://localhost:11434/api/generate"
CROPS_DIR: str = "crops"

DETECT_THRESHOLD:  float = 0.5
SFX_THRESHOLD:     float = 0.3
MIN_BUBBLE_AREA:   int   = 400
MAX_FONT_SIZE:     int   = 90
INPAINT_SHRINK:    int   = 1
TRANSLATE_CHUNK_SIZE: int = 5
TRANSLATE_RETRIES: int   = 3

_LARGE_BUBBLE_PX = 80_000
_INPAINT_MAX_PIXELS = 12_000_000

READING_BAND_MIN = 40
READING_BAND_DIVISOR = 25
SFX_OVERLAP_FRAC = 0.3
CTD_MIN_COVERAGE = 0.05
INK_DARK_PERCENTILE = 25
INK_LIGHT_PERCENTILE = 75
BALANCED_LINE_TOLERANCE = 1.3
MIN_EFFECTIVE_BOX = 20
VERTICAL_TEXT_ANGLE = 45
ROTATE_SUPERSAMPLE = 2
MIN_FONT_SIZE = 2
TRANSLATE_NUM_CTX = 8192

OCR_NUM_PREDICT = 256
OCR_PROMPT = "Text Recognition:"
OCR_STOP = ["```"]

def _translation_schema(n: int) -> dict:
    return {
        "type": "array",
        "minItems": n,
        "maxItems": n,
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "translation": {"type": "string"},
            },
            "required": ["id", "translation"],
        },
    }

GLOSSARY: list[dict] = []
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

BUBBLE_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
BUBBLE_CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}

LLM_MODEL = ""

LAMA_REPOS = [
    ("deckyfx/anime-big-lama", "anime-manga-big-lama.pt"),
    ("df1412/anime-big-lama",  "anime-manga-big-lama.pt"),
]

DEFAULT_FONT = "arial.ttf"

_META_MARKERS: tuple[str, ...] = (
    "depending on", "it could be", "it can be translated",
    "note:", "please note", "keep in mind",
    "в зависимости", "можно перевести", "можно оставить",
    "это можно", "стоит отметить", "следует отметить",
    "примечание:", "обратите внимание",
)


def load_inpainting_model():
    last_error = None
    for repo_id, filename in LAMA_REPOS:
        try:
            print(f"[lama] Loading {repo_id}/{filename}...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model = torch.jit.load(model_path, map_location=DEVICE)
            model.eval()
            print(f"[lama] Loaded from {repo_id} ({DEVICE})")
            return model
        except Exception as e:
            print(f"[lama] Failed with {repo_id}: {e}")
            last_error = e
    print(f"[lama] ⚠ All sources unavailable, inpainting will fall back to cv2.inpaint")
    print(f"       Last error: {last_error}")
    return None


def load_detector():
    processor = AutoImageProcessor.from_pretrained(
        BUBBLE_MODEL_ID,
        size={"height": 960, "width": 960},
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(BUBBLE_MODEL_ID)
    model.eval()
    return processor, model


_inpaint_model = None
_inpaint_loaded = False
_detector_processor = None
_detector_model = None


def get_inpaint_model():
    global _inpaint_model, _inpaint_loaded
    if not _inpaint_loaded:
        _inpaint_model = load_inpainting_model()
        _inpaint_loaded = True
    return _inpaint_model


def get_detector():
    global _detector_processor, _detector_model
    if _detector_model is None:
        _detector_processor, _detector_model = load_detector()
    return _detector_processor, _detector_model


OCR_HF_ID = "zai-org/GLM-OCR"
_ocr_processor = None
_ocr_model = None
_ocr_loaded = False


def get_ocr_model():
    global _ocr_processor, _ocr_model, _ocr_loaded
    if not _ocr_loaded:
        _ocr_loaded = True
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            print(f"[ocr] Loading {OCR_HF_ID} (transformers)...")
            _ocr_processor = AutoProcessor.from_pretrained(OCR_HF_ID)
            _ocr_model = AutoModelForImageTextToText.from_pretrained(
                OCR_HF_ID, dtype="auto",
            ).to(DEVICE).eval()
            print(f"[ocr] Loaded ({DEVICE})")
        except Exception as e:
            print(f"[ocr] ⚠ transformers OCR unavailable ({e}); falling back to Ollama glm-ocr")
            _ocr_processor = _ocr_model = None
    return _ocr_processor, _ocr_model

_ctd_session = None
_CTD_MODEL_ID   = "mayocream/comic-text-detector-onnx"
_CTD_MODEL_FILE = "comic-text-detector.onnx"
_CTD_INPUT_SIZE = 1024
def _add_torch_cuda_dll_dir():
    if DEVICE != "cuda" or os.name != "nt":
        return
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass


def _get_ctd_session():
    global _ctd_session
    if _ctd_session is not None:
        return _ctd_session if _ctd_session is not False else None

    _add_torch_cuda_dll_dir()
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ctd] ⚠ Install: pip install onnxruntime-gpu  (или onnxruntime для CPU)")
        _ctd_session = False
        return None

    try:
        ckpt = hf_hub_download(_CTD_MODEL_ID, _CTD_MODEL_FILE)
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if DEVICE == "cuda" else ["CPUExecutionProvider"])
        sess = ort.InferenceSession(ckpt, providers=providers)
        _ctd_session = sess
        used = sess.get_providers()[0].replace("ExecutionProvider", "")
        print(f"[ctd] Loaded Comic Text Detector ({used})")
        return sess
    except Exception as e:
        print(f"[ctd] ⚠ Ошибка загрузки модели: {e}")
        _ctd_session = False
        return None


def _ctd_page_mask(img_cv: np.ndarray) -> np.ndarray | None:
    sess = _get_ctd_session()
    if sess is None:
        return None

    h, w = img_cv.shape[:2]
    s = _CTD_INPUT_SIZE

    scale = s / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_cv, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((s, s, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized.astype(np.float32) / 255.0

    inp = canvas.transpose(2, 0, 1)[np.newaxis]

    try:
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: inp})
    except Exception as e:
        print(f"[ctd] inference error: {e}")
        return None

    mask_raw = None
    for out in outputs:
        if out.ndim >= 2 and min(out.shape[-2:]) >= s // 2:
            mask_raw = out
            break
    if mask_raw is None:
        mask_raw = max(outputs, key=lambda o: o.size)

    m = mask_raw.squeeze()
    if m.ndim == 3:
        m = m[1] if m.shape[0] == 2 else m[0]

    if m.max() <= 1.0:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255

    m_crop = m[:nh, :nw]
    return cv2.resize(m_crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


class CharacterArchive:

    def __init__(self, path: str = "characters.json"):
        self.path = path
        self.characters: dict = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.characters = json.load(f)
            print(f"  Loaded archive: {len(self.characters)} character(s)")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.characters, f, ensure_ascii=False, indent=2)

    def to_prompt(self) -> str:
        if not self.characters:
            return "CHARACTER ARCHIVE: (empty, this may be the first page)"
        lines = ["CHARACTER ARCHIVE (these characters have been seen before):"]
        for cid, c in self.characters.items():
            lines.append(
                f"- ID={cid} | {c['name']} | gender={c['gender']} | "
                f"appearance: {c['appearance']} | notes: {c.get('notes', '')} | "
                f"first seen: page {c['first_seen']}"
            )
        return "\n".join(lines)

    def _unique_name(self, proposed: str, cid: str) -> str:
        existing_names = {c["name"].lower() for c in self.characters.values()}
        if proposed.lower() not in existing_names:
            return proposed

        base_words = set(re.findall(r"[a-z]+", proposed.lower()))
        id_words = re.findall(r"[a-z]+", cid.lower())
        extras = [w for w in id_words if w not in base_words and len(w) > 2]
        if extras:
            candidate = f"{proposed} ({' '.join(extras)})"
            if candidate.lower() not in existing_names:
                return candidate

        for i in range(2, 100):
            candidate = f"{proposed} #{i}"
            if candidate.lower() not in existing_names:
                return candidate
        return proposed

    def update_from_json(self, data: list, page_idx: int):
        for char in data:
            cid = char.get("id", "").strip()
            if not cid:
                continue
            if cid in self.characters:
                new_notes = char.get("notes", "")
                if new_notes and new_notes not in self.characters[cid].get("notes", ""):
                    self.characters[cid]["notes"] = (
                        self.characters[cid].get("notes", "") + "; " + new_notes
                    ).strip("; ")
            else:
                raw_name = char.get("name", cid)
                unique_name = self._unique_name(raw_name, cid)
                if unique_name != raw_name:
                    print(f"  [archive] Name '{raw_name}' already taken → '{unique_name}'")

                self.characters[cid] = {
                    "name": unique_name,
                    "gender": char.get("gender", "unknown"),
                    "appearance": char.get("appearance", ""),
                    "notes": char.get("notes", ""),
                    "first_seen": page_idx,
                }
                print(f"  [archive] New character: {unique_name}")
        self.save()

    def find_character(self, description: str) -> dict | None:
        if not description:
            return None
        desc_lower = description.lower()
        for cid, c in self.characters.items():
            if c["name"].lower() in desc_lower or cid.lower() in desc_lower:
                return c
        return None


class MangaContext:

    def __init__(self):
        self.page_summaries: list[str] = []

    def update(self, summary: str):
        if not summary or not summary.strip():
            return
        self.page_summaries.append(summary)
        if len(self.page_summaries) > 5:
            self.page_summaries.pop(0)

    def to_prompt(self) -> str:
        if not self.page_summaries:
            return "This is the first page."
        lines = "\n".join(
            f"Page {i+1}: {s}" for i, s in enumerate(self.page_summaries)
        )
        return f"STORY SO FAR:\n{lines}"


class ErrorLog:

    def __init__(self, path: str = "errors.log"):
        self.path = path
        self.entries: list[dict] = []
        if os.path.exists(self.path):
            os.remove(self.path)

    def add(self, page: int, kind: str, message: str, **details):
        entry = {
            "page": page,
            "kind": kind,
            "message": message,
            **details,
        }
        self.entries.append(entry)

    def summary(self) -> dict:
        counts: dict = {}
        for e in self.entries:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        return counts

    def save(self):
        if not self.entries:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"# Error log — {len(self.entries)} entries total\n")
            f.write(f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            by_page: dict = {}
            for e in self.entries:
                by_page.setdefault(e["page"], []).append(e)

            for page in sorted(by_page):
                f.write(f"\n{'='*60}\n")
                f.write(f"Page {page}\n")
                f.write(f"{'='*60}\n")
                for e in by_page[page]:
                    f.write(f"\n  [{e['kind']}] {e['message']}\n")
                    extras = {k: v for k, v in e.items()
                              if k not in ("page", "kind", "message")}
                    if extras:
                        for k, v in extras.items():
                            v_str = str(v)
                            if len(v_str) > 300:
                                v_str = v_str[:300] + "... [truncated]"
                            f.write(f"    {k}: {v_str}\n")


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


OLLAMA_IMAGE_MAX = 1024


def _prep_ollama_image(path: str) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > OLLAMA_IMAGE_MAX:
        img.thumbnail((OLLAMA_IMAGE_MAX, OLLAMA_IMAGE_MAX))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        raise ValueError(
            f"Cannot decode image (corrupt, unsupported, or exceeds pixel limit): {path}"
        )
    return img

def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`+", "", text)
    def _circled(m):
        n = int(m.group(1))
        if 1 <= n <= 20:
            return chr(0x2460 + n - 1)
        return m.group(0)
    text = re.sub(r"\$\s*\\?textcircled\{(\d+)\}\s*\$", _circled, text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_glossary_prompt() -> str:
    if not GLOSSARY:
        return ""
    lines = ["GLOSSARY — always use these exact translations, even if context suggests otherwise:"]
    for entry in GLOSSARY:
        src = entry.get("source", "").strip()
        tgt = entry.get("target", "").strip()
        if not src or not tgt:
            continue
        note = entry.get("note", "").strip()
        line = f"  {src} → {tgt}"
        if note:
            line += f"  ({note})"
        lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def _infer_emotion_tag(text: str) -> str:
    t = text.replace('！', '!').replace('？', '?').replace('…', '...')

    has_double_excl = bool(re.search(r'!{2,}', t))
    has_excl        = '!' in t
    has_question    = '?' in t
    has_mixed       = bool(re.search(r'[?!][!?]', t))
    has_ellipsis    = '...' in t or '…' in text
    has_repetition  = bool(re.search(r'(.{2,4})\1', text))
    all_caps        = (text.replace(' ', '').replace('\n', '').isupper()
                       and len(text.strip()) > 2)

    if has_double_excl or (has_excl and has_repetition):
        return "excited/shouting"
    if has_mixed:
        return "shocked/alarmed"
    if has_ellipsis and not has_excl:
        return "hesitant/trailing-off"
    if has_excl:
        return "emphatic"
    if has_question:
        return "questioning"
    if all_caps:
        return "loud/sfx"
    return "neutral"


DEBUG_LLM = os.environ.get("OLLAMA_DEBUG", "").strip() in ("1", "true", "yes", "on")
MASK_DEBUG_DIR: str | None = None
_OLLAMA_CALL_NUM = 0


def _log_llm_call(call_num: int, model: str, prompt: str, response: str,
                   opts: dict, has_image: bool) -> None:
    sep = "─" * 76
    print(f"\n{sep}")
    print(f"[LLM call #{call_num}] model={model} options={opts} "
          f"image={'yes' if has_image else 'no'}")
    print(f"{sep}")
    print(f"PROMPT ({len(prompt)} chars):")
    print(prompt)
    print(f"{sep}")
    print(f"RESPONSE ({len(response)} chars):")
    if response:
        print(response)
    else:
        print("(empty response)")
    print(f"{sep}\n")


LLM_FALLBACK_MODELS: list[str] = []


def _discover_fallback_models() -> list[str]:
    preferred = ["gemma4:26b", "gemma3:27b", "gemma3:12b", "llava:13b",
                 "qwen2.5-vl:7b", "minicpm-v"]
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return []
    fallbacks = [m for m in preferred
                  if m in installed and m != LLM_MODEL]
    others = sorted(installed - set(preferred) - {LLM_MODEL})
    fallbacks.extend(others)
    return fallbacks


def ollama(model_name: str, prompt: str, image_path: str = None,
           timeout: int = 800, num_predict: int = 6000,
           temperature: float = 0.1, system: str | None = None,
           fmt=None, stop: list[str] | None = None,
           num_ctx: int | None = None,
           repeat_penalty: float | None = None) -> str:
    global _OLLAMA_CALL_NUM
    import random

    base_opts = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx is not None:
        base_opts["num_ctx"] = num_ctx
    if stop:
        base_opts["stop"] = stop
    if repeat_penalty is not None:
        base_opts["repeat_penalty"] = repeat_penalty

    def _call(opts: dict, model: str) -> str:
        global _OLLAMA_CALL_NUM
        _OLLAMA_CALL_NUM += 1
        call_num = _OLLAMA_CALL_NUM
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if fmt is not None:
            payload["format"] = fmt
        if opts:
            payload["options"] = opts
        if image_path:
            payload["images"] = [_prep_ollama_image(image_path)]
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response = r.json().get("response", "").strip()
        if DEBUG_LLM:
            _log_llm_call(call_num, model, prompt, response,
                           opts, bool(image_path))
        return response

    response = _call(dict(base_opts), model_name)
    if response:
        return response

    response = _call(dict(base_opts, seed=random.randint(1, 100000)), model_name)
    if response:
        print("     [ollama-retry-ok] recovered with new seed")
        return response

    if not LLM_FALLBACK_MODELS:
        return ""

    for fb_model in LLM_FALLBACK_MODELS:
        print(f"     [ollama-fallback] trying alternate model: {fb_model}")
        try:
            response = _call(dict(base_opts), fb_model)
        except Exception as e:
            print(f"     [ollama-fallback] {fb_model} error: {e}")
            continue
        if response:
            print(f"     [ollama-retry-ok] recovered with fallback model {fb_model}")
            return response
    return ""

def parse_json_array(text: str) -> list:
    if not text:
        return []

    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    start = cleaned.find("[")
    if start < 0:
        return []

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i, ch in enumerate(cleaned[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        candidate = cleaned[start:]
    else:
        candidate = cleaned[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r",\s*([\]\}])", r"\1", candidate)
    last_close = repaired.rfind("}")
    if last_close > 0:
        repaired = repaired[: last_close + 1] + "]"
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return []

def natural_key(filename: str) -> list:
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", filename)
    ]


def detect_character_intro_page(image_path: str) -> bool:
    prompt = """Look at this manga page.

Is this a CHARACTER INTRODUCTION page — a structured VISUAL ROSTER where multiple
individual character PORTRAIT ILLUSTRATIONS are arranged in a grid or list, each
portrait labeled with the character's name?

Answer YES only if ALL of these are true:
- There are multiple separate character portrait drawings on the page
- Each portrait has the character's name printed next to or below it
- The layout is clearly a roster/lineup/gallery (not a story scene)

Answer NO if the page shows:
- Characters in a normal scene, talking, or doing actions
- A single large illustration (even with text)
- A credits, staff, or cast TEXT listing (names in columns without portrait art)
- An announcement or teaser page ("Season 3 Coming Soon", etc.)
- Primarily text with little or no character art

Answer with EXACTLY one word: YES or NO."""

    raw = ollama(LLM_MODEL, prompt, image_path, timeout=120).strip().upper()
    is_intro = raw.startswith("YES")
    print(f"  [intro detect] {raw[:30]} → {'gallery page' if is_intro else 'regular page'}")
    return is_intro


def extract_character_intros(image_path: str,
                              archive: CharacterArchive,
                              page_idx: int) -> str:
    print("  Extracting character introductions...")

    prompt = f"""This is a CHARACTER INTRODUCTION page — a roster of characters
with their names and descriptions.

{archive.to_prompt()}

For EACH character portrait on this page, extract:
1. Their REAL name as written on the page (next to portrait, in caption, in nameplate)
2. Their appearance (hair, clothing, build)
3. Any role/description text written next to them

IMPORTANT:
- The name MUST come from text printed on this page, not invented
- If a character matches one in the archive above, use the EXISTING id
- All names must be unique
- Read the text carefully — names are often written in romaji, katakana, or both

Return ONLY a JSON array:
[
  {{
    "id": "existing_or_snake_case_of_name",
    "name": "EXACT name as printed on the page",
    "gender": "male/female/unknown",
    "appearance": "hair color+style, face, clothing, build",
    "role": "their role/description as written on page (e.g. 'Princess', 'Knight')",
    "notes": "any other info from the page",
    "is_new": true/false
  }}
]"""

    raw = ollama(LLM_MODEL, prompt, image_path)
    chars = parse_json_array(raw)

    if not chars:
        print(f"  [warn] character intros did not parse: {raw[:200]}")
        return "CHARACTERS ON THIS PAGE: unknown"

    for c in chars:
        role = c.pop("role", "")
        if role:
            existing_notes = c.get("notes", "")
            c["notes"] = f"role: {role}" + (f"; {existing_notes}" if existing_notes else "")

    chars = deduplicate_characters(chars, archive)
    archive.update_from_json(chars, page_idx)

    lines = ["CHARACTERS INTRODUCED ON THIS PAGE:"]
    for c in chars:
        marker = "[NEW]" if c.get("is_new") else "[known]"
        lines.append(
            f"- {marker} {c.get('name', '?')} | {c.get('gender', '?')} | "
            f"{c.get('notes', '')}"
        )
    return "\n".join(lines)


def analyze_page_full(image_path: str, archive: CharacterArchive,
                      manga_ctx: MangaContext,
                      page_idx: int) -> tuple[str, str, str]:
    print("  Analyzing page (characters + scene)...")

    prompt = f"""You are a manga analyst tracking characters across pages AND describing scenes.

{archive.to_prompt()}

{manga_ctx.to_prompt()}

Analyze this manga page. Produce TWO blocks in this exact format:

=== CHARACTERS ===
A JSON array of all visible characters. For each one:

STEP 1 — MATCH: Compare to the archive above.
- Match by: hair color, style, face, clothing, body type
- If matched → use the EXISTING id and name from the archive
- Only create NEW if no match found

STEP 2 — NAMING:
- Prefer real names visible in the page (nameplates, captions, dialogue addressing them
  — e.g. text "Hakusen", "Sumire", "Princess" near a character IS their name)
- Otherwise invent a SPECIFIC distinctive description (NOT generic like "dark haired woman")
- All names MUST be unique across this response AND the archive

JSON schema:
[
  {{
    "id": "existing_or_new_snake_case",
    "name": "UNIQUE name",
    "gender": "male/female/unknown",
    "appearance": "hair color+style, face, clothing, build",
    "position": "top-left / center / bottom-right / etc",
    "emotion": "calm / angry / surprised / etc",
    "notes": "any relevant info",
    "is_new": true/false
  }}
]

=== SCENE ===
CONTEXT: <2-3 sentences about what is happening on this page>
SUMMARY: <one short sentence for future reference>

=== END ==="""

    raw = ollama(LLM_MODEL, prompt, image_path)

    chars_match = re.search(
        r"===\s*CHARACTERS\s*===(.*?)===\s*SCENE\s*===",
        raw, re.DOTALL | re.IGNORECASE,
    )
    chars_section = chars_match.group(1) if chars_match else raw
    chars = parse_json_array(chars_section)

    if not chars:
        print(f"  [warn] characters did not parse: {chars_section[:200]}")
        characters_context = "CHARACTERS ON THIS PAGE: unknown"
    else:
        chars = deduplicate_characters(chars, archive)
        archive.update_from_json(chars, page_idx)
        lines = ["CHARACTERS ON THIS PAGE:"]
        for c in chars:
            marker = "[NEW]" if c.get("is_new") else "[known]"
            lines.append(
                f"- {marker} {c.get('name', c.get('id', '?'))} | "
                f"{c.get('gender', '?')} | pos={c.get('position', '?')} | "
                f"emotion={c.get('emotion', '?')}"
            )
        characters_context = "\n".join(lines)

    scene_match = re.search(
        r"===\s*SCENE\s*===(.*?)(?:===\s*END\s*===|\Z)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    scene_section = scene_match.group(1) if scene_match else raw

    context_match = re.search(
        r"CONTEXT:\s*(.+?)\s*(?:SUMMARY:|===|\Z)",
        scene_section, re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r"SUMMARY:\s*(.+?)\s*(?:===|\Z)",
        scene_section, re.DOTALL | re.IGNORECASE,
    )

    page_context = (context_match.group(1).strip() if context_match
                    else scene_section[:300].strip())
    page_summary = (summary_match.group(1).strip() if summary_match
                    else page_context[:120])

    page_context = page_context[:800]
    page_summary = page_summary.split("\n")[0][:200]

    return characters_context, page_context, page_summary


def deduplicate_characters(chars: list, archive: CharacterArchive) -> list:
    result = []
    for char in chars:
        cid = char.get("id", "")
        if cid in archive.characters:
            char["is_new"] = False
            result.append(char)
            continue

        if char.get("is_new", True):
            match = find_similar_in_archive(
                char.get("appearance", "").lower(), archive
            )
            if match:
                existing_id, existing = match
                print(f"  [dedup] '{char.get('name')}' → '{existing['name']}' (id={existing_id})")
                char.update({
                    "id": existing_id,
                    "name": existing["name"],
                    "gender": existing["gender"],
                    "is_new": False,
                })
        result.append(char)
    return result


def find_similar_in_archive(appearance: str,
                             archive: CharacterArchive) -> tuple | None:
    keywords = extract_appearance_keywords(appearance)
    if not keywords:
        return None

    new_distinct = extract_distinctive_features(appearance)

    best_match, best_score = None, 0
    for cid, c in archive.characters.items():
        archived_appearance = c.get("appearance", "").lower()
        archived_keywords = extract_appearance_keywords(archived_appearance)
        common = keywords & archived_keywords

        min_size = min(len(keywords), len(archived_keywords))
        threshold = max(3, min_size // 2)

        if len(common) < threshold:
            continue

        archived_distinct = extract_distinctive_features(archived_appearance)
        if features_conflict(new_distinct, archived_distinct):
            print(f"  [dedup-skip] '{c['name']}' rejected: "
                  f"волосы {new_distinct['hair_colors']} vs {archived_distinct['hair_colors']}")
            continue

        if len(common) > best_score:
            best_score = len(common)
            best_match = (cid, c)

    return best_match


def extract_appearance_keywords(text: str) -> set:
    stopwords = {
        "a", "an", "the", "with", "and", "or", "is", "are", "has", "have",
        "wearing", "looking", "man", "woman", "person", "character", "young",
        "old", "tall", "short", "small", "large", "none", "partially", "covered",
        "hair", "eyes", "face", "expression", "style", "kimono", "shirt",
        "pants", "dress", "clothes", "clothing", "patterned",
    }
    return {
        w for w in re.findall(r"\b[a-z]+\b", text)
        if w not in stopwords and len(w) > 2
    }


HAIR_COLORS = {
    "dark", "black", "brown", "blonde", "blond", "yellow", "light",
    "white", "silver", "gray", "grey", "red", "ginger", "auburn",
    "pink", "blue", "green", "purple", "orange",
}
HAIR_STYLES = {
    "long", "short", "bob", "ponytail", "twintails", "braid", "braids",
    "curly", "straight", "wavy", "spiky", "bangs", "fringe",
}


def extract_distinctive_features(text: str) -> dict:
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    return {
        "hair_colors": words & HAIR_COLORS,
        "hair_styles": words & HAIR_STYLES,
    }


def features_conflict(a: dict, b: dict) -> bool:
    if a["hair_colors"] and b["hair_colors"]:
        if not (a["hair_colors"] & b["hair_colors"]):
            return True
    return False


def attribute_bubbles(image_path: str, bubbles: list[dict],
                      page_context: str, characters_context: str,
                      archive: CharacterArchive) -> list[dict]:
    print("  Attributing speech bubbles...")
    bubble_list = "\n".join(
        f'[{i+1}] pos=({b["x"]},{b["y"]}) size={b["width"]}x{b["height"]} '
        f'text="{b.get("text", "").replace(chr(10), " ")}"'
        for i, b in enumerate(bubbles)
    )

    prompt = f"""You are analyzing a manga page.

{archive.to_prompt()}

{characters_context}

PAGE CONTEXT:
{page_context}

SPEECH BUBBLES:
{bubble_list}

Determine the speaker for each bubble by position and context.
Use names and genders from the archive — do NOT reassign gender.

Return ONLY JSON:
[
  {{"bubble": 1, "speaker": "character name", "gender": "male/female/unknown"}},
  ...
]"""

    raw = ollama(LLM_MODEL, prompt, image_path)
    for attr in parse_json_array(raw):
        idx = attr.get("bubble", 0) - 1
        if 0 <= idx < len(bubbles):
            speaker = attr.get("speaker", "unknown")
            gender = attr.get("gender", "unknown")
            known = archive.find_character(speaker)
            if known:
                speaker = known["name"]
                gender = known["gender"]
            bubbles[idx]["speaker"] = speaker
            bubbles[idx]["gender"] = gender

    return bubbles


def translate_batch(bubbles: list[dict], page_context: str,
                    manga_ctx: MangaContext, target_lang: str = "Russian",
                    retries: int = 3,
                    errors: ErrorLog | None = None,
                    page_idx: int = 0) -> None:
    gender_hints = {
        "male": "мужской род",
        "female": "женский род",
        "unknown": "род неизвестен",
    }

    to_translate = []
    for i, b in enumerate(bubbles):
        if b.get("text", "").strip():
            to_translate.append((i, b))
        else:
            b["translation"] = ""
            if errors:
                errors.add(page_idx, "ocr_empty",
                           f"Bubble #{i+1} has no OCR text — nothing to translate",
                           bubble_idx=i+1,
                           bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})")

    if not to_translate:
        return


    missing_indices = []

    chunks = [to_translate[i:i + TRANSLATE_CHUNK_SIZE]
              for i in range(0, len(to_translate), TRANSLATE_CHUNK_SIZE)]
    if len(chunks) > 1:
        print(f"     [batch] Splitting {len(to_translate)} bubbles into "
              f"{len(chunks)} chunks of up to {TRANSLATE_CHUNK_SIZE}")
    for chunk_idx, chunk in enumerate(chunks):
        _translate_chunk(
            chunk, chunk_idx, to_translate, missing_indices,
            page_context, manga_ctx, target_lang, gender_hints,
            errors, page_idx, retries,
        )

    if missing_indices:
        print(f"     [retry] {len(missing_indices)} bubble(s) missing translation, "
              f"translating one-by-one with persistent fallback...")
        for seq in missing_indices:
            bubble_idx, b = to_translate[seq]
            translation = _translate_persistent(
                b, page_context, manga_ctx, target_lang, gender_hints
            )
            if translation:
                b["translation"] = translation
                print(f"     [retry-ok] bubble #{bubble_idx+1}: {translation[:50]}")
            else:
                b["translation"] = "[error]"
                if errors:
                    errors.add(page_idx, "translation_missing",
                               f"Bubble #{bubble_idx+1}: all persistent retries failed",
                               bubble_idx=bubble_idx+1,
                               original_text=b.get("text", ""),
                               speaker=b.get("speaker", "?"))


def _build_chunk_entries(chunk: list, gender_hints: dict) -> list[dict]:
    return [
        {
            "id": i + 1,
            "speaker": b["speaker"],
            "gender": gender_hints.get(b["gender"], "gender unknown"),
            "text": b["text"],
            "emotion": b.get("emotion_hint", "neutral"),
        }
        for i, (_, b) in enumerate(chunk)
    ]


def _translation_system(target_lang: str) -> str:
    return f"""You are a professional manga translator. Translate each bubble's text into {target_lang}.

EMOTION GUIDE — each bubble has an "emotion" field inferred from punctuation.
Match the emotional INTENSITY of the original, not just the literal meaning:
- "excited/shouting"      → energetic vocabulary, exclamation marks (e.g. «Невероятно!!» not «Я рада»)
- "shocked/alarmed"       → express clear urgency or surprise (e.g. «Что?!», «Не может быть!»)
- "hesitant/trailing-off" → trailing «...», incomplete or softened phrasing
- "emphatic"              → strong word choice, keep the exclamation mark
- "questioning"           → preserve the inquisitive intonation
- "loud/sfx"              → all-caps or emphatic equivalents for sound effects
- "neutral"               → natural conversational register

For EACH bubble translate the ENTIRE text content:
- Dialogue: match the speaker's personality and gender. Preserve both the emotional
  register and its intensity — a bubble tagged "excited/shouting" must feel
  exciting in the translation, not merely semantically correct.
- Sound effects (PANT, HUFF, TCH, AHH, EEK, etc.): produce a natural equivalent in {target_lang}.
- Announcements, credits, cast/staff lists, copyright notices: translate ALL lines
  faithfully — do NOT summarize, omit, or shorten. Keep proper names (people,
  companies, characters) unchanged. Only translate structural labels
  (STAFF→ПЕРСОНАЛ, CAST→В РОЛЯХ, etc.) if translating to {target_lang}.
- Only filter genuine OCR noise: isolated stray characters with no meaning
  (e.g. a lone "·" or "|"). Never discard recognizable words or names.

Return one object per input bubble, in the SAME order, each with "id" (matching the
input id) and "translation". The "translation" field must contain ONLY the translated
text — no explanations or commentary. Always produce a translation, even for single
words or sound effects."""


def _build_translation_prompt(entries: list[dict], page_context: str,
                              manga_ctx: MangaContext) -> str:
    inputs_json = json.dumps(entries, ensure_ascii=False, indent=2)
    glossary_section = _build_glossary_prompt()
    parts = [manga_ctx.to_prompt()]
    if page_context:
        parts.append(f"PAGE CONTEXT:\n{page_context}")
    if glossary_section:
        parts.append(glossary_section)
    parts.append(f"INPUT (JSON array of {len(entries)} bubbles):\n{inputs_json}")
    return "\n\n".join(parts)


def _call_translation_llm(prompt: str, system: str, n: int,
                          chunk_idx: int, retries: int) -> str:
    for attempt in range(retries):
        try:
            return ollama(LLM_MODEL, prompt, timeout=600, num_predict=6000,
                          system=system, fmt=_translation_schema(n),
                          num_ctx=TRANSLATE_NUM_CTX)
        except requests.exceptions.ReadTimeout:
            print(f"     [timeout] chunk {chunk_idx+1}, retry {attempt+1}/{retries}...")
    return ""


def _looks_like_commentary(t: str, src: str) -> bool:
    if any(t.lower().startswith(m) for m in _META_MARKERS):
        return True
    return len(src) <= 15 and len(t) > max(40, len(src) * 6)


def _apply_chunk_results(results: list, chunk: list, local_to_global: dict,
                         to_translate: list, missing_indices: list) -> None:
    id_to_translation = {r.get("id"): r.get("translation", "")
                          for r in results if isinstance(r, dict)}

    for local_id in range(1, len(chunk) + 1):
        global_idx = local_to_global[local_id]
        _, b = to_translate[global_idx]
        translation = id_to_translation.get(local_id, "")
        if not translation and local_id - 1 < len(results):
            r = results[local_id - 1]
            if isinstance(r, dict):
                translation = r.get("translation", "")
        if translation and translation.strip():
            t = translation.strip()
            if _looks_like_commentary(t, b.get("text", "")):
                missing_indices.append(global_idx)
            else:
                b["translation"] = t
        else:
            missing_indices.append(global_idx)


def _translate_chunk(chunk: list, chunk_idx: int, to_translate: list,
                     missing_indices: list, page_context: str,
                     manga_ctx: MangaContext, target_lang: str,
                     gender_hints: dict, errors, page_idx: int,
                     retries: int) -> None:
    chunk_offset = chunk_idx * TRANSLATE_CHUNK_SIZE
    local_to_global = {i + 1: chunk_offset + i for i in range(len(chunk))}

    entries = _build_chunk_entries(chunk, gender_hints)
    prompt = _build_translation_prompt(entries, page_context, manga_ctx)
    system = _translation_system(target_lang)
    raw = _call_translation_llm(prompt, system, len(entries), chunk_idx, retries)

    if not raw:
        missing_indices.extend(local_to_global.values())
        if errors:
            errors.add(page_idx, "timeout",
                       f"Chunk {chunk_idx+1} translation failed — all timeouts",
                       bubbles_affected=len(chunk))
        return

    results = parse_json_array(raw)
    if not results:
        if errors:
            errors.add(page_idx, "json_parse",
                       f"Chunk {chunk_idx+1}: model response did not parse",
                       raw_response=raw[:500],
                       expected_count=len(chunk))
        missing_indices.extend(local_to_global.values())
        return

    expected = len(chunk)
    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) != expected:
        print(f"     [chunk {chunk_idx+1}] count mismatch: got {len(valid)} items, "
              f"expected {expected} → one-by-one fallback")
        if errors:
            errors.add(page_idx, "count_mismatch",
                       f"Chunk {chunk_idx+1}: model returned {len(valid)} items, expected {expected}",
                       expected_count=expected,
                       got_count=len(valid))
        missing_indices.extend(local_to_global.values())
        return

    _apply_chunk_results(results, chunk, local_to_global, to_translate, missing_indices)


def _translate_persistent(bubble: dict, page_context: str,
                          manga_ctx: MangaContext, target_lang: str,
                          gender_hints: dict) -> str:
    text = bubble.get("text", "").strip()
    if not text:
        return ""

    speaker = bubble.get("speaker", "unknown")
    gender = bubble.get("gender", "unknown")
    gender_h = gender_hints.get(gender, "gender unknown")
    emotion = bubble.get("emotion_hint", "neutral")
    _gloss_p = _build_glossary_prompt()

    strategies = [
        ("ctx-speaker", lambda: (
            f"Translate this manga text to {target_lang}.\n"
            f"Speaker: {speaker} ({gender_h}). Tone/emotion: {emotion}.\n"
            + (f"{_gloss_p}\n" if _gloss_p else "")
            + f"Source: {text}\n\n"
            f"Rules: match the emotional intensity of the original "
            f"('{emotion}' means the translation must feel {emotion}). "
            f"Keep proper nouns, titles, and names unchanged. "
            f"Output ONLY the translated text. No quotes. No explanations."
        )),
        ("ui-style", lambda: (
            f"Переведи на русский (только перевод, без пояснений): {text}"
            if target_lang.lower() == "russian"
            else f"Translate to {target_lang} (output translation only): {text}"
        )),
        ("raw-imperative", lambda: (
            f"Переведи строку на {target_lang}, только результат: {text}"
            if target_lang.lower() == "russian"
            else f"{target_lang} translation only: {text}"
        )),
    ]

    refusal_markers = (
        "i cannot", "i can't", "cannot translate", "i am unable",
        "sorry,", "as an ai", "no translation",
        "не могу", "невозможно", "к сожалению",
    ) + _META_MARKERS

    for attempt, (strat_label, build_prompt) in enumerate(strategies, start=1):
        prompt = build_prompt()
        try:
            raw = ollama(LLM_MODEL, prompt, timeout=60,
                          num_predict=6000, temperature=0.3)
        except Exception as e:
            print(f"     [persistent {attempt} {strat_label}] error: {e}")
            continue

        if not raw:
            print(f"     [persistent {attempt} {strat_label}] empty response")
            continue

        cleaned = _clean_translation(raw)
        if not cleaned:
            print(f"     [persistent {attempt} {strat_label}] cleanup left nothing; "
                   f"raw[:80]={raw[:80]!r}")
            continue
        if any(m in cleaned.lower() for m in refusal_markers):
            print(f"     [persistent {attempt} {strat_label}] refusal: '{cleaned[:50]}'")
            continue
        print(f"     [persistent ok @ attempt {attempt} / {strat_label}]")
        return cleaned

    print(f"     [persistent fail] '{text[:40]}' — all 3 strategies failed")
    return ""


def _clean_translation(raw: str) -> str:
    result = raw.strip()
    if result.startswith("```"):
        result = re.sub(r"^```[a-zA-Z]*\s*", "", result)
        result = result.rstrip("`").strip()
    result = result.strip('"').strip("'").strip()

    prefixes = (
        "translation:", "перевод:", "russian:", "english:",
        "answer:", "result:", "ответ:",
    )
    for prefix in prefixes:
        if result.lower().startswith(prefix):
            result = result[len(prefix):].strip().strip('"').strip("'").strip()
            break

    lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
    if not lines:
        return ""

    _meta_intro = ("вот ", "here is", "below is", "please note", "note:")
    first_low = lines[0].lower()
    if (any(first_low.startswith(m) for m in _meta_intro)
            or any(first_low.startswith(m) for m in _META_MARKERS)):
        lines = lines[1:]
        if not lines:
            return ""

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", cleaned)

    stripped_lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    if len(stripped_lines) > 1:
        return cleaned.strip()
    first_line = stripped_lines[0] if stripped_lines else ""
    return first_line.strip('"').strip("'").strip()


def preprocess_crop(img_cv: np.ndarray, x: int, y: int,
                    w: int, h: int) -> np.ndarray:
    ih, iw = img_cv.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    crop = img_cv[y:y2, x:x2]
    h2, w2 = crop.shape[:2]
    if h2 == 0 or w2 == 0:
        return np.full((128, 128), 255, dtype=np.uint8)
    crop = cv2.resize(crop, (w2 * 3, h2 * 3), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.copyMakeBorder(gray, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=255)


def preprocess_crop_minimal(img_cv: np.ndarray, x: int, y: int,
                            w: int, h: int) -> np.ndarray:
    ih, iw = img_cv.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    crop = img_cv[y:y2, x:x2]
    h2, w2 = crop.shape[:2]
    if h2 == 0 or w2 == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8)
    crop = cv2.resize(crop, (w2 * 2, h2 * 2), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(crop, 32, 32, 32, 32, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))


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
    "read and return", "read any text", "return only what is written",
    "no explanation", "visible in this image", "letters, or characters",
    "line breaks and layout", "output only the text",
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


def _ocr_stream(crop_path: str) -> str:
    payload = {
        "model": "glm-ocr:latest",
        "prompt": OCR_PROMPT,
        "images": [image_to_base64(crop_path)],
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
        with requests.post(OLLAMA_URL, json=payload, timeout=60, stream=True) as r:
            for chunk in r.iter_lines():
                if not chunk:
                    continue
                piece = json.loads(chunk).get("response", "")
                buf += piece
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not _take(line):
                        return "\n".join(kept)
    except requests.exceptions.RequestException:
        return "\n".join(kept)

    _take(buf)
    return "\n".join(kept)


def _ocr_infer(crop_path: str) -> str | None:
    processor, model = get_ocr_model()
    if model is None:
        return None
    pil_img = Image.open(crop_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": OCR_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=OCR_NUM_PREDICT)
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def _ocr_call(crop_path: str) -> str:
    text = _ocr_infer(crop_path)
    if text is None:
        text = _ocr_stream(crop_path)
    return _clean_ocr(text)


def ocr_region(img_cv: np.ndarray, x: int, y: int, w: int, h: int,
               idx: int, page_idx: int) -> str:
    os.makedirs(CROPS_DIR, exist_ok=True)

    crop_path = os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{idx:02d}.png")
    cv2.imwrite(crop_path, preprocess_crop(img_cv, x, y, w, h))
    cleaned = _ocr_call(crop_path)
    if cleaned and len(cleaned) >= 3:
        print(f"     [OCR ✓] bubble {idx}: {cleaned[:50]!r}")
        return cleaned

    print(f"     [OCR retry] bubble {idx} — trying soft preprocessing")
    crop_path2 = os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{idx:02d}_retry.png")
    cv2.imwrite(crop_path2, preprocess_crop_minimal(img_cv, x, y, w, h))
    cleaned2 = _ocr_call(crop_path2)

    final = cleaned2 if len(cleaned2) > len(cleaned) else cleaned

    if not final:
        print(f"     [OCR ✗ EMPTY] bubble {idx} (both passes empty) → {crop_path}, {crop_path2}")
    elif len(final) < 3:
        print(f"     [OCR ? SHORT] bubble {idx}: {final!r} (after retry)")
    else:
        source = "retry" if final == cleaned2 and cleaned2 != cleaned else "primary"
        print(f"     [OCR ✓ {source}] bubble {idx}: {final[:50]!r}")

    return final


def detect_bubbles(image_pil: Image.Image, threshold: float = 0.5) -> list[dict]:
    processor, model = get_detector()
    w, h = image_pil.size
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[h, w]]), threshold=threshold,
    )[0]

    bubbles = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        bubbles.append({
            "class": BUBBLE_CLASSES.get(int(label)),
            "x": x1, "y": y1,
            "width": x2 - x1, "height": y2 - y1,
            "confidence": float(score),
            "speaker": "unknown",
            "gender": "unknown",
        })
    return bubbles


def _largest_subrect(bx: int, by: int, bw: int, bh: int,
                     ox: int, oy: int, ow: int, oh: int
                     ) -> tuple[int, int, int, int]:
    ix1 = max(bx, ox); iy1 = max(by, oy)
    ix2 = min(bx + bw, ox + ow); iy2 = min(by + bh, oy + oh)
    if ix1 >= ix2 or iy1 >= iy2:
        return bx, by, bw, bh

    options: list[tuple[int, int, int, int]] = []
    nw = ox - bx
    if nw > 0: options.append((bx, by, nw, bh))
    nx = ox + ow; nw2 = (bx + bw) - nx
    if nw2 > 0: options.append((nx, by, nw2, bh))
    nh = oy - by
    if nh > 0: options.append((bx, by, bw, nh))
    ny = oy + oh; nh2 = (by + bh) - ny
    if nh2 > 0: options.append((bx, ny, bw, nh2))
    if not options:
        return bx, by, 0, 0
    return max(options, key=lambda r: r[2] * r[3])


def _clip_overlapping_boxes(bubbles: list[dict], min_area: int = 400) -> list[dict]:
    if len(bubbles) <= 1:
        return bubbles

    order = sorted(
        range(len(bubbles)),
        key=lambda i: (bubbles[i]["width"] * bubbles[i]["height"],
                       bubbles[i]["y"], bubbles[i]["x"]),
    )

    result: list[dict] = []
    dropped = 0

    for idx in order:
        b = dict(bubbles[idx])
        bx, by, bw, bh = b["x"], b["y"], b["width"], b["height"]

        for c in result:
            bx, by, bw, bh = _largest_subrect(
                bx, by, bw, bh, c["x"], c["y"], c["width"], c["height"]
            )
            if bw == 0 or bh == 0:
                break

        if bw * bh >= min_area:
            b["x"], b["y"], b["width"], b["height"] = bx, by, bw, bh
            result.append(b)
        else:
            dropped += 1

    if dropped:
        print(f"  [clip] {dropped} bubble(s) dropped (too small after clipping)")
    clipped_count = sum(
        1 for orig, res in zip(
            sorted(bubbles, key=lambda b: b["width"] * b["height"]),
            result,
        )
        if orig["x"] != res["x"] or orig["y"] != res["y"]
           or orig["width"] != res["width"] or orig["height"] != res["height"]
    )
    if clipped_count:
        print(f"  [clip] {clipped_count} bubble(s) clipped to avoid overlap")
    return result


def _binarize_text_mask(crop_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _compute_text_masks(img_cv: np.ndarray, bubbles: list[dict],
                        page_name: str = "") -> None:
    active = [b for b in bubbles
              if b.get("translation") and b.get("_sam2_mask") is None]
    if not active:
        return

    page_mask = _ctd_page_mask(img_cv)
    if page_mask is None:
        return

    h_img, w_img = img_cv.shape[:2]
    _dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for b in active:
        x, y, bw, bh = b["x"], b["y"], b["width"], b["height"]
        x  = max(0, x);  y  = max(0, y)
        x2 = min(w_img, x + bw); y2 = min(h_img, y + bh)
        if x2 <= x or y2 <= y:
            b["_sam2_mask"] = None
            continue

        ctd_crop = page_mask[y:y2, x:x2]
        crop_bgr = img_cv[y:y2, x:x2]

        otsu_crop    = _binarize_text_mask(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        ctd_dilated  = cv2.dilate(ctd_crop, _dilate_k)
        otsu_in_zone = cv2.bitwise_and(otsu_crop, ctd_dilated)
        combined     = cv2.bitwise_or(ctd_crop, otsu_in_zone)

        m = np.zeros((h_img, w_img), dtype=np.uint8)
        m[y:y2, x:x2] = combined
        b["_sam2_mask"] = m

        pts = cv2.findNonZero(combined)
        if pts is not None and len(pts) >= 30:
            coords = pts.reshape(-1, 2).astype(np.float64)
            centered = coords - coords.mean(axis=0)
            _, sv, vt = np.linalg.svd(centered, full_matrices=False)
            elongation = sv[0] / max(sv[1], 1e-6)
            if elongation >= 2.0:
                dx, dy = vt[0]
                angle = float(np.degrees(np.arctan2(dy, dx)))
                if angle > 90:  angle -= 180
                if angle <= -90: angle += 180
                b["_text_angle"] = angle
            else:
                b["_text_angle"] = 0.0
        else:
            b["_text_angle"] = 0.0


        if MASK_DEBUG_DIR:
            idx = b.get("idx", id(b))
            overlay = crop_bgr.copy()
            overlay[combined > 0] = (0, 200, 0)
            vis = cv2.addWeighted(crop_bgr, 0.5, overlay, 0.5, 0)
            prefix = f"{page_name}_" if page_name else ""
            dbg_path = os.path.join(MASK_DEBUG_DIR, f"{prefix}b{idx:03d}.png")
            cv2.imwrite(dbg_path, vis)
            print(f"  [ctd-dbg] → {dbg_path}")


def detect_text_color(img_cv: np.ndarray, bubble: dict,
                      default: tuple = (0, 0, 0)) -> tuple:
    ih, iw = img_cv.shape[:2]
    x, y = max(0, bubble["x"]), max(0, bubble["y"])
    x2 = min(iw, bubble["x"] + bubble["width"])
    y2 = min(ih, bubble["y"] + bubble["height"])
    w, h = x2 - x, y2 - y
    if w <= 0 or h <= 0:
        return default
    crop = img_cv[y:y2, x:x2]
    if crop.size == 0:
        return default

    sam2_mask = bubble.get("_sam2_mask")
    if sam2_mask is not None:
        crop_mask = sam2_mask[y:y2, x:x2]
        text_pixels = crop[crop_mask > 0]
        if len(text_pixels) >= 10:
            bg_pix = crop[crop_mask == 0]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            bg_gray = float(np.mean(bg_pix)) if len(bg_pix) > 0 else float(np.mean(crop_gray))
            gray_vals = np.mean(text_pixels.astype(np.float32), axis=1)
            if bg_gray >= 128:
                thr = min(float(np.percentile(gray_vals, INK_DARK_PERCENTILE)), 110)
                ink = text_pixels[gray_vals <= thr]
            else:
                thr = max(float(np.percentile(gray_vals, INK_LIGHT_PERCENTILE)), 150)
                ink = text_pixels[gray_vals >= thr]
            if len(ink) >= 5:
                bv, gv, rv = np.median(ink, axis=0).astype(int)
                return (int(rv), int(gv), int(bv))
            bv, gv, rv = np.median(text_pixels, axis=0).astype(int)
            return (int(rv), int(gv), int(bv))

    bubble_area = bubble.get("width", 0) * bubble.get("height", 0)
    if bubble.get("class") == "text_free" and bubble_area < _LARGE_BUBBLE_PX:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mask = (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 80)
        sat_frac = np.mean(sat_mask)
        sat_pixels = crop[sat_mask]
        if sat_frac < 0.4 and len(sat_pixels) >= 30:
            bv, gv, rv = np.median(sat_pixels, axis=0).astype(int)
            rv, gv, bv = int(rv), int(gv), int(bv)
            if max(abs(rv - gv), abs(gv - bv), abs(rv - bv)) > 40:
                return (rv, gv, bv)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _binarize_text_mask(gray)
    text_pixels = crop[mask > 0]
    if len(text_pixels) < 10:
        return default
    bv, gv, rv = np.median(text_pixels, axis=0).astype(int)
    bg_pixels = crop[mask == 0]
    if len(bg_pixels) > 0:
        bb, bg_g, br = np.median(bg_pixels, axis=0).astype(int)
        if max(abs(rv - br), abs(gv - bg_g), abs(bv - bb)) < 40:
            return default
    return (int(rv), int(gv), int(bv))


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = mask
    inv = cv2.bitwise_not(padded)
    cv2.floodFill(inv, None, (0, 0), 0)
    holes = inv[1:h+1, 1:w+1]
    return cv2.bitwise_or(mask, holes)


def build_inpaint_mask(img_cv: np.ndarray, bubbles: list[dict],
                        shrink: int = 1) -> np.ndarray:
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    connect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilate_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closing_k = cv2.getStructuringElement(cv2.MORPH_RECT,    (17, 3))

    for b in bubbles:
        if not b.get("translation"):
            continue
        x, y, bw, bh = b["x"], b["y"], b["width"], b["height"]
        x0 = max(0, x + shrink)
        y0 = max(0, y + shrink)
        x1 = min(w, x + bw - shrink)
        y1 = min(h, y + bh - shrink)
        if x1 <= x0 or y1 <= y0:
            continue

        sam2_mask = b.get("_sam2_mask")
        if sam2_mask is not None:
            crop_m = sam2_mask[y0:y1, x0:x1]
            coverage = np.count_nonzero(crop_m) / max(crop_m.size, 1)
            if coverage >= CTD_MIN_COVERAGE and crop_m.any():
                closed = cv2.morphologyEx(crop_m, cv2.MORPH_CLOSE, closing_k)
                glyph  = cv2.dilate(closed, inpaint_k)

                rows_with_text = np.any(crop_m > 0, axis=1)
                if rows_with_text.any():
                    r_top = int(np.argmax(rows_with_text))
                    r_bot = int(len(rows_with_text) - 1 - np.argmax(rows_with_text[::-1]))
                    sub_gray = img_gray[y0 + r_top: y0 + r_bot + 1, x0:x1]
                    otsu_sub = _binarize_text_mask(sub_gray)
                    otsu_sub = cv2.dilate(
                        otsu_sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    )
                    otsu_full = np.zeros(crop_m.shape, dtype=np.uint8)
                    otsu_full[r_top: r_bot + 1, :] = otsu_sub
                    glyph = cv2.bitwise_or(glyph, otsu_full)

                glyph  = _fill_mask_holes(glyph)
                mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], glyph)
                continue

        if bw * bh < _LARGE_BUBBLE_PX:
            mask[y0:y1, x0:x1] = 255
            continue

        crop_gray = img_gray[y0:y1, x0:x1]
        glyph = _binarize_text_mask(crop_gray)
        if np.count_nonzero(glyph) > 0.75 * glyph.size:
            mask[y0:y1, x0:x1] = 255
        else:
            glyph = cv2.dilate(glyph, connect_k)
            contours, _ = cv2.findContours(glyph, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(glyph, contours, -1, 255, cv2.FILLED)
            glyph = cv2.dilate(glyph, dilate_k)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], glyph)

    return mask


def _pad_to_multiple_of_8(img_np: np.ndarray) -> tuple[np.ndarray, tuple]:
    h, w = img_np.shape[:2]
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    if new_h == h and new_w == w:
        return img_np, (h, w)

    if img_np.ndim == 3:
        padded = np.full((new_h, new_w, img_np.shape[2]), 255, dtype=img_np.dtype)
    else:
        padded = np.zeros((new_h, new_w), dtype=img_np.dtype)
    padded[:h, :w] = img_np
    return padded, (h, w)


def inpaint_page(img_cv: np.ndarray, bubbles: list[dict]) -> np.ndarray:
    mask_np = build_inpaint_mask(img_cv, bubbles, shrink=INPAINT_SHRINK)

    if not mask_np.any():
        return img_cv.copy()

    model = get_inpaint_model()
    if model is None:
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)

    h_px, w_px = img_cv.shape[:2]
    if h_px * w_px > _INPAINT_MAX_PIXELS:
        print(f"  [inpaint] page {w_px}x{h_px} exceeds LaMa pixel cap → cv2.inpaint")
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)

    try:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_padded, (orig_h, orig_w) = _pad_to_multiple_of_8(img_rgb)
        mask_padded, _ = _pad_to_multiple_of_8(mask_np)

        img_tensor = torch.from_numpy(img_padded).float().div(255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(mask_padded).float().div(255.0)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            result_tensor = model(img_tensor, mask_tensor)

        result_np = result_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result_np = (result_np * 255).astype(np.uint8)
        result_np = result_np[:orig_h, :orig_w]
        result_cv = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        out = img_cv.copy()
        mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR) // 255
        out = out * (1 - mask_3ch) + result_cv * mask_3ch
        return out.astype(np.uint8)

    except Exception as e:
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"  [inpaint ⚠] LaMa failed ({e}), falling back to cv2.inpaint")
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)


def fit_text_in_box(draw, text: str, box_w: int, box_h: int,
                    font_path: str | None = None) -> tuple:
    if font_path is None:
        font_path = DEFAULT_FONT
    padding = 6
    usable_w = box_w - padding * 2
    usable_h = box_h - padding * 2

    def try_load_font(size: int):
        if font_path is None:
            return None
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            return None

    def text_width(s: str, font) -> int:
        return draw.textbbox((0, 0), s, font=font)[2]

    def line_height(font) -> int:
        ascent, descent = font.getmetrics()
        return ascent + descent

    def hyphenate(word: str, font, max_w: int) -> list[str]:
        if text_width(word, font) <= max_w:
            return [word]
        chunks = []
        cur = ""
        for ch in word:
            test = cur + ch + "-"
            if text_width(test, font) <= max_w:
                cur += ch
            else:
                if cur:
                    chunks.append(cur + "-")
                    cur = ch
                else:
                    chunks.append(ch)
                    cur = ""
        if cur:
            chunks.append(cur)
        return chunks

    def wrap_greedy(words: list[str], font,
                     allow_hyphenation: bool = False) -> list[str] | None:
        lines = []
        line = ""
        for word in words:
            test = (line + " " + word).strip()
            if text_width(test, font) <= usable_w:
                line = test
            else:
                if line:
                    lines.append(line)
                    line = ""
                if text_width(word, font) > usable_w:
                    if not allow_hyphenation:
                        return None
                    chunks = hyphenate(word, font, usable_w)
                    if chunks:
                        lines.extend(chunks[:-1])
                        line = chunks[-1]
                else:
                    line = word
        if line:
            lines.append(line)
        return lines or None

    def wrap_balanced(words: list[str], font, target_lines: int) -> list[str] | None:
        if target_lines < 1 or target_lines > len(words):
            return None
        total_chars = sum(len(w) for w in words) + (len(words) - 1)
        target_line_chars = total_chars / target_lines

        lines = []
        line = ""
        for word in words:
            test = (line + " " + word).strip()
            if not line:
                line = word
            elif (len(test) <= target_line_chars * BALANCED_LINE_TOLERANCE
                  and text_width(test, font) <= usable_w):
                line = test
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        if len(lines) != target_lines:
            return None
        for ln in lines:
            if text_width(ln, font) > usable_w:
                return None
        return lines

    def measure(lines: list[str], font) -> tuple[int, int, int] | None:
        if not lines:
            return None
        lh = line_height(font)
        heights = [lh] * len(lines)
        spacing = max(2, font.size // 7)
        total_h = sum(heights) + spacing * (len(lines) - 1)
        max_w = max(text_width(ln, font) for ln in lines)
        if total_h <= usable_h and max_w <= usable_w:
            return total_h, max_w, spacing
        return None

    def score(lines: list[str], total_h: int, max_w: int, font) -> float:
        fill_h = total_h / usable_h
        fill_w = max_w / usable_w
        widths = [text_width(ln, font) for ln in lines]
        if widths:
            avg_w = sum(widths) / len(widths)
            uniformity = avg_w / max(widths) if max(widths) else 1.0
        else:
            uniformity = 1.0
        return (fill_h + fill_w) * 0.5 + uniformity * 0.2

    def best_wrap_for_size(font, allow_hyphenation: bool = False
                             ) -> tuple[list[str], int, int, int] | None:
        paragraphs = text.split("\n")

        if len(paragraphs) > 1:
            all_lines: list[str] = []
            for para in paragraphs:
                para_words = para.split()
                if not para_words:
                    all_lines.append("")
                    continue
                wrapped = wrap_greedy(para_words, font,
                                      allow_hyphenation=allow_hyphenation)
                if wrapped is None:
                    return None
                all_lines.extend(wrapped)
            m = measure(all_lines, font)
            return (all_lines, *m) if m is not None else None

        words = text.split() or [text]
        candidates: list[tuple[list[str], int, int, int]] = []

        greedy = wrap_greedy(words, font, allow_hyphenation=allow_hyphenation)
        if greedy:
            m = measure(greedy, font)
            if m:
                candidates.append((greedy, *m))

        for n_lines in range(1, min(5, len(words) + 1)):
            balanced = wrap_balanced(words, font, n_lines)
            if balanced:
                m = measure(balanced, font)
                if m:
                    candidates.append((balanced, *m))

        if not candidates:
            return None
        return max(candidates, key=lambda c: score(c[0], c[1], c[2], font))

    if try_load_font(10) is None:
        font = ImageFont.load_default()
        return font, [text], [12], 2

    def _search(allow_hyphenation: bool):
        lo, hi = MIN_FONT_SIZE, MAX_FONT_SIZE
        best_local = None
        while lo <= hi:
            mid = (lo + hi) // 2
            font = try_load_font(mid)
            result = best_wrap_for_size(font, allow_hyphenation=allow_hyphenation)
            if result is not None:
                lines, total_h, max_w, spacing = result
                lh = line_height(font)
                heights = [lh] * len(lines)
                best_local = (font, lines, heights, spacing)
                lo = mid + 1
            else:
                hi = mid - 1
        return best_local

    best = _search(allow_hyphenation=False)
    if best is None:
        best = _search(allow_hyphenation=True)

    if best is not None:
        return best

    font = try_load_font(MIN_FONT_SIZE) or ImageFont.load_default()
    words = text.split() or [text]
    lines = wrap_greedy(words, font, allow_hyphenation=True) or [text]
    try:
        lh = line_height(font)
    except Exception:
        lh = 8
    return font, lines, [lh] * len(lines), 2


def _effective_box(bx: int, by: int, bw: int, bh: int,
                   smaller_bubbles: list[dict]) -> tuple[int, int, int, int]:
    ex, ey, ew, eh = bx, by, bw, bh
    for ob in smaller_bubbles:
        ex, ey, ew, eh = _largest_subrect(
            ex, ey, ew, eh, ob["x"], ob["y"], ob["width"], ob["height"]
        )
        if ew == 0 or eh == 0:
            break
    return ex, ey, ew, eh


def _find_font_variant(base_path: str | None, bold: bool, italic: bool) -> str | None:
    if not base_path:
        return None
    stem, ext = os.path.splitext(base_path)
    dir_ = os.path.dirname(base_path)
    name_only = os.path.basename(stem)
    base_no_ext = os.path.join(dir_, name_only) if dir_ else name_only
    if bold and italic:
        suffixes = ["bi", "BI", "BoldItalic", "Bold-Italic", "bolditalic"]
    elif bold:
        suffixes = ["bd", "BD", "b", "Bold", "-Bold", "bold", "B"]
    else:
        suffixes = ["i", "I", "Italic", "-Italic", "italic"]
    for s in suffixes:
        for path in (f"{stem}{s}{ext}", f"{base_no_ext}{s}{ext}"):
            try:
                ImageFont.truetype(path, 12)
                return path
            except OSError:
                pass
    return None


def _render_text_block(text: str, box_w: int, box_h: int,
                        color: tuple, font_path: str | None = None,
                        font_size_override: int | None = None,
                        text_align: str = "center",
                        bold: bool = False, italic: bool = False,
                        underline: bool = False,
                        outline_color: tuple | None = None,
                        outline_width: int = 0) -> Image.Image:
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    eff_font = font_path
    sim_bold = False
    if bold or italic:
        variant = _find_font_variant(font_path or DEFAULT_FONT, bold, italic)
        if variant:
            eff_font = variant
        elif bold:
            sim_bold = True

    if font_size_override:
        font, lines, line_heights, spacing = _fit_at_exact_size(
            draw, text, box_w, box_h, eff_font, int(font_size_override)
        )
    else:
        font, lines, line_heights, spacing = fit_text_in_box(
            draw, text, box_w, box_h, eff_font
        )

    padding = 6
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    text_y = padding + max(0, (box_h - padding * 2 - total_h) // 2)

    rgba_color = (color[0], color[1], color[2], 255)

    if outline_color and outline_width > 0:
        stroke_fill: tuple | None = (outline_color[0], outline_color[1], outline_color[2], 255)
        stroke_w = outline_width
    elif sim_bold:
        stroke_fill = rgba_color
        stroke_w = 1
    else:
        stroke_fill = None
        stroke_w = 0

    for j, line in enumerate(lines):
        bb = draw.textbbox((0, 0), line, font=font)
        line_w = bb[2] - bb[0]

        if text_align == "left":
            text_x = padding
        elif text_align == "right":
            text_x = max(padding, box_w - padding - line_w)
        else:
            text_x = padding + max(0, (box_w - padding * 2 - line_w) // 2)
            text_x = min(text_x, box_w - padding - line_w)

        draw.text((text_x, text_y), line, fill=rgba_color, font=font,
                  stroke_fill=stroke_fill, stroke_width=stroke_w)

        if underline and line.strip():
            ascent, _ = font.getmetrics()
            uy = text_y + ascent + 1
            thickness = max(1, font.size // 14)
            draw.rectangle([text_x, uy, text_x + line_w, uy + thickness], fill=rgba_color)

        text_y += line_heights[j] + spacing

    return img


def _fit_at_exact_size(draw, text: str, box_w: int, box_h: int,
                        font_path: str | None, size: int) -> tuple:
    if font_path is None:
        font_path = DEFAULT_FONT
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        try:
            font = ImageFont.truetype(DEFAULT_FONT, size)
        except OSError:
            font = ImageFont.load_default()

    padding = 6
    usable_w = box_w - padding * 2

    def w_of(s):
        return draw.textbbox((0, 0), s, font=font)[2]

    def _wrap_para(para_words: list[str]) -> list[str]:
        result: list[str] = []
        cur = ""
        for word in para_words:
            test = (cur + " " + word).strip()
            if w_of(test) <= usable_w:
                cur = test
            else:
                if cur:
                    result.append(cur)
                if w_of(word) > usable_w:
                    buf = ""
                    for ch in word:
                        if w_of(buf + ch + "-") <= usable_w:
                            buf += ch
                        else:
                            if buf:
                                result.append(buf + "-")
                            buf = ch
                    cur = buf
                else:
                    cur = word
        if cur:
            result.append(cur)
        return result

    lines: list[str] = []
    for para in text.split("\n"):
        para_words = para.split()
        if not para_words:
            lines.append("")
        else:
            lines.extend(_wrap_para(para_words))

    ascent, descent = font.getmetrics()
    lh = ascent + descent
    heights = [lh] * len(lines)
    spacing = max(2, size // 7)
    return font, lines, heights, spacing


def _contains_cjk(text: str) -> bool:
    return any('぀' <= c <= 'ヿ' or '一' <= c <= '鿿' for c in text)


def _extract_bubble_style(b: dict) -> dict:
    raw_oc = b.get("outline_color")
    return dict(
        font_path=b.get("font_path"),
        font_size_override=b.get("font_size"),
        text_align=b.get("text_align", "center") or "center",
        bold=bool(b.get("bold", False)),
        italic=bool(b.get("italic", False)),
        underline=bool(b.get("underline", False)),
        outline_color=tuple(raw_oc) if raw_oc else None,
        outline_width=int(b.get("outline_width", 0) or 0),
    )


def _render_bubble_block(translation: str, color: tuple, style: dict,
                         box: tuple, eff: tuple, text_angle: float) -> Image.Image:
    bx, by, bw, bh = box
    ex, ey, ew, eh = eff
    off_x, off_y = ex - bx, ey - by

    if abs(text_angle) > VERTICAL_TEXT_ANGLE:
        rot_deg = -90 if text_angle > 0 else 90
        block = _render_text_block(translation, bh, bw, color, **style)
        block = block.rotate(rot_deg, expand=True,
                             resample=Image.Resampling.BICUBIC)
        bw_r, bh_r = block.size
        cx_r = max(0, (bw_r - bw) // 2)
        cy_r = max(0, (bh_r - bh) // 2)
        return block.crop((cx_r + off_x, cy_r + off_y,
                           cx_r + off_x + ew, cy_r + off_y + eh))

    if abs(text_angle) > 1:
        ss = ROTATE_SUPERSAMPLE
        big = _render_text_block(translation, bw * ss, bh * ss, color, **style)
        big = big.rotate(-text_angle, expand=True,
                         resample=Image.Resampling.BICUBIC)
        bw_big, bh_big = big.size
        cx = max(0, (bw_big - bw * ss) // 2)
        cy = max(0, (bh_big - bh * ss) // 2)
        big = big.crop((cx, cy, cx + bw * ss, cy + bh * ss))
        block_full = big.resize((bw, bh), Image.Resampling.LANCZOS)
        return block_full.crop((off_x, off_y, off_x + ew, off_y + eh))

    block = _render_text_block(translation, bw, bh, color, **style)
    if off_x or off_y or ew != bw or eh != bh:
        block = block.crop((off_x, off_y, off_x + ew, off_y + eh))
    return block


def _draw_debug_overlay(pil: Image.Image, bubbles: list[dict]) -> None:
    draw = ImageDraw.Draw(pil)
    try:
        font_small = ImageFont.truetype(DEFAULT_FONT, 14)
    except OSError:
        font_small = ImageFont.load_default()

    for i, b in enumerate(bubbles):
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        if not b.get("text"):
            color = (255, 0, 0)
        elif not b.get("translation"):
            color = (255, 140, 0)
        elif b["class"] == "text_bubble":
            color = (0, 200, 0)
        else:
            color = (0, 150, 255)

        draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=2)
        draw.rectangle([(x, y-22), (x+18, y)], fill=color)
        draw.text((x+3, y-20), str(i+1), fill=(0, 0, 0), font=font_small)


def draw_results(img_cv: np.ndarray, bubbles: list[dict],
                 debug: bool = False, page_name: str = "") -> np.ndarray:
    print("  Segmenting text regions (CTD)...")
    user_angles = {id(b): b["_text_angle"] for b in bubbles if "_text_angle" in b}
    _compute_text_masks(img_cv, bubbles, page_name=page_name)
    for b in bubbles:
        if id(b) in user_angles:
            b["_text_angle"] = user_angles[id(b)]

    for b in bubbles:
        if b.get("translation") and b.get("_text_color") is None:
            b["_text_color"] = detect_text_color(img_cv, b)

    print("  Inpainting original text (LaMa)...")
    inpainted = inpaint_page(img_cv, bubbles)
    pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)).convert("RGBA")

    render_order = sorted(
        bubbles,
        key=lambda b: b.get("width", 0) * b.get("height", 0),
        reverse=True,
    )
    for i, b in enumerate(render_order):
        translation = b.get("translation", "")
        if not translation:
            continue
        bx, by, bw, bh = b["x"], b["y"], b["width"], b["height"]
        color = b.get("_text_color", (0, 0, 0))

        smaller = [s for s in render_order[i + 1:] if s.get("translation")]
        ex, ey, ew, eh = _effective_box(bx, by, bw, bh, smaller)
        if ew < MIN_EFFECTIVE_BOX or eh < MIN_EFFECTIVE_BOX:
            continue

        text_angle = b.get("_text_angle", 0.0)
        if abs(text_angle) > VERTICAL_TEXT_ANGLE and _contains_cjk(b.get("text", "")):
            print(f"  [jp→ru] bubble {i+1} angle={text_angle:.0f}° japanese → render horizontal")
            text_angle = 0.0

        block = _render_bubble_block(
            translation, color, _extract_bubble_style(b),
            (bx, by, bw, bh), (ex, ey, ew, eh), text_angle,
        )
        pil.paste(block, (ex, ey), block)

    if debug:
        _draw_debug_overlay(pil, bubbles)

    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def reading_order(bubble: dict, band: int = 150) -> tuple:
    return (bubble["y"] // max(1, band), -bubble["x"])


def _is_covered_by(bubble: dict, others: list[dict], min_frac: float) -> bool:
    bx, by, bw, bh = bubble["x"], bubble["y"], bubble["width"], bubble["height"]
    for e in others:
        ix = max(0, min(bx + bw, e["x"] + e["width"]) - max(bx, e["x"]))
        iy = max(0, min(by + bh, e["y"] + e["height"]) - max(by, e["y"]))
        if ix * iy > min_frac * bw * bh:
            return True
    return False


def _detect_text_bubbles(img_cv: np.ndarray, image_pil: Image.Image) -> list[dict]:
    low_threshold = min(DETECT_THRESHOLD, SFX_THRESHOLD)
    detections = detect_bubbles(image_pil, threshold=low_threshold)

    bubbles = [b for b in detections if b["confidence"] >= DETECT_THRESHOLD]
    for b in detections:
        if b["confidence"] >= DETECT_THRESHOLD or b["class"] != "text_free":
            continue
        if not _is_covered_by(b, bubbles, SFX_OVERLAP_FRAC):
            bubbles.append(b)
            print(f"  [sfx+] text_free @ ({b['x']},{b['y']}) score={b['confidence']:.2f}")

    bubbles = _clip_overlapping_boxes(bubbles, min_area=MIN_BUBBLE_AREA)
    band = max(READING_BAND_MIN, img_cv.shape[0] // READING_BAND_DIVISOR)
    return sorted(
        [b for b in bubbles if b["class"] in ("text_bubble", "text_free")],
        key=lambda b: reading_order(b, band),
    )


def _handle_intro_page(image_path: str, archive: CharacterArchive,
                       manga_ctx: MangaContext, page_idx: int,
                       output_path: str, img_cv: np.ndarray) -> bool:
    print("\n── Checking: character gallery? ──")
    if not detect_character_intro_page(image_path):
        return False

    print("\n── Extracting introductions ──")
    characters_context = extract_character_intros(image_path, archive, page_idx)
    print(characters_context)
    if characters_context == "CHARACTERS ON THIS PAGE: unknown":
        print("  [intro detect] extraction failed → treating as regular page")
        return False

    manga_ctx.update("Character introduction page.")
    cv2.imwrite(output_path, img_cv)
    print(f"\nSaved (unchanged): {output_path}")
    print("  ⓘ character introduction page — no translation needed")
    return True


def _ocr_bubbles(img_cv: np.ndarray, text_bubbles: list[dict],
                 page_idx: int, errors: ErrorLog | None) -> None:
    print("\n── OCR ──")
    for i, b in enumerate(text_bubbles):
        b["text"] = ocr_region(img_cv, b["x"], b["y"], b["width"], b["height"],
                               idx=i + 1, page_idx=page_idx)
        b["emotion_hint"] = _infer_emotion_tag(b.get("text", ""))
        if not b["text"] and errors:
            errors.add(page_idx, "ocr_empty",
                       f"Bubble #{i+1}: OCR returned no text after two passes",
                       bubble_idx=i+1,
                       bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})",
                       crop=os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{i+1:02d}.png"))


def _analyze_and_attribute(image_path: str, text_bubbles: list[dict],
                           archive: CharacterArchive, manga_ctx: MangaContext,
                           page_idx: int, fast_mode: bool,
                           errors: ErrorLog | None, stage) -> tuple[str, str, str]:
    if fast_mode:
        print("\n[fast mode] skipping analyze + attribute stages")
        for b in text_bubbles:
            b["speaker"] = "unknown"
            b["gender"] = "unknown"
        return "CHARACTERS ON THIS PAGE: unknown", "", ""

    print("\n── Page analysis ──")
    stage("stage_analyze")
    characters_context, page_context, page_summary = analyze_page_full(
        image_path, archive, manga_ctx, page_idx
    )
    print(characters_context)
    print(f"  context: {page_context}")
    print(f"  summary: {page_summary}")
    if characters_context == "CHARACTERS ON THIS PAGE: unknown" and errors:
        errors.add(page_idx, "character_parse_failed",
                   "Character analysis produced no result — JSON did not parse",
                   raw_response_snippet=characters_context)

    print("\n── Attribution ──")
    stage("stage_attribute")
    attribute_bubbles(image_path, text_bubbles, page_context,
                      characters_context, archive)
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('speaker', '?')} ({b.get('gender', '?')}): "
              f"{b.get('text', '')[:40]}")
        if b.get("text") and b.get("speaker") == "unknown" and errors:
            errors.add(page_idx, "speaker_unknown",
                       f"Bubble #{i+1}: could not determine speaker",
                       bubble_idx=i+1,
                       text=b.get("text", "")[:100])

    return characters_context, page_context, page_summary


def _print_page_stats(text_bubbles: list[dict], output_path: str) -> None:
    empty_ocr = sum(1 for b in text_bubbles if not b.get("text"))
    no_translation = sum(1 for b in text_bubbles
                          if b.get("text") and not b.get("translation"))
    err_translation = sum(1 for b in text_bubbles
                          if b.get("translation") == "[error]")
    ok = len(text_bubbles) - empty_ocr - no_translation - err_translation
    print(f"\nSaved: {output_path}")
    print(f"  ✓ translated: {ok} | ⚠ no translation: {no_translation} | "
          f"✗ OCR empty: {empty_ocr} | ✗ translation error: {err_translation}")


def process_page(image_path: str, page_idx: int,
                 manga_ctx: MangaContext, archive: CharacterArchive,
                 output_path: str, target_lang: str = "Russian",
                 debug: bool = False,
                 fast_mode: bool = False,
                 errors: ErrorLog | None = None,
                 on_stage=None):
    def stage(key):
        if on_stage:
            try:
                on_stage(page_idx, key)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"Page {page_idx}: {image_path}")
    if fast_mode:
        print("[fast mode] skipping page analysis and speaker attribution")
    print(f"{'='*60}")

    stage("stage_detect")
    img_cv = read_image(image_path)
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    text_bubbles = _detect_text_bubbles(img_cv, image_pil)
    print(f"Bubbles: {len(text_bubbles)}")

    if not text_bubbles and errors:
        errors.add(page_idx, "no_bubbles",
                   "No text bubbles found on this page",
                   image=image_path)

    if not fast_mode and len(text_bubbles) <= 2:
        if _handle_intro_page(image_path, archive, manga_ctx, page_idx,
                              output_path, img_cv):
            return text_bubbles

    stage("stage_ocr")
    _ocr_bubbles(img_cv, text_bubbles, page_idx, errors)

    _, page_context, page_summary = _analyze_and_attribute(
        image_path, text_bubbles, archive, manga_ctx, page_idx,
        fast_mode, errors, stage,
    )

    print("\n── Translation ──")
    stage("stage_translate")
    translate_batch(text_bubbles, page_context, manga_ctx, target_lang,
                    retries=TRANSLATE_RETRIES, errors=errors, page_idx=page_idx)
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('text', '')[:25]} → {b.get('translation', '')[:40]}")

    manga_ctx.update(page_summary)

    stage("stage_inpaint")
    annotated = draw_results(img_cv, text_bubbles, debug=debug,
                             page_name=os.path.splitext(os.path.basename(output_path))[0])
    cv2.imwrite(output_path, annotated)

    _print_page_stats(text_bubbles, output_path)
    return text_bubbles


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}с"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m}м {s}с"
    return f"{m}м {s}с"


def process_directory(input_dir: str, output_dir: str = "results",
                      target_lang: str = "Russian",
                      font_path: str = "arial.ttf",
                      debug: bool = False,
                      fast_mode: bool = False,
                      error_log_path: str = "errors.log",
                      llm_model: str | None = None,
                      ollama_url: str | None = None,
                      detect_threshold: float | None = None,
                      sfx_threshold: float | None = None,
                      min_bubble_area: int | None = None,
                      max_font_size: int | None = None,
                      inpaint_shrink: int | None = None,
                      chunk_size: int | None = None,
                      translate_retries: int | None = None,
                      on_page_done=None,
                      on_start=None,
                      on_finish=None,
                      on_stage=None,
                      cancel_event=None):
    global LLM_MODEL, OLLAMA_URL
    global DETECT_THRESHOLD, SFX_THRESHOLD, MIN_BUBBLE_AREA
    global MAX_FONT_SIZE, INPAINT_SHRINK, TRANSLATE_CHUNK_SIZE, TRANSLATE_RETRIES

    if llm_model:
        LLM_MODEL = llm_model
        print(f"[llm] Using model: {llm_model}")
    if not LLM_MODEL:
        raise ValueError(
            "No LLM model selected. Pass llm_model=... (e.g. a multimodal "
            "Ollama model like 'gemma3:27b') — there is no default."
        )
    if ollama_url:
        OLLAMA_URL = ollama_url
        print(f"[llm] Ollama URL: {ollama_url}")
    if detect_threshold is not None:
        DETECT_THRESHOLD = float(detect_threshold)
    if sfx_threshold is not None:
        SFX_THRESHOLD = float(sfx_threshold)
    if min_bubble_area is not None:
        MIN_BUBBLE_AREA = int(min_bubble_area)
    if max_font_size is not None:
        MAX_FONT_SIZE = int(max_font_size)
    if inpaint_shrink is not None:
        INPAINT_SHRINK = int(inpaint_shrink)
    if chunk_size is not None:
        TRANSLATE_CHUNK_SIZE = int(chunk_size)
    if translate_retries is not None:
        TRANSLATE_RETRIES = int(translate_retries)
    def _resolve_font(requested: str) -> str | None:
        try:
            ImageFont.truetype(requested, 12)
            return requested
        except OSError:
            pass
        candidates = [
            "Ace 2.0 BB Cyr.ttf", "animeace2_bld.ttf", "animeace2_reg.ttf",
            "CCWildWords.ttf", "wildwords.ttf",
            "arialbd.ttf", "ARIALBD.TTF", "calibrib.ttf", "verdanab.ttf",
            "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
            "arial.ttf",
        ]
        if os.name == "nt":
            sys_dirs = ["C:/Windows/Fonts"]
            local = os.environ.get("LOCALAPPDATA", "")
            if local:
                sys_dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
        else:
            sys_dirs = [
                "/usr/share/fonts", "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts"),
                "/Library/Fonts", "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ]
        bold_kw = ("bold", "bd", "black", "heavy")
        for d in sys_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for name in files:
                    if any(kw in name.lower() for kw in bold_kw):
                        candidates.append(os.path.join(root, name))
                break

        for cand in candidates:
            try:
                ImageFont.truetype(cand, 12)
                return cand
            except OSError:
                continue
        return None

    resolved = _resolve_font(font_path)
    global DEFAULT_FONT
    if resolved:
        DEFAULT_FONT = resolved
        if resolved == font_path:
            print(f"[font] Using: {resolved}")
        else:
            print(f"[font] '{font_path}' not found — auto-selected: {resolved}")
    else:
        print(f"[font] ⚠ No usable font found, including system fallbacks. "
              f"Will use PIL default (text may render poorly).")

    files = sorted(
        [f for f in os.listdir(input_dir)
         if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS],
        key=natural_key,
    )
    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Pages found: {len(files)}")
    os.makedirs(output_dir, exist_ok=True)

    manga_ctx = MangaContext()
    archive = CharacterArchive("characters.json")
    errors = ErrorLog(error_log_path)

    if on_start:
        on_start(len(files))

    total_start = time.perf_counter()
    page_times: list[float] = []
    failed = 0
    all_bubbles: list[list[dict]] = []

    for page_idx, filename in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            print("[abort] Translation cancelled by user.")
            break
        input_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}_translated.png")

        page_start = time.perf_counter()
        bubbles_result = []
        try:
            bubbles_result = process_page(
                input_path, page_idx, manga_ctx, archive,
                output_path, target_lang, debug=debug,
                fast_mode=fast_mode, errors=errors,
                on_stage=on_stage,
            ) or []
            elapsed = time.perf_counter() - page_start
            page_times.append(elapsed)
            print(f"  ⏱  page processed in {format_duration(elapsed)}")
        except Exception as e:
            failed += 1
            elapsed = time.perf_counter() - page_start
            print(f"\n[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()
            errors.add(page_idx, "page_failed",
                       f"Page not processed: {type(e).__name__}: {e}",
                       filename=filename,
                       traceback=traceback.format_exc())

        all_bubbles.append(bubbles_result)
        if on_page_done:
            try:
                on_page_done(page_idx, len(files), filename, output_path,
                             bubbles_result, elapsed)
            except Exception as cb_err:
                print(f"  [callback warn] on_page_done: {cb_err}")

    total_elapsed = time.perf_counter() - total_start

    errors.save()

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Pages processed:    {len(page_times)} / {len(files)}")
    if failed:
        print(f"  Errors:             {failed}")
    print(f"  Total time:         {format_duration(total_elapsed)}")
    if page_times:
        avg = sum(page_times) / len(page_times)
        print(f"  Avg per page:       {format_duration(avg)}")
        print(f"  Fastest:            {format_duration(min(page_times))}")
        print(f"  Slowest:            {format_duration(max(page_times))}")
    print(f"  Results:            {output_dir}")
    print(f"  Character archive:  characters.json ({len(archive.characters)} characters)")

    if errors.entries:
        print(f"\n  ⚠ Issues recorded: {len(errors.entries)}")
        for kind, count in sorted(errors.summary().items(),
                                    key=lambda x: -x[1]):
            print(f"     {kind:30s} {count}")
        print(f"  Details:            {error_log_path}")
    else:
        print(f"  ✓ No issues recorded")

    stats = {
        "total_pages": len(files),
        "processed": len(page_times),
        "failed": failed,
        "total_seconds": total_elapsed,
        "avg_seconds": (sum(page_times) / len(page_times)) if page_times else 0,
        "errors": errors.summary(),
        "characters_count": len(archive.characters),
    }
    if on_finish:
        on_finish(stats)
    return stats


if __name__ == "__main__":
    process_directory(
        input_dir="input",
        output_dir="results",
        target_lang="Russian",
        font_path="arial.ttf",
        debug=False,
        error_log_path="errors.log",
    )
