"""
manga_translator.py
Автоматический перевод манги с помощью локальных LLM (Ollama / gemma4).

Пайплайн на страницу:
  1. Детекция баблов     — RT-DETRv2
  2. OCR                 — glm-ocr через Ollama
  3. Анализ персонажей   — gemma4 + CharacterArchive
  4. Анализ сцены        — gemma4 + MangaContext
  5. Атрибуция реплик    — gemma4 (кто говорит + пол)
  6. Перевод             — gemma4 с учётом контекста
  7. Отрисовка и запись  — PIL, OpenCV
"""

import os
import re
import json
import base64
import warnings

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoPipelineForInpainting
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

warnings.filterwarnings("ignore")


# ─── константы ────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_URL = "http://localhost:11434/api/generate"
CROPS_DIR = "crops"
RESULTS_DIR = "results"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

BUBBLE_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
BUBBLE_CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── инициализация моделей ────────────────────────────────────────────────────

def load_inpainting_pipe():
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe

def load_detector():
    processor = AutoImageProcessor.from_pretrained(BUBBLE_MODEL_ID)
    model = RTDetrV2ForObjectDetection.from_pretrained(BUBBLE_MODEL_ID)
    model.eval()
    return processor, model

pipe = load_inpainting_pipe()
detector_processor, detector_model = load_detector()


# ─── архив персонажей ─────────────────────────────────────────────────────────

class CharacterArchive:
    """
    Хранит описания персонажей между запусками (characters.json).
    Позволяет LLM переиспользовать уже известных персонажей.
    """

    def __init__(self, path: str = "characters.json"):
        self.path = path
        self.characters: dict = {}   # id → {name, gender, appearance, notes, first_seen}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.characters = json.load(f)
            print(f"  Загружен архив: {len(self.characters)} персонажей")

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

    def update_from_json(self, data: list, page_idx: int):
        """Принимает список персонажей от LLM и обновляет архив."""
        for char in data:
            cid = char.get("id", "").strip()
            if not cid:
                continue
            if cid in self.characters:
                # Добавляем новые заметки к существующим
                new_notes = char.get("notes", "")
                if new_notes and new_notes not in self.characters[cid].get("notes", ""):
                    self.characters[cid]["notes"] = (
                        self.characters[cid].get("notes", "") + "; " + new_notes
                    ).strip("; ")
            else:
                self.characters[cid] = {
                    "name": char.get("name", cid),
                    "gender": char.get("gender", "unknown"),
                    "appearance": char.get("appearance", ""),
                    "notes": char.get("notes", ""),
                    "first_seen": page_idx,
                }
                print(f"  [архив] Новый персонаж: {char.get('name', cid)}")
        self.save()

    def find_character(self, description: str) -> dict | None:
        """Находит персонажа по имени или ID в строке описания."""
        desc_lower = description.lower()
        for cid, c in self.characters.items():
            if c["name"].lower() in desc_lower or cid.lower() in desc_lower:
                return c
        return None


# ─── контекст сюжета ──────────────────────────────────────────────────────────

class MangaContext:
    """Хранит краткое содержание последних страниц для контекстного перевода."""

    def __init__(self):
        self.page_summaries: list[str] = []

    def update(self, summary: str):
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


# ─── вспомогательные функции ──────────────────────────────────────────────────

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_text(text: str) -> str:
    """Убирает markdown-артефакты и лишние пробелы."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def ollama(model_name: str, prompt: str, image_path: str = None,
           timeout: int = 800) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 6000},
    }
    if image_path:
        payload["images"] = [image_to_base64(image_path)]
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    return r.json().get("response", "").strip()

def parse_json_array(text: str) -> list:
    """Извлекает первый JSON-массив из произвольного текста."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []

def natural_key(filename: str) -> list:
    """Сортировочный ключ: page2 < page10."""
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", filename)
    ]


# ─── анализ персонажей ────────────────────────────────────────────────────────

def analyze_characters(image_path: str, archive: CharacterArchive,
                        page_idx: int) -> str:
    """Возвращает текстовый блок о персонажах на странице и обновляет архив."""
    print("  Анализ персонажей...")

    prompt = f"""You are a manga character analyst tracking characters across pages.

{archive.to_prompt()}

Look at this manga page. For EACH visible character:

STEP 1 — MATCH: Compare to every character in the archive.
- Match by: hair color, style, face, clothing, body type
- If matched → use existing ID and name
- Only create NEW if no match found

STEP 2 — Return JSON array:
[
  {{
    "id": "existing_id_or_new_snake_case",
    "name": "character name or description",
    "gender": "male/female/unknown",
    "appearance": "hair color+style, face, clothing, build",
    "position": "top-left / center / bottom-right / etc",
    "emotion": "calm / angry / surprised / etc",
    "notes": "any relevant info",
    "is_new": true/false
  }}
]

RULES:
- Known character → use EXACT existing id and gender
- is_new = false if matched, true if genuinely new
- No duplicates"""

    raw = ollama("gemma4:26b", prompt, image_path)
    chars = parse_json_array(raw)

    if not chars:
        print(f"  [warn] персонажи не распарсились: {raw[:200]}")
        return "CHARACTERS ON THIS PAGE: unknown"

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
    return "\n".join(lines)


def deduplicate_characters(chars: list, archive: CharacterArchive) -> list:
    """
    Дополнительная проверка на дубли со стороны Python:
    сравниваем ключевые слова внешности с архивом.
    """
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
                print(f"  [дедупл] '{char.get('name')}' → '{existing['name']}' (id={existing_id})")
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
    """Возвращает (id, character), если нашлось ≥2 совпавших ключевых слова."""
    keywords = extract_appearance_keywords(appearance)
    if not keywords:
        return None

    best_match, best_score = None, 0
    for cid, c in archive.characters.items():
        common = keywords & extract_appearance_keywords(
            c.get("appearance", "").lower()
        )
        if len(common) >= 2 and len(common) > best_score:
            best_score = len(common)
            best_match = (cid, c)
    return best_match


def extract_appearance_keywords(text: str) -> set:
    """Значимые слова внешности для сравнения (без стоп-слов)."""
    stopwords = {
        "a", "an", "the", "with", "and", "or", "is", "are", "has", "have",
        "wearing", "looking", "man", "woman", "person", "character", "young",
        "old", "tall", "short", "small", "large", "none", "partially", "covered",
    }
    return {
        w for w in re.findall(r"\b[a-z]+\b", text)
        if w not in stopwords and len(w) > 2
    }


# ─── анализ сцены ─────────────────────────────────────────────────────────────

def analyze_page(image_path: str, manga_ctx: MangaContext,
                 characters_context: str) -> tuple[str, str]:
    """
    Возвращает (page_context, page_summary) — оба извлекаются из размеченного
    ответа модели. В page_context не попадает преамбула с архивом, чтобы
    не дублировать её в последующих промптах.
    """
    print("  Анализ сцены...")
    prompt = f"""You are a manga analyst.

{manga_ctx.to_prompt()}

{characters_context}

Describe this manga page using EXACTLY this format:

CONTEXT:
<2-3 sentences about what is happening in this scene>

SUMMARY:
<one short sentence for future reference>

END"""

    raw = ollama("gemma4:26b", prompt, image_path)

    # Жёстко извлекаем содержимое между маркерами
    context_match = re.search(
        r"CONTEXT:\s*(.+?)\s*(?:SUMMARY:|END|\Z)", raw, re.DOTALL | re.IGNORECASE
    )
    summary_match = re.search(
        r"SUMMARY:\s*(.+?)\s*(?:END|\Z)", raw, re.DOTALL | re.IGNORECASE
    )

    page_context = context_match.group(1).strip() if context_match else raw[:300].strip()
    page_summary = summary_match.group(1).strip() if summary_match else page_context[:120]

    # Подстрахуемся от слишком длинных артефактов
    page_context = page_context[:800]
    page_summary = page_summary.split("\n")[0][:200]

    return page_context, page_summary


# ─── атрибуция реплик ─────────────────────────────────────────────────────────

def attribute_bubbles(image_path: str, bubbles: list[dict],
                      page_context: str, characters_context: str,
                      archive: CharacterArchive) -> list[dict]:
    """Определяет говорящего и его пол для каждого бабла."""
    print("  Атрибуция баблов...")
    bubble_list = "\n".join(
        f'[{i+1}] pos=({b["x"]},{b["y"]}) size={b["width"]}x{b["height"]} '
        f'text="{b.get("text", "")}"'
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

    raw = ollama("gemma4:26b", prompt, image_path)
    for attr in parse_json_array(raw):
        idx = attr.get("bubble", 0) - 1
        if 0 <= idx < len(bubbles):
            speaker = attr.get("speaker", "unknown")
            gender = attr.get("gender", "unknown")
            # Пол из архива имеет приоритет
            known = archive.find_character(speaker)
            if known:
                speaker = known["name"]
                gender = known["gender"]
            bubbles[idx]["speaker"] = speaker
            bubbles[idx]["gender"] = gender

    return bubbles


# ─── перевод ──────────────────────────────────────────────────────────────────

def translate_batch(bubbles: list[dict], page_context: str,
                    manga_ctx: MangaContext, target_lang: str = "Russian",
                    retries: int = 3) -> None:
    """
    Переводит все реплики страницы за один вызов LLM.
    Записывает результат в поле bubble["translation"] in-place.
    Баблы без текста пропускаются и получают пустую строку.
    """
    gender_hints = {
        "male": "мужской род",
        "female": "женский род",
        "unknown": "род неизвестен",
    }

    # Отбираем только баблы с текстом; остальные сразу помечаем пустыми
    to_translate = []
    for i, b in enumerate(bubbles):
        if b.get("text", "").strip():
            to_translate.append((i, b))
        else:
            b["translation"] = ""

    if not to_translate:
        return

    lines = "\n".join(
        f'[{seq+1}] speaker="{b["speaker"]}" '
        f'gender_hint="{gender_hints.get(b["gender"], "род неизвестен")}" '
        f'text="{b["text"]}"'
        for seq, (_, b) in enumerate(to_translate)
    )

    prompt = f"""Translate manga dialogue to {target_lang}.

{manga_ctx.to_prompt()}

PAGE CONTEXT:
{page_context}

Translate each line naturally, matching each character's personality.
Filter OCR artifacts. Preserve original emotion and register.

Return ONLY a JSON array — one object per input line, in the same order:
[
  {{"id": 1, "translation": "..."}},
  ...
]

Lines to translate:
{lines}"""

    for attempt in range(retries):
        try:
            raw = ollama("gemma4:26b", prompt, timeout=600)
            break
        except requests.exceptions.ReadTimeout:
            print(f"     [timeout] попытка {attempt+1}/{retries}...")
            raw = ""

    results = parse_json_array(raw)
    # Сопоставляем по порядковому id (1-based) из ответа
    id_to_translation = {r["id"]: r.get("translation", "") for r in results if "id" in r}

    for seq, (bubble_idx, b) in enumerate(to_translate):
        translation = id_to_translation.get(seq + 1, "")
        if not translation:
            # Fallback: берём по позиции если модель не вернула id
            translation = results[seq].get("translation", "") if seq < len(results) else "[error]"
        b["translation"] = translation


# ─── препроцессинг + OCR ─────────────────────────────────────────────────────

def preprocess_crop(img_cv: np.ndarray, x: int, y: int,
                    w: int, h: int) -> np.ndarray:
    """Вырезает регион, масштабирует ×3, выравнивает контраст, добавляет отступ."""
    crop = img_cv[y:y+h, x:x+w]
    h2, w2 = crop.shape[:2]
    crop = cv2.resize(crop, (w2 * 3, h2 * 3), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.copyMakeBorder(gray, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=255)


def ocr_region(img_cv: np.ndarray, x: int, y: int, w: int, h: int,
               idx: int, page_idx: int) -> str:
    processed = preprocess_crop(img_cv, x, y, w, h)
    crop_path = os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{idx:02d}.png")
    cv2.imwrite(crop_path, processed)
    raw = ollama(
        "glm-ocr:latest",
        "Read and return ONLY the text in this manga speech bubble. No explanation.",
        crop_path,
        timeout=60,
    )
    print(f"     [OCR raw] {repr(raw)}")
    return clean_text(raw)


# ─── детекция баблов ─────────────────────────────────────────────────────────

def detect_bubbles(image_pil: Image.Image, threshold: float = 0.5) -> list[dict]:
    """Принимает уже загруженный PIL-объект — файл не читается повторно."""
    w, h = image_pil.size
    inputs = detector_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = detector_model(**inputs)
    results = detector_processor.post_process_object_detection(
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


# ─── отрисовка ────────────────────────────────────────────────────────────────

def fit_text_in_box(draw: ImageDraw.Draw, text: str, box_w: int,
                    box_h: int, font_path: str = "arial.ttf") -> tuple:
    """
    Подбирает максимальный размер шрифта бинарным поиском (O(log n) вместо O(n)).
    Возвращает (font, lines, line_heights, spacing).
    """
    padding = 4
    usable_w = box_w - padding * 2
    usable_h = box_h - padding * 2

    def try_load_font(size: int):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            return None

    def wrap_and_measure(font) -> tuple[list[str], list[int], int] | None:
        """Переносит текст и измеряет итоговые размеры. None = не помещается."""
        words = text.split()
        lines, line = [], ""
        for word in words:
            test = (line + " " + word).strip()
            if draw.textbbox((0, 0), test, font=font)[2] <= usable_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        if not lines:
            return None

        line_heights = [
            draw.textbbox((0, 0), ln, font=font)[3] - draw.textbbox((0, 0), ln, font=font)[1]
            for ln in lines
        ]
        spacing = max(2, font.size // 8)
        total_h = sum(line_heights) + spacing * (len(lines) - 1)
        max_w = max(draw.textbbox((0, 0), ln, font=font)[2] for ln in lines)

        if total_h <= usable_h and max_w <= usable_w:
            return lines, line_heights, spacing
        return None

    # Проверяем, грузится ли шрифт вообще (один раз)
    if try_load_font(10) is None:
        font = ImageFont.load_default()
        return font, [text], [12], 2

    # Бинарный поиск: ищем максимальный размер, при котором текст помещается
    lo, hi = 5, 40
    best: tuple | None = None
    best_size = lo

    while lo <= hi:
        mid = (lo + hi) // 2
        font = try_load_font(mid)
        result = wrap_and_measure(font)
        if result is not None:
            best = (font, *result)
            best_size = mid
            lo = mid + 1   # пробуем крупнее
        else:
            hi = mid - 1   # слишком большой — уменьшаем

    if best is not None:
        return best

    # Fallback: минимальный размер, текст обрезается
    font = try_load_font(5) or ImageFont.load_default()
    return font, [text[:20]], [8], 2


def draw_results(img_cv: np.ndarray, bubbles: list[dict]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font_small = ImageFont.load_default()

    for i, b in enumerate(bubbles):
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        translation = b.get("translation", "")
        color = (0, 200, 0) if b["class"] == "text_bubble" else (0, 150, 255)

        draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=2)
        draw.rectangle([(x, y-22), (x+18, y)], fill=color)
        draw.text((x+3, y-20), str(i+1), fill=(0, 0, 0), font=font_small)

        if not translation:
            continue

        padding = 4
        draw.rectangle([(x+1, y+1), (x+w-1, y+h-1)], fill=(255, 255, 255))
        font, lines, line_heights, spacing = fit_text_in_box(draw, translation, w, h)
        total_h = sum(line_heights) + spacing * (len(lines) - 1)
        text_y = y + padding + max(0, (h - padding * 2 - total_h) // 2)

        for j, line in enumerate(lines):
            bb = draw.textbbox((0, 0), line, font=font)
            line_w = bb[2] - bb[0]
            text_x = x + padding + max(0, (w - padding * 2 - line_w) // 2)
            text_x = min(text_x, x + w - padding - line_w)
            draw.text((text_x, text_y), line, fill=(180, 0, 0), font=font)
            text_y += line_heights[j] + spacing

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─── обработка страницы ───────────────────────────────────────────────────────

def reading_order(bubble: dict) -> tuple:
    """Сортировка баблов: сверху вниз, справа налево (японский порядок)."""
    return (bubble["y"] // 150, -bubble["x"])


def process_page(image_path: str, page_idx: int,
                 manga_ctx: MangaContext, archive: CharacterArchive,
                 output_path: str, target_lang: str = "Russian"):
    print(f"\n{'='*60}")
    print(f"Страница {page_idx}: {image_path}")
    print(f"{'='*60}")

    # Читаем файл один раз — оба формата получают данные из одного буфера
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        img_cv = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # 1. Детекция и сортировка баблов (PIL уже в памяти — файл не перечитывается)
    bubbles = detect_bubbles(image_pil, threshold=0.5)
    text_bubbles = sorted(
        [b for b in bubbles if b["class"] in ("text_bubble", "text_free")],
        key=reading_order,
    )
    print(f"Баблов: {len(text_bubbles)}")

    # 2. OCR
    print("\n── OCR ──")
    for i, b in enumerate(text_bubbles):
        b["text"] = ocr_region(img_cv, b["x"], b["y"], b["width"], b["height"],
                               idx=i + 1, page_idx=page_idx)

    # 3. Персонажи
    print("\n── Персонажи ──")
    characters_context = analyze_characters(image_path, archive, page_idx)
    print(characters_context)

    # 4. Сцена
    print("\n── Сцена ──")
    page_context, page_summary = analyze_page(image_path, manga_ctx, characters_context)
    print(f"  context: {page_context}")
    print(f"  summary: {page_summary}")

    # 5. Атрибуция реплик
    print("\n── Атрибуция ──")
    text_bubbles = attribute_bubbles(
        image_path, text_bubbles, page_context, characters_context, archive
    )
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('speaker', '?')} ({b.get('gender', '?')}): "
              f"{b.get('text', '')[:40]}")

    # 6. Перевод — один вызов LLM на всю страницу
    print("\n── Перевод ──")
    translate_batch(text_bubbles, page_context, manga_ctx, target_lang)
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('text', '')[:25]} → {b.get('translation', '')[:40]}")

    # 7. Обновляем сюжетный контекст
    manga_ctx.update(page_summary)

    # 8. Сохраняем аннотированное изображение
    annotated = draw_results(img_cv, text_bubbles)
    cv2.imwrite(output_path, annotated)
    print(f"\nСохранено: {output_path}")
    return text_bubbles


# ─── обработка директории ─────────────────────────────────────────────────────

def process_directory(input_dir: str, output_dir: str = RESULTS_DIR,
                      target_lang: str = "Russian"):
    files = sorted(
        [f for f in os.listdir(input_dir)
         if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS],
        key=natural_key,
    )
    if not files:
        print(f"Изображения не найдены в {input_dir}")
        return

    print(f"Найдено страниц: {len(files)}")
    os.makedirs(output_dir, exist_ok=True)

    manga_ctx = MangaContext()
    archive = CharacterArchive("characters.json")

    for page_idx, filename in enumerate(files, start=1):
        input_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}_translated.png")
        try:
            process_page(input_path, page_idx, manga_ctx, archive,
                         output_path, target_lang)
        except Exception as e:
            print(f"\n[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Готово! Страниц: {len(files)} | Результаты: {output_dir}")
    print(f"Архив персонажей: characters.json ({len(archive.characters)} персонажей)")


if __name__ == "__main__":
    process_directory(
        input_dir="input",
        output_dir="results",
        target_lang="Russian",
    )
