import cv2
import numpy as np
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoPipelineForInpainting
import base64
import requests
import os
import re
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()

MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}
OLLAMA_URL = "http://localhost:11434/api/generate"
CROPS_DIR = "crops"
RESULTS_DIR = "results"
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID)
model.eval()

# ─── архив персонажей ─────────────────────────────────────────────────────────

class CharacterArchive:
    def __init__(self, path: str = "characters.json"):
        self.path = path
        self.characters: dict = {}  # id → {name, gender, appearance, notes, first_seen}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.characters = json.load(f)
            print(f"  Загружен архив персонажей: {len(self.characters)} шт.")

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
        """Принимает список персонажей от Gemma и обновляет архив."""
        for char in data:
            cid = char.get("id", "").strip()
            if not cid:
                continue
            if cid in self.characters:
                # Обновляем заметки если появилось что-то новое
                existing = self.characters[cid]
                new_notes = char.get("notes", "")
                if new_notes and new_notes not in existing.get("notes", ""):
                    existing["notes"] = (existing.get("notes", "") + "; " + new_notes).strip("; ")
            else:
                # Новый персонаж
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
        """Ищет персонажа по описанию (для override gender)."""
        desc_lower = description.lower()
        for cid, c in self.characters.items():
            if c["name"].lower() in desc_lower or cid.lower() in desc_lower:
                return c
        return None


# ─── глобальный контекст манги ────────────────────────────────────────────────

class MangaContext:
    def __init__(self):
        self.story_so_far: str = ""
        self.page_summaries: list[str] = []

    def update(self, page_summary: str):
        self.page_summaries.append(page_summary)
        if len(self.page_summaries) > 5:
            self.page_summaries.pop(0)
        self.story_so_far = "\n".join(
            f"Page {i+1}: {s}" for i, s in enumerate(self.page_summaries)
        )

    def to_prompt(self) -> str:
        if not self.story_so_far:
            return "This is the first page."
        return f"STORY SO FAR:\n{self.story_so_far}"


# ─── утилиты ──────────────────────────────────────────────────────────────────

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)
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
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


# ─── анализ персонажей на странице ───────────────────────────────────────────

def natural_key(filename):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

def analyze_characters(image_path: str, archive: CharacterArchive,
                        page_idx: int) -> str:
    print("  Анализ персонажей...")

    prompt = f"""You are a manga character analyst tracking characters across pages.

{archive.to_prompt()}

Look at this manga page. For EACH visible character, follow these steps:

STEP 1 - MATCH: Compare their appearance to every character in the archive above.
- Look for matches by: hair color, hair style, face features, clothing, body type
- If appearance matches an existing character → use their existing ID and name
- Only create a NEW entry if NO existing character matches

STEP 2 - OUTPUT: Return a JSON array:
[
  {{
    "id": "existing_id_or_new_snake_case",
    "name": "character name or description",
    "gender": "male/female/unknown",
    "appearance": "detailed: hair color+style, face, clothing, build",
    "position": "top-left / center / bottom-right / etc",
    "emotion": "calm / angry / surprised / etc",
    "notes": "any relevant info",
    "is_new": true/false
  }},
  ...
]

STRICT RULES:
- If a character is in the archive, use their EXACT existing id
- gender must come from archive if character is known
- appearance description must be detailed enough to match on future pages
- do NOT create duplicate entries for the same character
- is_new = false if matched to archive, true if genuinely new
"""

    raw = ollama("gemma4:26b", prompt, image_path)
    chars = parse_json_array(raw)

    if not chars:
        print(f"  [warn] персонажи не распарсились: {raw[:200]}")
        return "CHARACTERS ON THIS PAGE: unknown"

    # Дополнительная проверка дублей на нашей стороне
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
    Проверяем каждого нового персонажа — не похож ли он на кого-то в архиве.
    Сравниваем по ключевым словам внешности.
    """
    result = []
    for char in chars:
        cid = char.get("id", "")
        appearance = char.get("appearance", "").lower()

        # Если ID уже есть в архиве — это не новый персонаж
        if cid in archive.characters:
            char["is_new"] = False
            result.append(char)
            continue

        # Ищем похожего по внешности если модель пометила как новый
        if char.get("is_new", True):
            match = find_similar_in_archive(appearance, archive)
            if match:
                existing_id, existing = match
                print(f"  [дедупл] '{char.get('name')}' → совпадает с '{existing['name']}' (id={existing_id})")
                char["id"] = existing_id
                char["name"] = existing["name"]
                char["gender"] = existing["gender"]
                char["is_new"] = False

        result.append(char)
    return result


def find_similar_in_archive(appearance: str, archive: CharacterArchive) -> tuple | None:
    """
    Простой поиск по ключевым словам внешности.
    Возвращает (id, character) если найдено совпадение.
    """
    # Извлекаем ключевые слова из описания внешности
    keywords = extract_appearance_keywords(appearance)
    if not keywords:
        return None

    best_match = None
    best_score = 0

    for cid, c in archive.characters.items():
        archived_appearance = c.get("appearance", "").lower()
        archived_keywords = extract_appearance_keywords(archived_appearance)

        # Считаем пересечение ключевых слов
        common = keywords & archived_keywords
        score = len(common)

        if score >= 2 and score > best_score:  # минимум 2 совпавших ключевых слова
            best_score = score
            best_match = (cid, c)

    return best_match


def extract_appearance_keywords(text: str) -> set:
    """Извлекает значимые слова внешности для сравнения."""
    # Стоп-слова которые не несут смысла для идентификации
    stopwords = {
        "a", "an", "the", "with", "and", "or", "is", "are", "has", "have",
        "wearing", "looking", "man", "woman", "person", "character", "young",
        "old", "tall", "short", "small", "large", "none", "partially", "covered"
    }
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return {w for w in words if w not in stopwords and len(w) > 2}


# ─── анализ страницы ──────────────────────────────────────────────────────────

def analyze_page(image_path: str, manga_ctx: MangaContext,
                 characters_context: str) -> tuple[str, str]:
    print("  Анализ сцены...")
    prompt = f"""You are a manga analyst.

{manga_ctx.to_prompt()}

{characters_context}

Look at this manga page and describe:

CONTEXT:
[What is happening in this scene, 2-3 sentences]

SUMMARY:
[One sentence for future reference]"""

    raw = ollama("gemma4:26b", prompt, image_path)
    summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\n\n|$)", raw, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else raw[:120]
    return raw, summary


# ─── атрибуция баблов ─────────────────────────────────────────────────────────

def attribute_bubbles(image_path: str, bubbles: list[dict],
                      page_context: str, characters_context: str,
                      archive: CharacterArchive) -> list[dict]:
    print("  Атрибуция баблов...")
    bubble_list = "\n".join([
        f"[{i+1}] pos=({b['x']},{b['y']}) size={b['width']}x{b['height']} "
        f"text=\"{b.get('text', '')}\""
        for i, b in enumerate(bubbles)
    ])

    prompt = f"""You are analyzing a manga page.

{archive.to_prompt()}

{characters_context}

PAGE CONTEXT:
{page_context}

SPEECH BUBBLES:
{bubble_list}

For each bubble determine the speaker by position and context.
Use character names and genders from the archive — do NOT reassign gender.

Return ONLY JSON:
[
  {{"bubble": 1, "speaker": "character name", "gender": "male/female/unknown"}},
  ...
]"""

    raw = ollama("gemma4:26b", prompt, image_path)
    attributions = parse_json_array(raw)

    for attr in attributions:
        idx = attr.get("bubble", 0) - 1
        if 0 <= idx < len(bubbles):
            speaker = attr.get("speaker", "unknown")
            gender = attr.get("gender", "unknown")
            # Проверяем архив для override пола
            known = archive.find_character(speaker)
            if known:
                gender = known["gender"]
                speaker = known["name"]
            bubbles[idx]["speaker"] = speaker
            bubbles[idx]["gender"] = gender

    return bubbles


# ─── перевод с контекстом ─────────────────────────────────────────────────────

def translate_with_context(text: str, speaker: str, gender: str,
                            page_context: str, manga_ctx: MangaContext,
                            target_lang: str = "Russian",
                            retries: int = 3) -> str:
    if not text:
        return ""

    gender_hint = {
        "male": "мужской персонаж, используй мужской род",
        "female": "женский персонаж, используй женский род",
        "unknown": "род неизвестен",
    }.get(gender, "род неизвестен")

    prompt = f"""Translate manga dialogue to {target_lang}.

{manga_ctx.to_prompt()}

PAGE CONTEXT:
{page_context}

SPEAKER: {speaker} ({gender_hint})

Translate naturally, matching character personality.
Filter OCR artifacts. Return ONLY the translation.

Text: {text}"""

    for attempt in range(retries):
        try:
            return ollama("gemma4:26b", prompt, timeout=280)
        except requests.exceptions.ReadTimeout:
            print(f"     [timeout] попытка {attempt+1}/{retries}...")
    return "[timeout]"


# ─── препроцессинг / OCR ──────────────────────────────────────────────────────

def preprocess_crop(img_cv: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
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
    raw = ollama("glm-ocr:latest",
                 "Read and return ONLY the text in this manga speech bubble. No explanation.",
                 crop_path, timeout=60)
    print(f"     [RAW] {repr(raw)}")
    return clean_text(raw)


# ─── детекция ─────────────────────────────────────────────────────────────────

def detect_bubbles(image_path: str, threshold: float = 0.5) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[h, w]]), threshold=threshold,
    )[0]
    bubbles = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        cls = int(label)
        x1, y1, x2, y2 = map(int, box.tolist())
        bubbles.append({
            "class": CLASSES.get(cls),
            "x": x1, "y": y1,
            "width": x2 - x1, "height": y2 - y1,
            "confidence": float(score),
            "speaker": "unknown", "gender": "unknown",
        })
    return bubbles


# ─── отрисовка ────────────────────────────────────────────────────────────────

def fit_text_in_box(draw, text: str, box_w: int, box_h: int,
                    font_path: str = "arial.ttf") -> tuple:
    padding = 4
    usable_w = box_w - padding * 2
    usable_h = box_h - padding * 2

    for font_size in range(40, 4, -1):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            return font, [text], [12], 2

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
            continue

        line_heights = [
            draw.textbbox((0, 0), ln, font=font)[3] - draw.textbbox((0, 0), ln, font=font)[1]
            for ln in lines
        ]
        spacing = max(2, font_size // 8)
        total_h = sum(line_heights) + spacing * (len(lines) - 1)
        max_line_w = max(draw.textbbox((0, 0), ln, font=font)[2] for ln in lines)

        if total_h <= usable_h and max_line_w <= usable_w:
            return font, lines, line_heights, spacing

    try:
        font = ImageFont.truetype(font_path, 5)
    except:
        font = ImageFont.load_default()
    return font, [text[:20]], [8], 2


def draw_results(img_cv: np.ndarray, bubbles: list[dict]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
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
        text_y = y + padding + max(0, (h - padding*2 - total_h) // 2)

        for j, line in enumerate(lines):
            bb = draw.textbbox((0, 0), line, font=font)
            line_w = bb[2] - bb[0]
            text_x = x + padding + max(0, (w - padding*2 - line_w) // 2)
            text_x = min(text_x, x + w - padding - line_w)
            draw.text((text_x, text_y), line, fill=(180, 0, 0), font=font)
            text_y += line_heights[j] + spacing

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─── обработка страницы ───────────────────────────────────────────────────────

def reading_order(bubble: dict) -> tuple:
    return (bubble["y"] // 150, -bubble["x"])

def process_page(image_path: str, page_idx: int,
                 manga_ctx: MangaContext, archive: CharacterArchive,
                 output_path: str, target_lang: str = "Russian"):
    print(f"\n{'='*60}")
    print(f"Страница {page_idx}: {image_path}")
    print(f"{'='*60}")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        img_cv = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    bubbles = detect_bubbles(image_path, threshold=0.5)
    text_bubbles = sorted(
        [b for b in bubbles if b["class"] in ("text_bubble", "text_free")],
        key=reading_order
    )
    print(f"Найдено баблов: {len(text_bubbles)}")

    # 1. OCR
    print("\n── OCR ──")
    for i, b in enumerate(text_bubbles):
        b["text"] = ocr_region(img_cv, b["x"], b["y"], b["width"], b["height"],
                               idx=i+1, page_idx=page_idx)

    # 2. Анализ и обновление архива персонажей
    print("\n── Персонажи ──")
    characters_context = analyze_characters(image_path, archive, page_idx)
    print(characters_context)

    # 3. Анализ сцены
    print("\n── Сцена ──")
    page_context, page_summary = analyze_page(image_path, manga_ctx, characters_context)
    print(f"  {page_context[:200]}...")

    # 4. Атрибуция баблов
    print("\n── Атрибуция ──")
    text_bubbles = attribute_bubbles(
        image_path, text_bubbles, page_context, characters_context, archive
    )
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('speaker','?')} ({b.get('gender','?')}): {b.get('text','')[:40]}")

    # 5. Перевод
    print("\n── Перевод ──")
    for i, b in enumerate(text_bubbles):
        b["translation"] = translate_with_context(
            b.get("text", ""), b.get("speaker", "unknown"),
            b.get("gender", "unknown"), page_context, manga_ctx, target_lang,
        )
        print(f"  [{i+1}] {b.get('text','')[:25]} → {b['translation'][:40]}")

    # 6. Обновляем сюжетный контекст
    manga_ctx.update(page_summary)

    # 7. Сохраняем
    annotated = draw_results(img_cv, text_bubbles)
    cv2.imwrite(output_path, annotated)
    print(f"\nСохранено: {output_path}")
    return text_bubbles


# ─── обработка директории ─────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def process_directory(input_dir: str, output_dir: str = RESULTS_DIR,
                       target_lang: str = "Russian"):
    files = sorted(
        [f for f in os.listdir(input_dir)
         if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS],
        key=natural_key
    )

    if not files:
        print(f"Изображения не найдены в {input_dir}")
        return

    print(f"Найдено страниц: {len(files)}")
    os.makedirs(output_dir, exist_ok=True)

    manga_ctx = MangaContext()
    archive = CharacterArchive("characters.json")  # персонажи сохраняются между запусками

    for page_idx, filename in enumerate(files, start=1):
        input_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}_translated.png")

        try:
            process_page(input_path, page_idx, manga_ctx, archive,
                         output_path, target_lang)
        except Exception as e:
            print(f"\n[ERROR] {filename}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Готово! Страниц: {len(files)} | Результаты: {output_dir}")
    print(f"Архив персонажей: characters.json ({len(archive.characters)} персонажей)")


if __name__ == "__main__":
    process_directory(
        input_dir="input",
        output_dir="results",
        target_lang="Russian",
    )