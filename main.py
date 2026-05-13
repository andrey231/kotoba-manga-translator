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


# ─── константы ────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_URL = "http://localhost:11434/api/generate"
CROPS_DIR = "crops"
RESULTS_DIR = "results"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

BUBBLE_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
BUBBLE_CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}

# anime-big-lama — модель для инпейтинга, заточенная под мангу.
# Список репозиториев пробуется по порядку: первый — официальное место,
# второй — зеркало df1412 на случай если первый недоступен.
LAMA_REPOS = [
    ("deckyfx/anime-big-lama", "anime-manga-big-lama.pt"),
    ("df1412/anime-big-lama",  "anime-manga-big-lama.pt"),
]

DEFAULT_FONT = "arial.ttf"   # переопределяется через process_directory(font_path=...)

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── инициализация моделей ────────────────────────────────────────────────────

def load_inpainting_model():
    """
    Загружает anime-big-lama (TorchScript) с HuggingFace.
    Пробует несколько репозиториев по очереди.
    """
    last_error = None
    for repo_id, filename in LAMA_REPOS:
        try:
            print(f"[lama] Загружаем {repo_id}/{filename}...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model = torch.jit.load(model_path, map_location=DEVICE)
            model.eval()
            print(f"[lama] Загружена с {repo_id} ({DEVICE})")
            return model
        except Exception as e:
            print(f"[lama] Не удалось с {repo_id}: {e}")
            last_error = e
    print(f"[lama] ⚠ Все источники недоступны, инпейтинг будет работать через cv2.inpaint")
    print(f"       Последняя ошибка: {last_error}")
    return None


def load_detector():
    processor = AutoImageProcessor.from_pretrained(BUBBLE_MODEL_ID)
    model = RTDetrV2ForObjectDetection.from_pretrained(BUBBLE_MODEL_ID)
    model.eval()
    return processor, model

inpaint_model = load_inpainting_model()
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

    def _unique_name(self, proposed: str, cid: str) -> str:
        """
        Возвращает имя, уникальное среди уже существующих в архиве.
        Если такое имя занято — добавляет дисамбигуатор из ID персонажа.
        """
        existing_names = {c["name"].lower() for c in self.characters.values()}
        if proposed.lower() not in existing_names:
            return proposed

        # Имя занято — пробуем извлечь отличительную часть из ID
        # Пример: dark_haired_woman_top_right → "top right"
        base_words = set(re.findall(r"[a-z]+", proposed.lower()))
        id_words = re.findall(r"[a-z]+", cid.lower())
        extras = [w for w in id_words if w not in base_words and len(w) > 2]
        if extras:
            candidate = f"{proposed} ({' '.join(extras)})"
            if candidate.lower() not in existing_names:
                return candidate

        # Последний резерв — числовой суффикс
        for i in range(2, 100):
            candidate = f"{proposed} #{i}"
            if candidate.lower() not in existing_names:
                return candidate
        return proposed  # сдаёмся

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
                # Гарантируем что имя не дублирует уже существующего персонажа
                raw_name = char.get("name", cid)
                unique_name = self._unique_name(raw_name, cid)
                if unique_name != raw_name:
                    print(f"  [архив] Имя '{raw_name}' уже занято → '{unique_name}'")

                self.characters[cid] = {
                    "name": unique_name,
                    "gender": char.get("gender", "unknown"),
                    "appearance": char.get("appearance", ""),
                    "notes": char.get("notes", ""),
                    "first_seen": page_idx,
                }
                print(f"  [архив] Новый персонаж: {unique_name}")
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


# ─── журнал ошибок ────────────────────────────────────────────────────────────

class ErrorLog:
    """
    Накапливает проблемы за весь прогон и в конце пишет их в файл.
    Полезно для отладки: можно понять что пошло не так с конкретным баблом
    без перечитывания всего консольного вывода.
    """

    def __init__(self, path: str = "errors.log"):
        self.path = path
        self.entries: list[dict] = []
        # Чистим старый лог при старте — иначе он будет накапливаться вечно
        if os.path.exists(self.path):
            os.remove(self.path)

    def add(self, page: int, kind: str, message: str, **details):
        """
        kind: 'ocr_empty', 'translation_missing', 'json_parse', 'timeout',
              'page_failed', 'attribution', ...
        details: любые доп. поля, попадут в JSON-блок при дампе
        """
        entry = {
            "page": page,
            "kind": kind,
            "message": message,
            **details,
        }
        self.entries.append(entry)

    def summary(self) -> dict:
        """Группирует ошибки по типу — для краткой сводки в консоли."""
        counts: dict = {}
        for e in self.entries:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        return counts

    def save(self):
        if not self.entries:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"# Журнал ошибок — всего {len(self.entries)} записей\n")
            f.write(f"# Создан: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Группируем по странице для удобства чтения
            by_page: dict = {}
            for e in self.entries:
                by_page.setdefault(e["page"], []).append(e)

            for page in sorted(by_page):
                f.write(f"\n{'='*60}\n")
                f.write(f"Страница {page}\n")
                f.write(f"{'='*60}\n")
                for e in by_page[page]:
                    f.write(f"\n  [{e['kind']}] {e['message']}\n")
                    extras = {k: v for k, v in e.items()
                              if k not in ("page", "kind", "message")}
                    if extras:
                        for k, v in extras.items():
                            # Длинные значения красиво обрезаем
                            v_str = str(v)
                            if len(v_str) > 300:
                                v_str = v_str[:300] + "... [обрезано]"
                            f.write(f"    {k}: {v_str}\n")


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


# ─── распознавание страниц-представлений персонажей ──────────────────────────

def detect_character_intro_page(image_path: str) -> bool:
    """
    Определяет, является ли страница «представлением персонажей» —
    галерея портретов с подписями (имя + краткое описание под каждым).
    Такие страницы обычно идут в начале главы/тома.
    """
    prompt = """Look at this manga page.

Is this a CHARACTER INTRODUCTION page — a gallery of character portraits where
each character is shown alongside their name and a brief description (like a
cast page, character roster, or "dramatis personae")?

This is NOT an introduction page if it shows characters in a normal scene,
talking, or doing actions. It IS an introduction page only if the layout is
clearly a roster/gallery with labels.

Answer with EXACTLY one word: YES or NO."""

    raw = ollama("gemma4:26b", prompt, image_path, timeout=120).strip().upper()
    is_intro = raw.startswith("YES")
    print(f"  [intro detect] {raw[:30]} → {'это галерея' if is_intro else 'обычная страница'}")
    return is_intro


def extract_character_intros(image_path: str,
                              archive: CharacterArchive,
                              page_idx: int) -> str:
    """
    Извлекает с галереи представлений пары «персонаж → имя + описание»
    и записывает их в архив с in-image именами.
    Возвращает characters_context для последующих этапов пайплайна.
    """
    print("  Извлечение представлений персонажей...")

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

    raw = ollama("gemma4:26b", prompt, image_path)
    chars = parse_json_array(raw)

    if not chars:
        print(f"  [warn] представления не распарсились: {raw[:200]}")
        return "CHARACTERS ON THIS PAGE: unknown"

    # Переносим role в notes — у обычных персонажей этого поля нет,
    # но информация ценная для последующих переводов
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


# ─── совмещённый анализ персонажей + сцены ────────────────────────────────────

def analyze_page_full(image_path: str, archive: CharacterArchive,
                      manga_ctx: MangaContext,
                      page_idx: int) -> tuple[str, str, str]:
    """
    Один вызов LLM возвращает и персонажей, и описание сцены, и summary.
    Экономит ~30-50% времени по сравнению с двумя отдельными vision-вызовами.

    Возвращает: (characters_context, page_context, page_summary).
    """
    print("  Анализ страницы (персонажи + сцена)...")

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

    raw = ollama("gemma4:26b", prompt, image_path)

    # --- разбираем блок CHARACTERS ---
    chars_match = re.search(
        r"===\s*CHARACTERS\s*===(.*?)===\s*SCENE\s*===",
        raw, re.DOTALL | re.IGNORECASE,
    )
    chars_section = chars_match.group(1) if chars_match else raw
    chars = parse_json_array(chars_section)

    if not chars:
        print(f"  [warn] персонажи не распарсились: {chars_section[:200]}")
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

    # --- разбираем блок SCENE ---
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

    # Подстрахуемся от слишком длинных артефактов
    page_context = page_context[:800]
    page_summary = page_summary.split("\n")[0][:200]

    return characters_context, page_context, page_summary


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
    """
    Ищет совпадение по внешности с архивом.
    Возвращает (id, character) только если:
      - найдено достаточно общих ключевых слов (порог зависит от длины описания)
      - НЕТ конфликтующих различающих признаков (цвет волос, пол)
    Это предотвращает ложные совпадения между разными персонажами
    в одинаковой одежде/стиле.
    """
    keywords = extract_appearance_keywords(appearance)
    if not keywords:
        return None

    new_distinct = extract_distinctive_features(appearance)

    best_match, best_score = None, 0
    for cid, c in archive.characters.items():
        archived_appearance = c.get("appearance", "").lower()
        archived_keywords = extract_appearance_keywords(archived_appearance)
        common = keywords & archived_keywords

        # Порог зависит от размера: для коротких описаний нужно больше доля совпадений
        min_size = min(len(keywords), len(archived_keywords))
        threshold = max(3, min_size // 2)   # было: 2, теперь жёстче

        if len(common) < threshold:
            continue

        # Проверяем различающие признаки — если они конфликтуют, это разные люди
        archived_distinct = extract_distinctive_features(archived_appearance)
        if features_conflict(new_distinct, archived_distinct):
            print(f"  [дедупл-skip] '{c['name']}' отклонён: "
                  f"волосы {new_distinct['hair_colors']} vs {archived_distinct['hair_colors']}")
            continue

        if len(common) > best_score:
            best_score = len(common)
            best_match = (cid, c)

    return best_match


def extract_appearance_keywords(text: str) -> set:
    """Значимые слова внешности для сравнения (без стоп-слов)."""
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


# Различающие признаки — слова, которые при несовпадении сильно намекают
# что это разные персонажи
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
    """Извлекает признаки, по которым стоит различать персонажей."""
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    return {
        "hair_colors": words & HAIR_COLORS,
        "hair_styles": words & HAIR_STYLES,
    }


def features_conflict(a: dict, b: dict) -> bool:
    """
    True если у двух описаний есть конфликтующие признаки:
    оба упоминают цвет волос, и эти цвета не пересекаются.
    """
    # Цвет волос: если оба указаны и не пересекаются — конфликт
    if a["hair_colors"] and b["hair_colors"]:
        if not (a["hair_colors"] & b["hair_colors"]):
            return True
    return False


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
                    retries: int = 3,
                    errors: ErrorLog | None = None,
                    page_idx: int = 0) -> None:
    """
    Переводит все реплики страницы за один вызов LLM.
    Записывает результат в поле bubble["translation"] in-place.
    Баблы без текста пропускаются и получают пустую строку.
    Ошибки записываются в errors (если передан).
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
            if errors:
                errors.add(page_idx, "ocr_empty",
                           f"Бабл #{i+1} не имеет OCR-текста — нечего переводить",
                           bubble_idx=i+1,
                           bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})")

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

    raw = ""
    timed_out = False
    for attempt in range(retries):
        try:
            raw = ollama("gemma4:26b", prompt, timeout=600)
            break
        except requests.exceptions.ReadTimeout:
            print(f"     [timeout] попытка {attempt+1}/{retries}...")
            timed_out = True
            raw = ""

    if timed_out and not raw:
        if errors:
            errors.add(page_idx, "timeout",
                       f"Перевод страницы не удался — {retries} таймаутов подряд",
                       bubbles_affected=len(to_translate))

    results = parse_json_array(raw)
    if raw and not results:
        if errors:
            errors.add(page_idx, "json_parse",
                       "Модель вернула ответ, но JSON-массив не распарсился",
                       raw_response=raw,
                       expected_count=len(to_translate))

    expected = len(to_translate)
    got = len(results)
    if results and got != expected:
        if errors:
            errors.add(page_idx, "count_mismatch",
                       f"Модель вернула {got} переводов, ожидалось {expected}",
                       expected_count=expected,
                       got_count=got)

    # Сопоставляем по порядковому id (1-based) из ответа
    id_to_translation = {r["id"]: r.get("translation", "") for r in results if "id" in r}

    for seq, (bubble_idx, b) in enumerate(to_translate):
        translation = id_to_translation.get(seq + 1, "")
        source = "by_id"
        if not translation:
            # Fallback: берём по позиции если модель не вернула id
            if seq < len(results):
                translation = results[seq].get("translation", "")
                source = "by_position"
            else:
                translation = "[error]"
                source = "missing"

        if not translation or translation == "[error]":
            translation = "[error]"
            if errors:
                errors.add(page_idx, "translation_missing",
                           f"Бабл #{bubble_idx+1}: модель не вернула перевод",
                           bubble_idx=bubble_idx+1,
                           original_text=b.get("text", ""),
                           speaker=b.get("speaker", "?"),
                           lookup_method=source)

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


def preprocess_crop_minimal(img_cv: np.ndarray, x: int, y: int,
                            w: int, h: int) -> np.ndarray:
    """
    Мягкий препроцессинг для второго прохода: только лёгкое увеличение
    и небольшой паддинг, без CLAHE и без grayscale.
    Помогает на коротких репликах и SFX, которые agressive-препроцессинг
    может «съесть».
    """
    crop = img_cv[y:y+h, x:x+w]
    h2, w2 = crop.shape[:2]
    # Скромное увеличение ×2 вместо ×3
    crop = cv2.resize(crop, (w2 * 2, h2 * 2), interpolation=cv2.INTER_CUBIC)
    # Цветной с небольшим белым паддингом
    return cv2.copyMakeBorder(crop, 32, 32, 32, 32, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))


def ocr_region(img_cv: np.ndarray, x: int, y: int, w: int, h: int,
               idx: int, page_idx: int) -> str:
    """
    OCR с двухпроходной стратегией:
      1) агрессивный препроцессинг + строгий промпт ("read the text")
      2) если пусто → мягкий препроцессинг + общий промпт ("any characters")
    """
    # ── Проход 1: основной ──
    processed = preprocess_crop(img_cv, x, y, w, h)
    crop_path = os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{idx:02d}.png")
    cv2.imwrite(crop_path, processed)
    raw = ollama(
        "glm-ocr:latest",
        "Read and return ONLY the text in this manga speech bubble. No explanation.",
        crop_path,
        timeout=60,
    )
    cleaned = clean_text(raw)

    if cleaned and len(cleaned) >= 3:
        print(f"     [OCR ✓] bubble {idx}: {cleaned[:50]!r}")
        return cleaned

    # ── Проход 2: ретрай с мягким препроцессингом ──
    print(f"     [OCR retry] bubble {idx} — пробуем мягкий препроцессинг")
    processed2 = preprocess_crop_minimal(img_cv, x, y, w, h)
    crop_path2 = os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{idx:02d}_retry.png")
    cv2.imwrite(crop_path2, processed2)
    raw2 = ollama(
        "glm-ocr:latest",
        ("Read any text, letters, or characters visible in this image, "
         "including short sounds, exclamations, sound effects, or single words. "
         "Return ONLY what is written, no explanation."),
        crop_path2,
        timeout=60,
    )
    cleaned2 = clean_text(raw2)

    # Берём лучший из двух результатов
    final = cleaned2 if len(cleaned2) > len(cleaned) else cleaned

    if not final:
        print(f"     [OCR ✗ EMPTY] bubble {idx} (оба прохода пусты) → {crop_path}, {crop_path2}")
    elif len(final) < 3:
        print(f"     [OCR ? SHORT] bubble {idx}: {final!r} (после retry)")
    else:
        source = "retry" if final == cleaned2 and cleaned2 != cleaned else "primary"
        print(f"     [OCR ✓ {source}] bubble {idx}: {final[:50]!r}")

    return final


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


# ─── анализ оригинального текста (цвет, ориентация) ──────────────────────────

def _binarize_text_mask(crop_gray: np.ndarray) -> np.ndarray:
    """
    Возвращает маску где текст = 255, фон = 0.
    Использует Оцу + автоматическое определение полярности
    (тёмный текст на светлом vs светлый текст на тёмном).
    """
    _, binary = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Если белого больше — значит фон белый, текст чёрный, инвертируем
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def detect_text_color(img_cv: np.ndarray, bubble: dict,
                      default: tuple = (0, 0, 0)) -> tuple:
    """
    Определяет цвет текста в бабле.
    Берём пиксели где маска текста активна и считаем медианный цвет.
    Медиана устойчивее к выбросам (антиалиасинг по краям букв).
    """
    x, y, w, h = bubble["x"], bubble["y"], bubble["width"], bubble["height"]
    crop = img_cv[y:y+h, x:x+w]
    if crop.size == 0:
        return default

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _binarize_text_mask(gray)

    text_pixels = crop[mask > 0]
    if len(text_pixels) < 10:    # текста толком не нашлось
        return default

    # Медиана по каждому каналу (BGR → возвращаем как RGB для PIL)
    b, g, r = np.median(text_pixels, axis=0).astype(int)
    return (int(r), int(g), int(b))


# ─── инпейтинг ────────────────────────────────────────────────────────────────

def build_inpaint_mask(image_shape: tuple, bubbles: list[dict],
                        shrink: int = 1) -> np.ndarray:
    """
    Строит бинарную маску для инпейтинга: 255 = заменяемая область, 0 = сохранить.
    Маска СТРОГО внутри bbox каждого бабла (без расширения наружу) — иначе
    SD начинает «творить» по соседству и оставляет артефакты.

    shrink — на сколько пикселей сжать маску ВНУТРЬ от границы bbox.
    Небольшое сжатие защищает контур самого бабла от перерисовки.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bubbles:
        if not b.get("translation"):
            continue
        x, y, bw, bh = b["x"], b["y"], b["width"], b["height"]
        x0 = max(0, x + shrink)
        y0 = max(0, y + shrink)
        x1 = min(w, x + bw - shrink)
        y1 = min(h, y + bh - shrink)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _pad_to_multiple_of_8(img_np: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Дополняет numpy-массив белыми пикселями справа/снизу до размеров
    кратных 8 (LaMa требует это для свёрток).
    В отличие от resize, сохраняет ВСЕ оригинальные координаты пиксель в пиксель.
    Возвращает (padded, оригинальный_размер_HxW).
    """
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
    """
    Стирает оригинальный текст с помощью anime-big-lama (TorchScript).
    LaMa — чисто свёрточная модель, без «творчества» как у SD: она восстанавливает
    фон опираясь на окружающие пиксели и не пытается ничего «дорисовать».

    Стратегия:
      1. Маска СТРОГО внутри bbox (shrink=1) — границы бабла защищены
      2. Падим до /8 (LaMa требует), потом обрезаем
      3. Композитинг: берём из LaMa только пиксели под маской

    Fallback: cv2.inpaint при ошибке LaMa или если модель не загрузилась.
    """
    mask_np = build_inpaint_mask(img_cv.shape, bubbles)

    if not mask_np.any():
        return img_cv.copy()    # инпейтить нечего

    if inpaint_model is None:
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)

    try:
        # 1. Готовим вход для LaMa
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_padded, (orig_h, orig_w) = _pad_to_multiple_of_8(img_rgb)
        mask_padded, _ = _pad_to_multiple_of_8(mask_np)

        # Image:  [1, 3, H, W] в диапазоне [0, 1]
        # Mask:   [1, 1, H, W] в диапазоне [0, 1] (1.0 = инпейтить)
        img_tensor = torch.from_numpy(img_padded).float().div(255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(mask_padded).float().div(255.0)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

        # 2. Прогон через LaMa
        with torch.no_grad():
            result_tensor = inpaint_model(img_tensor, mask_tensor)

        # 3. Конвертация обратно в numpy
        result_np = result_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result_np = (result_np * 255).astype(np.uint8)
        # Обрезаем падинг
        result_np = result_np[:orig_h, :orig_w]
        result_cv = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        # 4. Композитинг по маске: гарантия что вне bbox ничего не изменилось
        out = img_cv.copy()
        mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR) // 255
        out = out * (1 - mask_3ch) + result_cv * mask_3ch
        return out.astype(np.uint8)

    except Exception as e:
        print(f"  [inpaint ⚠] LaMa не справилась ({e}), fallback на cv2.inpaint")
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)


# ─── отрисовка ────────────────────────────────────────────────────────────────

def fit_text_in_box(draw: ImageDraw.Draw, text: str, box_w: int,
                    box_h: int, font_path: str | None = None) -> tuple:
    """
    Подбирает максимальный размер шрифта бинарным поиском (O(log n) вместо O(n)).
    Возвращает (font, lines, line_heights, spacing).
    """
    if font_path is None:
        font_path = DEFAULT_FONT
    padding = 4
    usable_w = box_w - padding * 2
    usable_h = box_h - padding * 2

    def try_load_font(size: int):
        if font_path is None:
            return None
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


def _render_text_block(text: str, box_w: int, box_h: int,
                        color: tuple, font_path: str | None = None) -> Image.Image:
    """
    Рендерит текст в прозрачный RGBA-блок размером box_w × box_h.
    Возвращает PIL.Image — её можно вращать и накладывать на инпейтнутую страницу.
    """
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font, lines, line_heights, spacing = fit_text_in_box(
        draw, text, box_w, box_h, font_path
    )
    padding = 4
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    text_y = padding + max(0, (box_h - padding * 2 - total_h) // 2)

    rgba_color = (color[0], color[1], color[2], 255)
    for j, line in enumerate(lines):
        bb = draw.textbbox((0, 0), line, font=font)
        line_w = bb[2] - bb[0]
        text_x = padding + max(0, (box_w - padding * 2 - line_w) // 2)
        text_x = min(text_x, box_w - padding - line_w)
        draw.text((text_x, text_y), line, fill=rgba_color, font=font)
        text_y += line_heights[j] + spacing

    return img


def draw_results(img_cv: np.ndarray, bubbles: list[dict],
                 debug: bool = False) -> np.ndarray:
    """
    Заливает оригинальный текст через LaMa-инпейтинг, затем рисует переводы
    с цветом оригинала. Текст рисуется горизонтально (поворот не используется —
    для большинства баблов японский текст должен читаться слева-направо после перевода).

    debug=False (по умолчанию): чистая страница, только заменённый текст.
    debug=True: дополнительно цветные рамки и номера баблов для диагностики.
    """
    # 1. Определяем цвет текста ДО того как сотрём оригинал
    for b in bubbles:
        if b.get("translation"):
            b["_text_color"] = detect_text_color(img_cv, b)

    # 2. Инпейнтим оригинальный текст
    print("  Инпейтинг оригинального текста (LaMa)...")
    inpainted = inpaint_page(img_cv, bubbles)

    # 3. Рендерим переводы поверх инпейтнутой страницы
    pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)).convert("RGBA")

    for b in bubbles:
        translation = b.get("translation", "")
        if not translation:
            continue
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        color = b.get("_text_color", (0, 0, 0))
        block = _render_text_block(translation, w, h, color)
        pil.paste(block, (x, y), block)

    # 4. Debug-оверлей рисуем поверх — рамки и номера
    if debug:
        draw = ImageDraw.Draw(pil)
        try:
            font_small = ImageFont.truetype(DEFAULT_FONT, 14)
        except OSError:
            font_small = ImageFont.load_default()

        for i, b in enumerate(bubbles):
            x, y, w, h = b["x"], b["y"], b["width"], b["height"]
            translation = b.get("translation", "")
            raw_text = b.get("text", "")

            if not raw_text:
                color = (255, 0, 0)        # OCR пустой
            elif not translation:
                color = (255, 140, 0)      # есть текст, нет перевода
            elif b["class"] == "text_bubble":
                color = (0, 200, 0)
            else:
                color = (0, 150, 255)

            draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=2)
            draw.rectangle([(x, y-22), (x+18, y)], fill=color)
            draw.text((x+3, y-20), str(i+1), fill=(0, 0, 0), font=font_small)

    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# ─── обработка страницы ───────────────────────────────────────────────────────

def reading_order(bubble: dict) -> tuple:
    """Сортировка баблов: сверху вниз, справа налево (японский порядок)."""
    return (bubble["y"] // 150, -bubble["x"])


def process_page(image_path: str, page_idx: int,
                 manga_ctx: MangaContext, archive: CharacterArchive,
                 output_path: str, target_lang: str = "Russian",
                 debug: bool = False,
                 errors: ErrorLog | None = None):
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

    if not text_bubbles and errors:
        errors.add(page_idx, "no_bubbles",
                   "На странице не найдено ни одного бабла с текстом",
                   image=image_path)

    # Эвристика: мало баблов → возможно это галерея представлений.
    # Проверка стоит ~1 vision-вызов, но на intro-странице мы сэкономим
    # обычный analyze_page_full и сразу получим правильные имена в архив.
    if len(text_bubbles) <= 2:
        print("\n── Проверка: галерея персонажей? ──")
        if detect_character_intro_page(image_path):
            print("\n── Извлечение представлений ──")
            characters_context = extract_character_intros(image_path, archive, page_idx)
            print(characters_context)

            # На галереях нечего переводить, но обновим сюжет и сохраним
            # картинку без изменений
            manga_ctx.update("Character introduction page.")
            cv2.imwrite(output_path, img_cv)
            print(f"\nСохранено (без изменений): {output_path}")
            print(f"  ⓘ страница представления персонажей — перевод не требуется")
            return text_bubbles

    # 2. OCR
    print("\n── OCR ──")
    for i, b in enumerate(text_bubbles):
        b["text"] = ocr_region(img_cv, b["x"], b["y"], b["width"], b["height"],
                               idx=i + 1, page_idx=page_idx)
        # Логируем пустые OCR как warning — иногда это правда пустые баблы
        # (только звукоподражания), иногда баг
        if not b["text"] and errors:
            errors.add(page_idx, "ocr_empty",
                       f"Бабл #{i+1}: OCR не вернул текста после двух проходов",
                       bubble_idx=i+1,
                       bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})",
                       crop=f"crops/p{page_idx:03d}_bubble_{i+1:02d}.png")

    # 3-4. Совмещённый анализ персонажей + сцены (один vision-вызов)
    print("\n── Анализ страницы ──")
    characters_context, page_context, page_summary = analyze_page_full(
        image_path, archive, manga_ctx, page_idx
    )
    print(characters_context)
    print(f"  context: {page_context}")
    print(f"  summary: {page_summary}")

    if characters_context == "CHARACTERS ON THIS PAGE: unknown" and errors:
        errors.add(page_idx, "character_parse_failed",
                   "Анализ персонажей не дал результата — JSON не распарсился",
                   raw_response_snippet=characters_context)

    # 5. Атрибуция реплик
    print("\n── Атрибуция ──")
    text_bubbles = attribute_bubbles(
        image_path, text_bubbles, page_context, characters_context, archive
    )
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('speaker', '?')} ({b.get('gender', '?')}): "
              f"{b.get('text', '')[:40]}")
        if b.get("text") and b.get("speaker") == "unknown" and errors:
            errors.add(page_idx, "speaker_unknown",
                       f"Бабл #{i+1}: не удалось определить говорящего",
                       bubble_idx=i+1,
                       text=b.get("text", "")[:100])

    # 6. Перевод — один вызов LLM на всю страницу
    print("\n── Перевод ──")
    translate_batch(text_bubbles, page_context, manga_ctx, target_lang,
                    errors=errors, page_idx=page_idx)
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('text', '')[:25]} → {b.get('translation', '')[:40]}")

    # 7. Обновляем сюжетный контекст
    manga_ctx.update(page_summary)

    # 8. Сохраняем аннотированное изображение
    annotated = draw_results(img_cv, text_bubbles, debug=debug)
    cv2.imwrite(output_path, annotated)

    # Сводка по странице
    empty_ocr = sum(1 for b in text_bubbles if not b.get("text"))
    no_translation = sum(1 for b in text_bubbles
                          if b.get("text") and not b.get("translation"))
    err_translation = sum(1 for b in text_bubbles
                          if b.get("translation") == "[error]")
    ok = len(text_bubbles) - empty_ocr - no_translation - err_translation
    print(f"\nСохранено: {output_path}")
    print(f"  ✓ переведено: {ok} | ⚠ без перевода: {no_translation} | "
          f"✗ OCR пуст: {empty_ocr} | ✗ ошибка перевода: {err_translation}")
    return text_bubbles


# ─── обработка директории ─────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Форматирует длительность как '1ч 23м 45с' / '23м 45с' / '45.3с'."""
    if seconds < 60:
        return f"{seconds:.1f}с"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m}м {s}с"
    return f"{m}м {s}с"


def process_directory(input_dir: str, output_dir: str = RESULTS_DIR,
                      target_lang: str = "Russian",
                      font_path: str = "arial.ttf",
                      debug: bool = False,
                      error_log_path: str = "errors.log"):
    """
    font_path — путь или имя файла шрифта для рендера переводов.
    Если в переводах ожидаются CJK-символы (оригинальные имена в скобках),
    укажите шрифт с их поддержкой, например:
      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc" (Linux)
      "C:/Windows/Fonts/YuGothM.ttc" (Windows)
      "/System/Library/Fonts/Hiragino Sans GB.ttc" (macOS)
    debug — если True, на финальной картинке рисуются цветные рамки
    и номера баблов для диагностики (OCR/перевод).
    error_log_path — куда писать журнал проблем за прогон.
    """
    # Проверяем что шрифт грузится — иначе CJK-символы будут квадратиками
    try:
        ImageFont.truetype(font_path, 12)
        global DEFAULT_FONT
        DEFAULT_FONT = font_path
        print(f"[font] Используется: {font_path}")
    except OSError:
        print(f"[font] ⚠ Шрифт '{font_path}' не найден — fallback на arial.ttf. "
              f"Для CJK-символов установите Noto Sans CJK и передайте его путь.")

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
    errors = ErrorLog(error_log_path)

    total_start = time.perf_counter()
    page_times: list[float] = []
    failed = 0

    for page_idx, filename in enumerate(files, start=1):
        input_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}_translated.png")

        page_start = time.perf_counter()
        try:
            process_page(input_path, page_idx, manga_ctx, archive,
                         output_path, target_lang, debug=debug, errors=errors)
            elapsed = time.perf_counter() - page_start
            page_times.append(elapsed)
            print(f"  ⏱  страница обработана за {format_duration(elapsed)}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()
            errors.add(page_idx, "page_failed",
                       f"Страница не обработана: {type(e).__name__}: {e}",
                       filename=filename,
                       traceback=traceback.format_exc())

    total_elapsed = time.perf_counter() - total_start

    # Сохраняем журнал ошибок
    errors.save()

    print(f"\n{'='*60}")
    print(f"Готово!")
    print(f"  Страниц обработано: {len(page_times)} / {len(files)}")
    if failed:
        print(f"  Ошибок:             {failed}")
    print(f"  Общее время:        {format_duration(total_elapsed)}")
    if page_times:
        avg = sum(page_times) / len(page_times)
        print(f"  Среднее на страницу: {format_duration(avg)}")
        print(f"  Самая быстрая:      {format_duration(min(page_times))}")
        print(f"  Самая медленная:    {format_duration(max(page_times))}")
    print(f"  Результаты:         {output_dir}")
    print(f"  Архив персонажей:   characters.json ({len(archive.characters)} персонажей)")

    # Сводка по журналу ошибок
    if errors.entries:
        print(f"\n  ⚠ Зафиксировано проблем: {len(errors.entries)}")
        for kind, count in sorted(errors.summary().items(),
                                    key=lambda x: -x[1]):
            print(f"     {kind:30s} {count}")
        print(f"  Подробности:        {error_log_path}")
    else:
        print(f"  ✓ Проблем не зафиксировано")


if __name__ == "__main__":
    process_directory(
        input_dir="input",
        output_dir="results",
        target_lang="Russian",
        # Для японских имён в скобках укажите CJK-шрифт, например:
        #   font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        #   font_path="C:/Windows/Fonts/YuGothM.ttc"
        font_path="arial.ttf",
        # debug=True — рисует рамки и номера баблов для диагностики
        debug=False,
        # куда писать журнал проблем
        error_log_path="errors.log",
    )