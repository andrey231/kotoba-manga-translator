"""
manga_translator.py
Kotoba — manga translator with character memory.

Автоматический перевод манги с помощью локальных LLM через Ollama.
Главное отличие от других open-source решений — накопление архива
персонажей и контекста сцены между страницами, что даёт более
согласованный перевод (правильные имена, грамматический род, тон речи).

Пайплайн на страницу:
  1. Детекция баблов     — /v2
  2. OCR                 — glm-ocr через Ollama
  3. Анализ персонажей   — vision LLM + CharacterArchive
  4. Анализ сцены        — vision LLM + MangaContext
  5. Атрибуция реплик    — vision LLM (кто говорит + пол)
  6. Перевод             — текстовый LLM с учётом контекста
  7. SAM2-сегментация    — точные маски текстовых пикселей
  8. Отрисовка и запись  — PIL, OpenCV (инпейтинг через LaMa)
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
CROPS_DIR: str = "crops"   # переопределяется web.py под каждый job_id
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

BUBBLE_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
BUBBLE_CLASSES = {0: "bubble", 1: "text_bubble", 2: "text_free"}

# Vision/LLM модель Ollama для анализа сцены, атрибуции, перевода.
# Должна быть мультимодальной. Переопределяется через
# process_directory(llm_model=...).
LLM_MODEL = "gemma4:26b"

# anime-big-lama — модель для инпейтинга, заточенная под мангу.
# Список репозиториев пробуется по порядку: первый — официальное место,
# второй — зеркало df1412 на случай если первый недоступен.
LAMA_REPOS = [
    ("deckyfx/anime-big-lama", "anime-manga-big-lama.pt"),
    ("df1412/anime-big-lama",  "anime-manga-big-lama.pt"),
]

DEFAULT_FONT = "arial.ttf"   # переопределяется через process_directory(font_path=...)

# Паттерны мета-рассуждений модели вместо перевода.
# Используются в _translate_chunk и _translate_persistent для отбраковки ответов.
_META_MARKERS: tuple[str, ...] = (
    "depending on", "it could be", "it can be translated",
    "note:", "please note", "keep in mind",
    "в зависимости", "можно перевести", "можно оставить",
    "это можно", "стоит отметить", "следует отметить",
    "примечание:", "обратите внимание",
)



# ─── инициализация моделей ────────────────────────────────────────────────────

def load_inpainting_model():
    """
    Загружает anime-big-lama (TorchScript) с HuggingFace.
    Пробует несколько репозиториев по очереди.
    """
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
    processor = AutoImageProcessor.from_pretrained(BUBBLE_MODEL_ID)
    model = RTDetrV2ForObjectDetection.from_pretrained(BUBBLE_MODEL_ID)
    model.eval()
    return processor, model

inpaint_model = load_inpainting_model()
detector_processor, detector_model = load_detector()

# SAM predictor — lazy-loaded on first use.
# SAM2 ломается в embedded Python (Hydra вызывает inspect.getsource на C-функциях).
# Порядок попыток:
# ─── Comic Text Detector (ONNX) ───────────────────────────────────────────────
# Специализированная модель для манги/комиксов — работает на полной странице,
# выдаёт попиксельную маску текста. Устанавливается через onnxruntime (PyPI),
# веса (~30 MB) скачиваются с HuggingFace при первом запуске.
_ctd_session = None
_CTD_MODEL_ID   = "mayocream/comic-text-detector-onnx"
_CTD_MODEL_FILE = "comic-text-detector.onnx"
_CTD_INPUT_SIZE = 1024


def _get_ctd_session():
    """Загружает ONNX-сессию Comic Text Detector. Возвращает None если недоступна."""
    global _ctd_session
    if _ctd_session is not None:
        return _ctd_session if _ctd_session is not False else None

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
    """
    Запускает CTD на полной странице.
    Возвращает uint8-маску того же размера (текст=255, фон=0) или None.
    """
    sess = _get_ctd_session()
    if sess is None:
        return None

    h, w = img_cv.shape[:2]
    s = _CTD_INPUT_SIZE

    # Letterbox: масштаб сохраняя пропорции, паддинг до квадрата
    scale = s / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_cv, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((s, s, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized.astype(np.float32) / 255.0

    # NCHW float32
    inp = canvas.transpose(2, 0, 1)[np.newaxis]

    try:
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: inp})
    except Exception as e:
        print(f"[ctd] inference error: {e}")
        return None

    # Ищем выход с пространственными размерами ≥ s/2
    mask_raw = None
    for out in outputs:
        if out.ndim >= 2 and min(out.shape[-2:]) >= s // 2:
            mask_raw = out
            break
    if mask_raw is None:
        mask_raw = max(outputs, key=lambda o: o.size)

    m = mask_raw.squeeze()
    if m.ndim == 3:
        # [C, H, W] — если 2 канала: берём текстовый (1-й), иначе 0-й
        m = m[1] if m.shape[0] == 2 else m[0]

    # Нормализуем к [0,255] uint8
    if m.max() <= 1.0:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255

    # Убираем паддинг и возвращаем к исходному размеру
    m_crop = m[:nh, :nw]
    return cv2.resize(m_crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


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
        # Пропускаем пустые строки (например в fast_mode где анализ не делается),
        # иначе они засоряли бы to_prompt() пустыми строками.
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
            f.write(f"# Error log — {len(self.entries)} entries total\n")
            f.write(f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Group by page for easier reading
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
                            # Truncate long values nicely
                            v_str = str(v)
                            if len(v_str) > 300:
                                v_str = v_str[:300] + "... [truncated]"
                            f.write(f"    {k}: {v_str}\n")


# ─── вспомогательные функции ──────────────────────────────────────────────────

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_text(text: str) -> str:
    """Убирает markdown-артефакты и лишние пробелы, сохраняя переносы строк."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`+", "", text)
    # LaTeX circled numbers → Unicode: $\textcircled{2}$ → ②
    def _circled(m):
        n = int(m.group(1))
        if 1 <= n <= 20:
            return chr(0x2460 + n - 1)
        return m.group(0)
    text = re.sub(r"\$\s*\\?textcircled\{(\d+)\}\s*\$", _circled, text)
    # Схлопываем пробелы/табы внутри строки, но НЕ переносы строк
    text = re.sub(r"[^\S\n]+", " ", text)
    # Убираем лишние пустые строки (3+ подряд → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Глобальный флаг подробного логирования всех LLM-запросов.
# Когда True — каждый вызов ollama() пишет в stdout полный prompt и ответ.
# Включается через env var OLLAMA_DEBUG=1 или прямой установкой DEBUG_LLM = True.
DEBUG_LLM = os.environ.get("OLLAMA_DEBUG", "").strip() in ("1", "true", "yes", "on")
# Когда задан — SAM debug-изображения сохраняются в эту папку (устанавливается web.py)
SAM_DEBUG_DIR: str | None = None
# Счётчик вызовов — чтобы в логе было видно номер запроса
_OLLAMA_CALL_NUM = 0


def _log_llm_call(call_num: int, model: str, prompt: str, response: str,
                   opts: dict, has_image: bool) -> None:
    """Подробный лог одного запроса к Ollama. Активен только при DEBUG_LLM."""
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


# Список fallback моделей: если текущая LLM_MODEL возвращает пустоту,
# пробуем эти по очереди. Помогает с abliterated/нестабильными моделями.
LLM_FALLBACK_MODELS: list[str] = []   # заполняется при инициализации/process_directory


def _discover_fallback_models() -> list[str]:
    """
    Пытается узнать у Ollama какие модели установлены, и формирует список
    fallback-моделей в порядке предпочтения:
      1. Текущая LLM_MODEL — НЕ включаем (это уже сломалось)
      2. Известные стабильные модели (gemma4:26b — обычно работает лучше abliterated)
      3. Любые остальные мультимодальные/LLM модели
    """
    preferred = ["gemma4:26b", "gemma3:27b", "gemma3:12b", "llava:13b",
                 "qwen2.5-vl:7b", "minicpm-v"]
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return []
    fallbacks = [m for m in preferred
                  if m in installed and m != LLM_MODEL]
    # Также добавляем любые другие установленные модели не из списка
    others = sorted(installed - set(preferred) - {LLM_MODEL})
    fallbacks.extend(others)
    return fallbacks


def ollama(model_name: str, prompt: str, image_path: str = None,
           timeout: int = 800, num_predict: int = 6000,
           temperature: float = 0.1) -> str:
    """
    Вызов Ollama. Если первый вызов вернул пустую строку, делает 1 retry
    с другим seed. Если и это не помогло — пробует fallback-модели.

    Замечание: при варьировании параметров между запросами Ollama может
    застрять в edge case KV cache (видно эмпирически на gemma4 + малых
    промптах после больших). Простой retry с другим seed обычно лечит,
    дальнейшие retries только теряют время.
    """
    global _OLLAMA_CALL_NUM
    import random

    def _call(opts: dict, label: str, model: str) -> str:
        global _OLLAMA_CALL_NUM
        _OLLAMA_CALL_NUM += 1
        call_num = _OLLAMA_CALL_NUM
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if opts:
            payload["options"] = opts
        if image_path:
            payload["images"] = [image_to_base64(image_path)]
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response = r.json().get("response", "").strip()
        if DEBUG_LLM:
            _log_llm_call(call_num, model, prompt, response,
                           opts, bool(image_path))
        return response

    # Попытка 1: запрошенные параметры, текущая модель
    response = _call(
        {"temperature": temperature, "num_predict": num_predict},
        "primary", model_name,
    )
    if response:
        return response

    # Попытка 2: тот же запрос с другим seed — основное лекарство от
    # пустого ответа на gemma4.
    response = _call(
        {"temperature": temperature, "num_predict": num_predict,
         "seed": random.randint(1, 100000)},
        "retry-seed", model_name,
    )
    if response:
        print(f"     [ollama-retry-ok] recovered with new seed")
        return response

    # Попытка 3: пробуем fallback-модели. Это помогает когда текущая модель
    # (особенно abliterated-варианты) имеет поломанные template-теги и
    # возвращает технически пустоту даже когда генерирует токены.
    if not LLM_FALLBACK_MODELS:
        return ""

    for fb_model in LLM_FALLBACK_MODELS:
        print(f"     [ollama-fallback] trying alternate model: {fb_model}")
        try:
            response = _call(
                {"temperature": temperature, "num_predict": num_predict},
                f"fallback-{fb_model}", fb_model,
            )
        except Exception as e:
            print(f"     [ollama-fallback] {fb_model} error: {e}")
            continue
        if response:
            print(f"     [ollama-retry-ok] recovered with fallback model {fb_model}")
            return response
    return ""

def parse_json_array(text: str) -> list:
    """
    Извлекает первый JSON-массив из произвольного текста.

    Делает несколько проходов:
      1. Убирает markdown-фенсы ```json ... ```
      2. Ищет первый `[` и собирает строку до парного `]` со счётом скобок
         (с учётом строк, чтобы не ломаться на `]` внутри текста перевода)
      3. Если парсинг падает — пробует "ремонт": убирает trailing-запятые
         и попытку обрезать незакрытые элементы (когда модель оборвалась)
    """
    if not text:
        return []

    # 1. Снимаем фенсы
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    # 2. Сканируем баланс скобок
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
        # Массив не закрылся — модель оборвалась.
        # Берём всё что есть, попробуем починить ниже.
        candidate = cleaned[start:]
    else:
        candidate = cleaned[start:end]

    # 3. Прямой парсинг
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 4. Ремонт: убираем висящие запятые и закрываем оборванный конец
    repaired = re.sub(r",\s*([\]\}])", r"\1", candidate)
    # Если строка закончилась посреди элемента — пробуем обрезать до последнего полного `}`
    last_close = repaired.rfind("}")
    if last_close > 0:
        repaired = repaired[: last_close + 1] + "]"
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
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
    """
    Извлекает с галереи представлений пары «персонаж → имя + описание»
    и записывает их в архив с in-image именами.
    Возвращает characters_context для последующих этапов пайплайна.
    """
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

    # --- разбираем блок CHARACTERS ---
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
            print(f"  [dedup-skip] '{c['name']}' rejected: "
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
                           f"Bubble #{i+1} has no OCR text — nothing to translate",
                           bubble_idx=i+1,
                           bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})")

    if not to_translate:
        return

    # Стратегия:
    # 1. Все страницы (даже с 1 баблом) идут через batch-формат — он стабильнее
    #    чем `_translate_single` при коротких репликах
    # 2. Чанки по 5 баблов, каждый чанк — отдельный batch-вызов
    # 3. Для непереведённых после batch — retry через _translate_single
    # 4. Для упрямо непереводящихся — минималистичный промпт без контекста

    missing_indices = []  # глобальные индексы в to_translate, которые не перевелись

    CHUNK_SIZE = 5
    chunks = [to_translate[i:i + CHUNK_SIZE]
              for i in range(0, len(to_translate), CHUNK_SIZE)]
    if len(chunks) > 1:
        print(f"     [batch] Splitting {len(to_translate)} bubbles into "
              f"{len(chunks)} chunks of up to {CHUNK_SIZE}")
    for chunk_idx, chunk in enumerate(chunks):
        _translate_chunk(
            chunk, chunk_idx, to_translate, missing_indices,
            page_context, manga_ctx, target_lang, gender_hints,
            errors, page_idx, retries,
        )

    # Второй проход — добиваем недостающие "настойчивым" переводом
    # по одному баблу. Делаем несколько попыток с разными промптами,
    # температурами и seed'ами пока что-нибудь не сработает.
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


def _translate_chunk(chunk: list, chunk_idx: int, to_translate: list,
                     missing_indices: list, page_context: str,
                     manga_ctx: MangaContext, target_lang: str,
                     gender_hints: dict, errors, page_idx: int,
                     retries: int) -> None:
    """
    Переводит один чанк баблов. Записывает результаты прямо в bubble["translation"],
    индексы непереведённых баблов добавляет в missing_indices.

    Для очень маленьких чанков (1-2 бабла) добавляются фейковые "padding"-записи
    чтобы общий промпт был достаточно большим — gemma4 иногда возвращает пустую
    строку на очень короткие промпты, и расширение помогает.
    """
    # Локальная нумерация в чанке (1-based), но запоминаем глобальный индекс
    local_to_global = {i + 1: chunk[i][0] for i in range(len(chunk))}

    # Padding для совсем маленьких чанков. Промпт становится "весомее"
    # и модель надёжнее возвращает структурированный JSON.
    padding_entries = []
    if len(chunk) < 3:
        padding_samples = [
            ("Narrator", "neutral", "The story continues."),
            ("Narrator", "neutral", "A new scene begins."),
            ("Narrator", "neutral", "The page turns."),
        ]
        need = 3 - len(chunk)
        for i in range(need):
            spk, gen, txt = padding_samples[i]
            padding_entries.append({
                "id": len(chunk) + i + 1,
                "speaker": spk,
                "gender": gen,
                "text": txt,
                "_padding": True,   # маркер — потом мы это отфильтруем
            })

    all_entries = [
        {
            "id": i + 1,
            "speaker": b["speaker"],
            "gender": gender_hints.get(b["gender"], "gender unknown"),
            "text": b["text"],
        }
        for i, (_, b) in enumerate(chunk)
    ] + [{k: v for k, v in e.items() if k != "_padding"} for e in padding_entries]

    inputs_json = json.dumps(all_entries, ensure_ascii=False, indent=2)

    is_cjk_or_cyrillic = target_lang.lower() in (
        "russian", "japanese", "chinese", "korean",
        "ukrainian", "bulgarian", "serbian"
    )
    per_bubble = 500 if is_cjk_or_cyrillic else 250
    total = len(all_entries)
    # Используем фиксированный num_predict=6000 (тот же что и в vision-вызовах
    # в analyze_page_full и attribute_bubbles). Если менять num_predict между
    # запросами, Ollama иногда застревает в KV cache mismatch и возвращает
    # пустоту. Большой запас не вредит — модель остановится естественно.
    dyn_predict = 6000

    n = total   # включая padding, чтобы модель видела согласованное число
    prompt = f"""You are translating manga content to {target_lang}.

{manga_ctx.to_prompt()}

PAGE CONTEXT:
{page_context}

You will receive a JSON array of {n} text bubbles. For EACH bubble translate
the ENTIRE text content:
- Dialogue: match the speaker's personality and gender, preserve emotion and register.
- Sound effects (PANT, HUFF, TCH, AHH, EEK, etc.): produce a natural equivalent in {target_lang}.
- Announcements, credits, cast/staff lists, copyright notices: translate ALL lines
  faithfully — do NOT summarize, omit, or shorten. Keep proper names (people,
  companies, characters) unchanged. Only translate structural labels
  (STAFF→ПЕРСОНАЛ, CAST→В РОЛЯХ, etc.) if translating to {target_lang}.
- Only filter genuine OCR noise: isolated stray characters with no meaning
  (e.g. a lone "·" or "|"). Never discard recognizable words or names.

INPUT (JSON array of {n} bubbles):
{inputs_json}

CRITICAL OUTPUT RULES:
1. Return EXACTLY {n} translations, one per input bubble, in the SAME order.
2. Output ONLY a JSON array, no prose before/after, no markdown fences, no comments.
3. Each item must have "id" (matching input id) and "translation" (string).
4. The "translation" field must contain ONLY the translated text — no explanations,
   no notes about your approach, no commentary.
5. Even for sound effects or single-word lines, produce a translation —
   never return empty string or skip an item.

OUTPUT (JSON array of {n} items):"""

    raw = ""
    for attempt in range(retries):
        try:
            raw = ollama(LLM_MODEL, prompt, timeout=600, num_predict=dyn_predict)
            break
        except requests.exceptions.ReadTimeout:
            print(f"     [timeout] chunk {chunk_idx+1}, retry {attempt+1}/{retries}...")
            raw = ""

    if not raw:
        # Все попытки провалились — все баблы чанка идут в missing
        for local_id, global_idx in local_to_global.items():
            missing_indices.append(global_idx)
        if errors:
            errors.add(page_idx, "timeout",
                       f"Chunk {chunk_idx+1} translation failed — all timeouts",
                       bubbles_affected=len(chunk))
        return

    results = parse_json_array(raw)
    if not results:
        # JSON не распарсился
        if errors:
            errors.add(page_idx, "json_parse",
                       f"Chunk {chunk_idx+1}: model response did not parse",
                       raw_response=raw[:500],
                       expected_count=len(chunk))
        for local_id, global_idx in local_to_global.items():
            missing_indices.append(global_idx)
        return

    expected = len(chunk)  # реальное число баблов, без padding
    got_real = sum(1 for r in results
                    if isinstance(r, dict) and r.get("id") and r.get("id") <= expected)
    if got_real != expected:
        print(f"     [chunk {chunk_idx+1}] got {got_real}/{expected} real translations "
              f"({len(results)} total in response, expected {len(all_entries)})")
        if errors:
            errors.add(page_idx, "count_mismatch",
                       f"Chunk {chunk_idx+1}: model returned {got_real} real translations, expected {expected}",
                       expected_count=expected,
                       got_count=got_real)

    id_to_translation = {r.get("id"): r.get("translation", "")
                          for r in results if isinstance(r, dict)}

    # Итерируемся только по реальным баблам (1..len(chunk)), padding игнорируем
    for local_id in range(1, len(chunk) + 1):
        global_idx = local_to_global[local_id]
        _, b = to_translate[global_idx]
        translation = id_to_translation.get(local_id, "")
        if not translation and local_id - 1 < len(results):
            # Fallback by position
            r = results[local_id - 1]
            if isinstance(r, dict):
                translation = r.get("translation", "")
        if translation and translation.strip():
            t = translation.strip()
            if any(t.lower().startswith(m) for m in _META_MARKERS):
                missing_indices.append(global_idx)  # отправляем в persistent fallback
            else:
                b["translation"] = t
        else:
            missing_indices.append(global_idx)


def _translate_persistent(bubble: dict, page_context: str,
                          manga_ctx: MangaContext, target_lang: str,
                          gender_hints: dict) -> str:
    """
    Настойчиво пытается перевести одну реплику.

    Использует 3 разные стратегии промптов в порядке убывания сложности.
    Каждая стратегия — один вызов LLM (плюс автоматический retry внутри
    ollama() при пустом ответе). Возвращает первый непустой результат.

    Цель — быстро попробовать радикально разные формулировки промпта.
    Если все 3 стратегии не помогли — возвращает "", помечаем как [error].
    Длинные циклы по 21+ попыткам делали обработку невыносимо медленной.
    """
    text = bubble.get("text", "").strip()
    if not text:
        return ""

    speaker = bubble.get("speaker", "unknown")
    gender = bubble.get("gender", "unknown")
    gender_h = gender_hints.get(gender, "gender unknown")

    # 3 стратегии — от самой структурированной до самой минималистичной.
    # Эмпирически: если ни одна не сработает, дополнительные попытки тоже
    # не помогут (модель действительно застряла), только теряем время.
    strategies = [
        # 1. С минимальным контекстом — speaker + текст
        ("ctx-speaker", lambda: (
            f"Translate this manga text to {target_lang}.\n"
            f"Speaker: {speaker} ({gender_h}).\n"
            f"Source: {text}\n\n"
            f"Rules: keep proper nouns, titles, and names unchanged. "
            f"Output ONLY the translated text. No quotes. No explanations."
        )),
        # 2. UI-style — как пользователь написал бы в Ollama UI
        ("ui-style", lambda: (
            f"Переведи на русский (только перевод, без пояснений): {text}"
            if target_lang.lower() == "russian"
            else f"Translate to {target_lang} (output translation only): {text}"
        )),
        # 3. Совсем минимально — последний шанс
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
            # Используем тот же num_predict=6000 что и в batch — унифицирует
            # параметры между всеми вызовами, избегаем KV cache edge case
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
        # Если цель — кириллический язык, но в ответе вообще нет кириллицы —
        # это мета-объяснение на английском вместо перевода.
        # Проверяем именно ОТСУТСТВИЕ кириллицы (< 5 символов), а не избыток латиницы:
        # перевод с именами собственными может содержать много латинских символов
        print(f"     [persistent ok @ attempt {attempt} / {strat_label}]")
        return cleaned

    print(f"     [persistent fail] '{text[:40]}' — all 3 strategies failed")
    return ""


def _clean_translation(raw: str) -> str:
    """
    Чистит ответ модели от типичного мусора:
    - markdown-фенсы и markdown-форматирование (**bold**, *italic*)
    - окружающие кавычки
    - префиксы "Translation:" / "Перевод:" / "Вот перевод:" и т.п.
    Если первая строка — мета-комментарий (Вот перевод..., Here is...), она
    отбрасывается, а весь остаток сохраняется (нужно для многострочных
    кредит-листов). Для однострочных диалогов возвращается первая строка
    оставшегося текста — т.к. модель обычно добавляет пояснения после.
    """
    result = raw.strip()
    if result.startswith("```"):
        result = re.sub(r"^```[a-zA-Z]*\s*", "", result)
        result = result.rstrip("`").strip()
    result = result.strip('"').strip("'").strip()

    # Уберём типичные label-префиксы в начале
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

    # Если первая строка — мета-комментарий, пропускаем её
    _meta_intro = ("вот ", "here is", "below is", "please note", "note:")
    first_low = lines[0].lower()
    if (any(first_low.startswith(m) for m in _meta_intro)
            or any(first_low.startswith(m) for m in _META_MARKERS)):
        lines = lines[1:]
        if not lines:
            return ""

    # Убираем markdown-форматирование (**bold**, *italic*, ***bold-italic***)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", cleaned)

    # Для многострочного контента (кредиты, анонсы) возвращаем весь текст.
    # Для однострочного — берём первую непустую строку (не захватываем пояснения после).
    stripped_lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    if len(stripped_lines) > 1:
        return cleaned.strip()
    first_line = stripped_lines[0] if stripped_lines else ""
    return first_line.strip('"').strip("'").strip()


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
    os.makedirs(CROPS_DIR, exist_ok=True)
    crop_path = os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{idx:02d}.png")
    cv2.imwrite(crop_path, processed)
    raw = ollama(
        "glm-ocr:latest",
        ("Read and return the text in this image. "
         "Preserve the original line breaks and layout structure. "
         "Output only the text, no explanation."),
        crop_path,
        timeout=60,
    )
    cleaned = clean_text(raw)

    if cleaned and len(cleaned) >= 3:
        print(f"     [OCR ✓] bubble {idx}: {cleaned[:50]!r}")
        return cleaned

    # ── Проход 2: ретрай с мягким препроцессингом ──
    print(f"     [OCR retry] bubble {idx} — trying soft preprocessing")
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
        print(f"     [OCR ✗ EMPTY] bubble {idx} (both passes empty) → {crop_path}, {crop_path2}")
    elif len(final) < 3:
        print(f"     [OCR ? SHORT] bubble {idx}: {final!r} (after retry)")
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


def _clip_overlapping_boxes(bubbles: list[dict], min_area: int = 400) -> list[dict]:
    """Обрезает перекрывающиеся баблы так, чтобы они не пересекались.

    Приоритет: меньший бабл сохраняется целиком, больший обрезается вокруг него.
    При одинаковом размере — приоритет сверху вниз, затем слева направо.
    Баблы, у которых после обрезки площадь < min_area, удаляются.
    """
    if len(bubbles) <= 1:
        return bubbles

    # Меньшая площадь = выше приоритет; ничья → верхний левее
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
            cx, cy, cw, ch = c["x"], c["y"], c["width"], c["height"]
            ix1 = max(bx, cx); iy1 = max(by, cy)
            ix2 = min(bx + bw, cx + cw); iy2 = min(by + bh, cy + ch)
            if ix1 >= ix2 or iy1 >= iy2:
                continue  # нет пересечения

            # 4 варианта обрезки — выбираем с наибольшей оставшейся площадью
            options: list[tuple[int, int, int, int]] = []
            nw = cx - bx
            if nw > 0: options.append((bx, by, nw, bh))
            nx = cx + cw; nw2 = (bx + bw) - nx
            if nw2 > 0: options.append((nx, by, nw2, bh))
            nh = cy - by
            if nh > 0: options.append((bx, by, bw, nh))
            ny = cy + ch; nh2 = (by + bh) - ny
            if nh2 > 0: options.append((bx, ny, bw, nh2))

            if options:
                bx, by, bw, bh = max(options, key=lambda r: r[2] * r[3])
            else:
                bw = bh = 0
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


def _compute_text_masks(img_cv: np.ndarray, bubbles: list[dict],
                        page_name: str = "") -> None:
    """
    Запускает Comic Text Detector ОДИН РАЗ на полной странице,
    затем нарезает маску по баблам.
    Результат в bubble["_sam2_mask"] — uint8, размер страницы.
    """
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

        # Otsu внутри CTD-зон — добирает тонкие штрихи которые CTD пропустил
        otsu_crop    = _binarize_text_mask(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        ctd_dilated  = cv2.dilate(ctd_crop, _dilate_k)
        otsu_in_zone = cv2.bitwise_and(otsu_crop, ctd_dilated)
        combined     = cv2.bitwise_or(ctd_crop, otsu_in_zone)

        m = np.zeros((h_img, w_img), dtype=np.uint8)
        m[y:y2, x:x2] = combined
        b["_sam2_mask"] = m

        if SAM_DEBUG_DIR:
            idx = b.get("idx", id(b))
            overlay = crop_bgr.copy()
            overlay[combined > 0] = (0, 200, 0)
            vis = cv2.addWeighted(crop_bgr, 0.5, overlay, 0.5, 0)
            prefix = f"{page_name}_" if page_name else ""
            dbg_path = os.path.join(SAM_DEBUG_DIR, f"{prefix}b{idx:03d}.png")
            cv2.imwrite(dbg_path, vis)
            print(f"  [ctd-dbg] → {dbg_path}")


def detect_text_color(img_cv: np.ndarray, bubble: dict,
                      default: tuple = (0, 0, 0)) -> tuple:
    """
    Определяет цвет текста в бабле.

    Приоритет:
    1. SAM2-маска (если доступна) — точные пиксели текста
    2. Маленький text_free (логотип) → HSV-насыщенность
    3. Fallback: Оцу + sanity-check
    """
    x, y, w, h = bubble["x"], bubble["y"], bubble["width"], bubble["height"]
    crop = img_cv[y:y+h, x:x+w]
    if crop.size == 0:
        return default

    # 1. CTD-маска — пиксели текстовой области (может включать фон)
    sam2_mask = bubble.get("_sam2_mask")
    if sam2_mask is not None:
        crop_mask = sam2_mask[y:y+h, x:x+w]
        text_pixels = crop[crop_mask > 0]
        if len(text_pixels) >= 10:
            # Полярность по фону: светлый фон → тёмный текст, тёмный → светлый.
            # CTD может захватывать полутоновый фон → берём только экстремальные пиксели.
            bg_pix = crop[crop_mask == 0]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            bg_gray = float(np.mean(bg_pix)) if len(bg_pix) > 0 else float(np.mean(crop_gray))
            gray_vals = np.mean(text_pixels.astype(np.float32), axis=1)
            if bg_gray >= 128:
                # Светлый/серый фон → чернила темнее → 25-й перцентиль
                thr = min(float(np.percentile(gray_vals, 25)), 110)
                ink = text_pixels[gray_vals <= thr]
            else:
                # Тёмный фон → чернила светлее → 75-й перцентиль
                thr = max(float(np.percentile(gray_vals, 75)), 150)
                ink = text_pixels[gray_vals >= thr]
            if len(ink) >= 5:
                bv, gv, rv = np.median(ink, axis=0).astype(int)
                return (int(rv), int(gv), int(bv))
            # Слишком мало экстремальных пикселей — обычная медиана
            bv, gv, rv = np.median(text_pixels, axis=0).astype(int)
            return (int(rv), int(gv), int(bv))

    # 2. Маленький text_free (логотип, заголовок) → HSV-насыщенность:
    # Оцу видит только тёмный контур, а цветная заливка букв — в насыщенных пикселях.
    # НО: если насыщенные пиксели — это ФОН (> 40% кропа), а не текст — пропускаем.
    # Пример: чёрный текст на красном фоне → почти весь кроп насыщен → это фон.
    bubble_area = bubble.get("width", 0) * bubble.get("height", 0)
    if bubble.get("class") == "text_free" and bubble_area < _LARGE_BUBBLE_PX:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mask = (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 80)
        sat_frac = np.mean(sat_mask)
        sat_pixels = crop[sat_mask]
        # sat_frac < 0.4: насыщенные пиксели — меньшинство → это текст (логотип)
        if sat_frac < 0.4 and len(sat_pixels) >= 30:
            bv, gv, rv = np.median(sat_pixels, axis=0).astype(int)
            rv, gv, bv = int(rv), int(gv), int(bv)
            if max(abs(rv - gv), abs(gv - bv), abs(rv - bv)) > 40:
                return (rv, gv, bv)

    # 3. Fallback: Оцу
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _binarize_text_mask(gray)
    text_pixels = crop[mask > 0]
    if len(text_pixels) < 10:
        return default
    bv, gv, rv = np.median(text_pixels, axis=0).astype(int)
    # Sanity check: если цвет текста слишком близок к фону — Оцу ошибся
    bg_pixels = crop[mask == 0]
    if len(bg_pixels) > 0:
        bb, bg_g, br = np.median(bg_pixels, axis=0).astype(int)
        if max(abs(rv - br), abs(gv - bg_g), abs(bv - bb)) < 40:
            return default
    return (int(rv), int(gv), int(bv))


# ─── инпейтинг ────────────────────────────────────────────────────────────────

_LARGE_BUBBLE_PX = 80_000  # порог (px²) — выше него используем глифовую маску


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Заполняет замкнутые дыры внутри бинарной маски (текст=255, фон=0).
    Алгоритм: заливка фона снаружи → всё что не залито и не текст = дыры.
    """
    h, w = mask.shape
    # Паддинг нулями гарантирует связность фона от края
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = mask
    inv = cv2.bitwise_not(padded)
    # Заливка от угла достигает всего внешнего фона
    cv2.floodFill(inv, None, (0, 0), 0)
    # Оставшиеся 255 в inv — замкнутые дыры
    holes = inv[1:h+1, 1:w+1]
    return cv2.bitwise_or(mask, holes)


def build_inpaint_mask(img_cv: np.ndarray, bubbles: list[dict],
                        shrink: int = 1) -> np.ndarray:
    """
    Строит маску для инпейтинга: 255 = стираем, 0 = оставляем.

    Приоритет на бабл:
    1. SAM2-маска (если доступна) — точная маска текстовых пикселей + дилатация 5px
    2. Маленький бабл (< _LARGE_BUBBLE_PX): полный bbox
    3. Большой бабл: глифовая маска через Оцу + морфология (fallback без SAM2)
    """
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    connect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilate_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Для CTD-маски: крупная дилатация + горизонтальное замыкание,
    # чтобы тонкие штрихи (1-2px) и разрывы между ними полностью покрывались.
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

        # 1. CTD-маска — замыкание соединяет горизонтальные разрывы, дилатация покрывает края
        sam2_mask = b.get("_sam2_mask")
        if sam2_mask is not None:
            crop_m = sam2_mask[y0:y1, x0:x1]
            coverage = np.count_nonzero(crop_m) / max(crop_m.size, 1)
            if coverage >= 0.05 and crop_m.any():
                # CTD нашёл достаточно текста — используем маску
                closed = cv2.morphologyEx(crop_m, cv2.MORPH_CLOSE, closing_k)
                glyph  = cv2.dilate(closed, inpaint_k)
                glyph  = _fill_mask_holes(glyph)   # заполняем дыры внутри символов
                mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], glyph)
                continue
            # CTD нашёл < 5% bbox — маска ненадёжна, падаем на полный bbox

        # 2. Маленький бабл — полный bbox (логотипы, мелкие области)
        if bw * bh < _LARGE_BUBBLE_PX:
            mask[y0:y1, x0:x1] = 255
            continue

        # 3. Большой бабл — глифовая маска через Оцу
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
    mask_np = build_inpaint_mask(img_cv, bubbles)

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
        print(f"  [inpaint ⚠] LaMa failed ({e}), falling back to cv2.inpaint")
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)


# ─── отрисовка ────────────────────────────────────────────────────────────────

def fit_text_in_box(draw, text: str, box_w: int, box_h: int,
                    font_path: str | None = None) -> tuple:
    """
    Подбирает максимальный размер шрифта и оптимальное разбиение на строки.

    Стратегия:
    - Для каждого кандидатного размера шрифта (бинарный поиск от 5 до 90px)
      пробуем РАЗНЫЕ способы разбиения текста на строки:
        1. Greedy (максимум слов в строке)
        2. Equal — стараемся сделать строки одинаковой длины
        3. Few — пробуем 1, 2, 3, 4 строки явно
      Выбираем то разбиение которое лучше заполняет баббл.
    - Если слово не помещается даже одно — переносим с дефисом ("ПРЕМИ-/АЛЬНАЯ").

    Возвращает (font, lines, line_heights, spacing).
    """
    if font_path is None:
        font_path = DEFAULT_FONT
    # Padding с запасом: 4 мало для крупных шрифтов где descender выпадает.
    # Дополнительный безопасный отступ для нижних выносных элементов.
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
        """
        Полная высота строки на основе метрики шрифта (ascent + descent).
        Это даёт ОДИНАКОВУЮ высоту для всех строк, независимо от того,
        есть ли в строке буквы с descender (р/у/д/щ/ц/ф/...).

        Раньше использовалось textbbox[3]-bbox[1] — это давало
        ФАКТИЧЕСКУЮ высоту нарисованных пикселей конкретной строки.
        Для строки "АБВ" она меньше чем для "дру", и когда мы складывали
        такие высоты, descender последней строки выпадал за границу баббла.
        """
        ascent, descent = font.getmetrics()
        return ascent + descent

    def hyphenate(word: str, font, max_w: int) -> list[str]:
        """Разбивает длинное слово с дефисом на куски по max_w."""
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
                    # один символ шире max_w — выкладываем как есть
                    chunks.append(ch)
                    cur = ""
        if cur:
            chunks.append(cur)
        return chunks

    def wrap_greedy(words: list[str], font,
                     allow_hyphenation: bool = False) -> list[str] | None:
        """
        Жадно запихиваем максимум слов в строку.
        Если allow_hyphenation=False и слово длиннее usable_w — возвращаем None
        (даём бинарному поиску шанс попробовать шрифт меньше).
        """
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
                # Слово не помещается само по себе
                if text_width(word, font) > usable_w:
                    if not allow_hyphenation:
                        return None  # пусть найдут шрифт поменьше
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
        """
        Пробует разбить ровно на target_lines примерно одинаковой длины.
        Используется чтобы у короткой фразы текст распределился по баблу.
        """
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
            elif len(test) <= target_line_chars * 1.3 and text_width(test, font) <= usable_w:
                line = test
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        if len(lines) != target_lines:
            return None
        # Все строки должны влезть по ширине
        for ln in lines:
            if text_width(ln, font) > usable_w:
                return None
        return lines

    def measure(lines: list[str], font) -> tuple[int, int, int] | None:
        """Возвращает (total_h, max_w, spacing) или None если не влезает."""
        if not lines:
            return None
        # Все строки имеют одинаковую полную высоту (ascent + descent)
        lh = line_height(font)
        heights = [lh] * len(lines)
        spacing = max(2, font.size // 7)
        total_h = sum(heights) + spacing * (len(lines) - 1)
        max_w = max(text_width(ln, font) for ln in lines)
        if total_h <= usable_h and max_w <= usable_w:
            return total_h, max_w, spacing
        return None

    def score(lines: list[str], total_h: int, max_w: int, font) -> float:
        """
        Чем выше — тем лучше. Идеал:
        - текст занимает большую часть баббла (90%+ по обоим осям)
        - строки примерно одинаковой длины (визуально красиво)
        """
        fill_h = total_h / usable_h
        fill_w = max_w / usable_w
        # Среднее заполнение + штраф за неравномерные строки
        widths = [text_width(ln, font) for ln in lines]
        if widths:
            avg_w = sum(widths) / len(widths)
            uniformity = avg_w / max(widths) if max(widths) else 1.0
        else:
            uniformity = 1.0
        return (fill_h + fill_w) * 0.5 + uniformity * 0.2

    def best_wrap_for_size(font, allow_hyphenation: bool = False
                             ) -> tuple[list[str], int, int, int] | None:
        """
        Перебирает стратегии разбиения для данного размера. Лучшая по score.
        Если allow_hyphenation=False — отказывается от разбиений где пришлось
        бы переносить слово с дефисом.

        Если text содержит \\n — каждый перенос трактуется как принудительный
        разрыв строки (параграф). Внутри параграфа делается word-wrap по ширине.
        В этом случае scoring не применяется — структура текста фиксирована.
        """
        paragraphs = text.split("\n")

        if len(paragraphs) > 1:
            # Текст с принудительными переносами — чтим структуру
            all_lines: list[str] = []
            for para in paragraphs:
                para_words = para.split()
                if not para_words:
                    all_lines.append("")   # пустая строка = визуальный разрыв
                    continue
                wrapped = wrap_greedy(para_words, font,
                                      allow_hyphenation=allow_hyphenation)
                if wrapped is None:
                    return None  # слово не вмещается — пробуем шрифт поменьше
                all_lines.extend(wrapped)
            m = measure(all_lines, font)
            return (all_lines, *m) if m is not None else None

        # Нет принудительных переносов — оригинальная логика
        words = text.split() or [text]
        candidates: list[tuple[list[str], int, int, int]] = []

        # Стратегия 1: жадная (с/без переносов)
        greedy = wrap_greedy(words, font, allow_hyphenation=allow_hyphenation)
        if greedy:
            m = measure(greedy, font)
            if m:
                candidates.append((greedy, *m))

        # Стратегия 2: явно 1, 2, 3, 4 строки сбалансированно
        # (wrap_balanced никогда не делает переносов — там просто слова в строки)
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
        """Бинарный поиск максимального размера. Возвращает best tuple или None."""
        lo, hi = 6, 90
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

    # Стратегия: сначала ищем размер БЕЗ переносов слов (визуально лучше).
    # Если ни при каком размере без переносов не помещается — пробуем с переносами.
    best = _search(allow_hyphenation=False)
    if best is None:
        best = _search(allow_hyphenation=True)

    if best is not None:
        return best

    # Fallback: минимальный размер с обрезкой
    font = try_load_font(5) or ImageFont.load_default()
    return font, [text[:20]], [8], 2


def _effective_box(bx: int, by: int, bw: int, bh: int,
                   smaller_bubbles: list[dict]) -> tuple[int, int, int, int]:
    """
    Возвращает (x, y, w, h) — наибольший прямоугольник внутри (bx, by, bw, bh),
    не пересекающийся с bbox'ами маленьких баблов.
    Для каждого пересечения выбирается один из 4 разрезов (лево/право/верх/низ)
    с максимальной оставшейся площадью. Это гарантирует что текст большого бабла
    не заходит в зону маленького, не выходя за рамки исходного bbox.
    """
    ex, ey, ew, eh = bx, by, bw, bh
    for ob in smaller_bubbles:
        ox, oy, ow, oh = ob["x"], ob["y"], ob["width"], ob["height"]
        ix1 = max(ex, ox)
        iy1 = max(ey, oy)
        ix2 = min(ex + ew, ox + ow)
        iy2 = min(ey + eh, oy + oh)
        if ix1 >= ix2 or iy1 >= iy2:
            continue  # нет пересечения
        options: list[tuple[int, int, int, int]] = []
        nw = ox - ex
        if nw > 0:
            options.append((ex, ey, nw, eh))          # обрезать справа
        nx = ox + ow
        nw2 = (ex + ew) - nx
        if nw2 > 0:
            options.append((nx, ey, nw2, eh))          # обрезать слева
        nh = oy - ey
        if nh > 0:
            options.append((ex, ey, ew, nh))           # обрезать снизу
        ny = oy + oh
        nh2 = (ey + eh) - ny
        if nh2 > 0:
            options.append((ex, ny, ew, nh2))          # обрезать сверху
        if options:
            ex, ey, ew, eh = max(options, key=lambda r: r[2] * r[3])
    return ex, ey, ew, eh


def _render_text_block(text: str, box_w: int, box_h: int,
                        color: tuple, font_path: str | None = None,
                        font_size_override: int | None = None) -> Image.Image:
    """
    Рендерит текст в прозрачный RGBA-блок размером box_w × box_h.
    Возвращает PIL.Image — её можно вращать и накладывать на инпейтнутую страницу.

    Если font_size_override указан — используем точно этот размер шрифта,
    без бинарного поиска (для пользовательских настроек из редактора).
    Текст всё равно переносится по ширине, но размер не подбирается.
    """
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if font_size_override:
        font, lines, line_heights, spacing = _fit_at_exact_size(
            draw, text, box_w, box_h, font_path, int(font_size_override)
        )
    else:
        font, lines, line_heights, spacing = fit_text_in_box(
            draw, text, box_w, box_h, font_path
        )
    # Padding должен совпадать с padding в fit_text_in_box / _fit_at_exact_size
    # (там используется 6 для безопасности descender'ов)
    padding = 6
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


def _fit_at_exact_size(draw, text: str, box_w: int, box_h: int,
                        font_path: str | None, size: int) -> tuple:
    """
    Рендерит при ФИКСИРОВАННОМ размере шрифта (override из редактора).
    Текст переносится по ширине жадно с дефисом если слово слишком длинное.
    Не пытается подобрать оптимум — пользователь сам выбрал размер.
    """
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
        """Жадный word-wrap одного параграфа с дефисным переносом длинных слов."""
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
            lines.append("")   # пустая строка = разрыв параграфа
        else:
            lines.extend(_wrap_para(para_words))

    # Полная высота строки через метрику шрифта (ascent + descent).
    # Одинаковая для всех строк — гарантирует что descender не выпадет.
    ascent, descent = font.getmetrics()
    lh = ascent + descent
    heights = [lh] * len(lines)
    spacing = max(2, size // 7)
    return font, lines, heights, spacing


def draw_results(img_cv: np.ndarray, bubbles: list[dict],
                 debug: bool = False, page_name: str = "") -> np.ndarray:
    """
    Заливает оригинальный текст через LaMa-инпейтинг, затем рисует переводы
    с цветом оригинала. Текст рисуется горизонтально (поворот не используется —
    для большинства баблов японский текст должен читаться слева-направо после перевода).

    debug=False (по умолчанию): чистая страница, только заменённый текст.
    debug=True: дополнительно цветные рамки и номера баблов для диагностики.
    """
    # 1. Маски текста через Comic Text Detector (один проход на страницу)
    print("  Segmenting text regions (CTD)...")
    _compute_text_masks(img_cv, bubbles, page_name=page_name)

    # 2. Определяем цвет текста ДО того как сотрём оригинал
    for b in bubbles:
        if b.get("translation") and b.get("_text_color") is None:
            b["_text_color"] = detect_text_color(img_cv, b)

    # 3. Инпейнтим оригинальный текст
    print("  Inpainting original text (LaMa)...")
    inpainted = inpaint_page(img_cv, bubbles)

    # 3. Рендерим переводы поверх инпейтнутой страницы.
    # Большие ТОJТЛы рисуем первыми — маленькие всегда окажутся поверх,
    # что устраняет перекрытие текста когда bbox'ы пересекаются.
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

        # Маленькие баблы (идут после в render_order) «занимают» свои области.
        # Обрезаем рабочую область текущего бабла, чтобы не рисовать поверх них.
        smaller = [s for s in render_order[i + 1:] if s.get("translation")]
        ex, ey, ew, eh = _effective_box(bx, by, bw, bh, smaller)

        if ew < 20 or eh < 20:
            continue

        block = _render_text_block(
            translation, ew, eh, color,
            font_path=b.get("font_path"),
            font_size_override=b.get("font_size"),
        )
        pil.paste(block, (ex, ey), block)

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
                 fast_mode: bool = False,
                 errors: ErrorLog | None = None,
                 on_stage=None):
    """
    on_stage(page_idx, stage_key) — опциональный коллбэк прогресса.
    stage_key ∈ {"stage_detect", "stage_ocr", "stage_analyze",
                 "stage_attribute", "stage_translate", "stage_inpaint"}

    fast_mode: если True, пропускает analyze_page_full, intro-detection и
    attribute_bubbles. Это экономит 2-3 vision-вызова на страницу (≈30-60с)
    но теряет контекст: персонажи не запоминаются, у баблов нет speaker'а,
    перевод делается без сцены. Полезно для быстрых черновых прогонов.
    """
    def stage(key):
        if on_stage:
            try:
                on_stage(page_idx, key)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"Page {page_idx}: {image_path}")
    if fast_mode:
        print(f"[fast mode] skipping page analysis and speaker attribution")
    print(f"{'='*60}")

    stage("stage_detect")

    # Читаем файл один раз — оба формата получают данные из одного буфера
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        img_cv = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # 1. Детекция и сортировка баблов (PIL уже в памяти — файл не перечитывается)
    bubbles = detect_bubbles(image_pil, threshold=0.5)
    bubbles = _clip_overlapping_boxes(bubbles)
    text_bubbles = sorted(
        [b for b in bubbles if b["class"] in ("text_bubble", "text_free")],
        key=reading_order,
    )
    print(f"Bubbles: {len(text_bubbles)}")

    if not text_bubbles and errors:
        errors.add(page_idx, "no_bubbles",
                   "No text bubbles found on this page",
                   image=image_path)

    # Эвристика: мало баблов → возможно это галерея представлений.
    # Проверка стоит ~1 vision-вызов, но на intro-странице мы сэкономим
    # обычный analyze_page_full и сразу получим правильные имена в архив.
    # В fast_mode эту проверку пропускаем — нет архива персонажей всё равно.
    if not fast_mode and len(text_bubbles) <= 2:
        print("\n── Checking: character gallery? ──")
        if detect_character_intro_page(image_path):
            print("\n── Extracting introductions ──")
            characters_context = extract_character_intros(image_path, archive, page_idx)
            print(characters_context)

            if characters_context != "CHARACTERS ON THIS PAGE: unknown":
                # Успешно распознана галерея — сохраняем без перевода
                manga_ctx.update("Character introduction page.")
                cv2.imwrite(output_path, img_cv)
                print(f"\nSaved (unchanged): {output_path}")
                print(f"  ⓘ character introduction page — no translation needed")
                return text_bubbles
            else:
                print("  [intro detect] extraction failed → treating as regular page")

    # 2. OCR
    print("\n── OCR ──")
    stage("stage_ocr")
    for i, b in enumerate(text_bubbles):
        b["text"] = ocr_region(img_cv, b["x"], b["y"], b["width"], b["height"],
                               idx=i + 1, page_idx=page_idx)
        if not b["text"] and errors:
            errors.add(page_idx, "ocr_empty",
                       f"Bubble #{i+1}: OCR returned no text after two passes",
                       bubble_idx=i+1,
                       bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})",
                       crop=os.path.join(CROPS_DIR, f"p{page_idx:03d}_bubble_{i+1:02d}.png"))

    if fast_mode:
        # Пропускаем стадии analyze и attribute. Без них:
        # - characters_context = заглушка
        # - speaker/gender у баблов = "unknown"
        # - page_context = пустой
        # translate_batch и _translate_persistent корректно работают с этими
        # пустыми значениями (просто без доп. контекста в промпте).
        print("\n[fast mode] skipping analyze + attribute stages")
        characters_context = "CHARACTERS ON THIS PAGE: unknown"
        page_context = ""
        page_summary = ""   # пустая строка — manga_ctx.update() её отфильтрует
        for b in text_bubbles:
            b["speaker"] = "unknown"
            b["gender"] = "unknown"
    else:
        # 3-4. Совмещённый анализ персонажей + сцены (один vision-вызов)
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

        # 5. Атрибуция реплик
        print("\n── Attribution ──")
        stage("stage_attribute")
        text_bubbles = attribute_bubbles(
            image_path, text_bubbles, page_context, characters_context, archive
        )
        for i, b in enumerate(text_bubbles):
            print(f"  [{i+1}] {b.get('speaker', '?')} ({b.get('gender', '?')}): "
                  f"{b.get('text', '')[:40]}")
            if b.get("text") and b.get("speaker") == "unknown" and errors:
                errors.add(page_idx, "speaker_unknown",
                           f"Bubble #{i+1}: could not determine speaker",
                           bubble_idx=i+1,
                           text=b.get("text", "")[:100])

    # 6. Перевод — один вызов LLM на всю страницу
    print("\n── Translation ──")
    stage("stage_translate")
    translate_batch(text_bubbles, page_context, manga_ctx, target_lang,
                    errors=errors, page_idx=page_idx)
    for i, b in enumerate(text_bubbles):
        print(f"  [{i+1}] {b.get('text', '')[:25]} → {b.get('translation', '')[:40]}")

    # 7. Обновляем сюжетный контекст
    manga_ctx.update(page_summary)

    # 8. Сохраняем аннотированное изображение
    stage("stage_inpaint")
    annotated = draw_results(img_cv, text_bubbles, debug=debug,
                             page_name=os.path.splitext(os.path.basename(output_path))[0])
    cv2.imwrite(output_path, annotated)

    # Сводка по странице
    empty_ocr = sum(1 for b in text_bubbles if not b.get("text"))
    no_translation = sum(1 for b in text_bubbles
                          if b.get("text") and not b.get("translation"))
    err_translation = sum(1 for b in text_bubbles
                          if b.get("translation") == "[error]")
    ok = len(text_bubbles) - empty_ocr - no_translation - err_translation
    print(f"\nSaved: {output_path}")
    print(f"  ✓ translated: {ok} | ⚠ no translation: {no_translation} | "
          f"✗ OCR empty: {empty_ocr} | ✗ translation error: {err_translation}")
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


def process_directory(input_dir: str, output_dir: str = "results",
                      target_lang: str = "Russian",
                      font_path: str = "arial.ttf",
                      debug: bool = False,
                      fast_mode: bool = False,
                      error_log_path: str = "errors.log",
                      llm_model: str | None = None,
                      on_page_done=None,
                      on_start=None,
                      on_finish=None,
                      on_stage=None):
    """
    llm_model — Ollama-модель для анализа/перевода. Если None — используется LLM_MODEL.
    fast_mode — быстрый режим: пропускает анализ страницы и атрибуцию баблов
        к спикерам. Экономит ~2-3 vision-вызова и 30-60с на страницу.
        Качество перевода может быть хуже (нет контекста сцены и speaker),
        зато перевод занимает минимальное время.
    on_stage(page_idx, stage_key) — прогресс внутри страницы:
        stage_detect → stage_ocr → stage_analyze → stage_attribute
        → stage_translate → stage_inpaint
    """
    if llm_model:
        global LLM_MODEL
        LLM_MODEL = llm_model
        print(f"[llm] Using model: {llm_model}")
    # Проверяем шрифт. Если указан — используем его. Если нет — пробуем
    # типичные манга-friendly шрифты по очереди, отдавая предпочтение жирным.
    def _resolve_font(requested: str) -> str | None:
        """Возвращает рабочий путьGТепеweТеУУCЫ к шрифту или None."""
        # 1. Сначала пробуем то что запросил пользователь
        try:
            ImageFont.truetype(requested, 12)
            return requested
        except OSError:
            pass
        # 2. Кандидаты в порядке предпочтения. Жирные манга-шрифты сверху,
        #    затем системные жирные fallback'и, в конце обычные.
        candidates = [
            "Ace 2.0 BB Cyr.ttf", "animeace2_bld.ttf", "animeace2_reg.ttf",
            "CCWildWords.ttf", "wildwords.ttf",
            "arialbd.ttf",    # Arial Bold — есть на каждой Windows
            "ARIALBD.TTF",
            "calibrib.ttf",   # Calibri Bold
            "verdanab.ttf",   # Verdana Bold
            "DejaVuSans-Bold.ttf",
            "arial.ttf",      # Final fallback
        ]
        # Также пробуем стандартные пути к шрифтам Windows
        win_fonts = "C:/Windows/Fonts/"
        if os.path.isdir(win_fonts):
            candidates = candidates + [
                os.path.join(win_fonts, name) for name in os.listdir(win_fonts)
                if any(kw in name.lower() for kw in ("bold", "bd", "black", "heavy"))
            ][:5]
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

    # Сохраняем журнал ошибок
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

    # Сводка по журналу ошибок
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
        # Для японских имён в скобках укажите CJK-шрифт, например:
        #   font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        #   font_path="C:/Windows/Fonts/YuGothM.ttc"
        font_path="arial.ttf",
        # debug=True — рисует рамки и номера баблов для диагностики
        debug=False,
        # куда писать журнал проблем
        error_log_path="errors.log",
    )
