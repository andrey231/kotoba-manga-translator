import json
import logging
import re

import requests

from context import ErrorLog, MangaContext
from llm import ollama
from settings import settings
from text_utils import parse_json_array

logger = logging.getLogger(__name__)

TRANSLATE_NUM_CTX = 8192
TRANSLATE_MAX_TOKENS = 2048
TRANSLATE_INPUT_BYTES = TRANSLATE_NUM_CTX - TRANSLATE_MAX_TOKENS - 512


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


_META_MARKERS: tuple[str, ...] = (
    "depending on",
    "it could be",
    "it can be translated",
    "note:",
    "please note",
    "keep in mind",
    "в зависимости",
    "можно перевести",
    "можно оставить",
    "это можно",
    "стоит отметить",
    "следует отметить",
    "примечание:",
    "обратите внимание",
)


def _build_glossary_prompt() -> str:
    if not settings().glossary:
        return ""
    lines = ["GLOSSARY — always use these exact translations, even if context suggests otherwise:"]
    for entry in settings().glossary:
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


def translate_batch(
    bubbles: list[dict],
    page_context: str,
    manga_ctx: MangaContext,
    target_lang: str = "Russian",
    retries: int = 3,
    errors: ErrorLog | None = None,
    page_idx: int = 0,
) -> None:
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
                errors.add(
                    page_idx,
                    "ocr_empty",
                    f"Bubble #{i + 1} has no OCR text — nothing to translate",
                    bubble_idx=i + 1,
                    bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})",
                )

    if not to_translate:
        return

    missing_indices = []

    chunks = _translation_chunks(to_translate, page_context, manga_ctx, target_lang, gender_hints)
    for chunk_idx, (offset, chunk) in enumerate(chunks):
        _translate_chunk(
            chunk,
            chunk_idx,
            offset,
            missing_indices,
            page_context,
            manga_ctx,
            target_lang,
            gender_hints,
            errors,
            page_idx,
        )

    if missing_indices:
        logger.debug(
            f"     [retry] {len(missing_indices)} bubble(s) missing translation, "
            f"translating one-by-one with persistent fallback..."
        )
        for seq in missing_indices:
            bubble_idx, b = to_translate[seq]
            translation = _translate_persistent(
                b, page_context, manga_ctx, target_lang, gender_hints, retries=retries
            )
            if translation:
                b["translation"] = translation
                logger.debug(f"     [retry-ok] bubble #{bubble_idx + 1}: {translation[:50]}")
            else:
                b["translation"] = ""
                if errors:
                    errors.add(
                        page_idx,
                        "translation_missing",
                        f"Bubble #{bubble_idx + 1}: all persistent retries failed",
                        bubble_idx=bubble_idx + 1,
                        original_text=b.get("text", ""),
                        speaker=b.get("speaker", "?"),
                    )


def _build_chunk_entries(chunk: list, gender_hints: dict) -> list[dict]:
    return [
        {
            "id": i + 1,
            "speaker": b.get("speaker", "unknown"),
            "gender": gender_hints.get(b.get("gender"), "gender unknown"),
            "text": b["text"],
            "emotion": b.get("emotion_hint", "neutral"),
        }
        for i, (_, b) in enumerate(chunk)
    ]


def _translation_chunks(items, page_context, manga_ctx, target_lang, gender_hints):
    offset = 0
    system = _translation_system(target_lang)
    while offset < len(items):
        count = min(settings().chunk_size, len(items) - offset)
        while count:
            chunk = items[offset : offset + count]
            prompt = _build_translation_prompt(
                _build_chunk_entries(chunk, gender_hints), page_context, manga_ctx
            )
            if len((system + prompt).encode("utf-8")) <= TRANSLATE_INPUT_BYTES:
                break
            count -= 1
        if not count:
            raise ValueError(
                "Translation input exceeds the context budget; shorten the glossary or source text"
            )
        yield offset, chunk
        offset += count


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
input id) and "translation". The "translation" field must contain ONLY the final
translated text — no explanations, no commentary, no notes, no "//" comments, no
parentheticals about layout or context. Always produce a translation, even for single
words or sound effects.

Examples of the "translation" field:
- Input "WITHOUT" → "БЕЗ"          (RIGHT)
- Input "WITHOUT" → "// a bit more space for context: БЕЗ"   (WRONG — never add notes/comments)
- Input "TRUE" → "// usually left in English: ВЕРНО"          (WRONG)
- Input "TRUE" → "ВЕРНО"           (RIGHT)"""


def _build_translation_prompt(
    entries: list[dict], page_context: str, manga_ctx: MangaContext
) -> str:
    inputs_json = json.dumps(entries, ensure_ascii=False, indent=2)
    glossary_section = _build_glossary_prompt()
    parts = [manga_ctx.to_prompt()]
    if page_context:
        parts.append(f"PAGE CONTEXT:\n{page_context}")
    if glossary_section:
        parts.append(glossary_section)
    parts.append(f"INPUT (JSON array of {len(entries)} bubbles):\n{inputs_json}")
    return "\n\n".join(parts)


def _call_translation_llm(prompt: str, system: str, n: int) -> str:
    return ollama(
        settings().llm_model,
        prompt,
        timeout=120,
        num_predict=TRANSLATE_MAX_TOKENS,
        temperature=0.0,
        system=system,
        fmt=_translation_schema(n),
        num_ctx=TRANSLATE_NUM_CTX,
    )


def _looks_like_commentary(t: str, src: str) -> bool:
    low = t.lower()
    if low.startswith("//") or any(low.startswith(m) for m in _META_MARKERS):
        return True
    return len(src) <= 15 and len(t) > max(40, len(src) * 6)


def _apply_chunk_results(results: list, chunk: list, offset: int, missing_indices: list) -> None:
    translations = {}
    duplicate_ids = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        local_id = result.get("id")
        text = result.get("translation")
        if type(local_id) is not int or not 1 <= local_id <= len(chunk):
            continue
        if local_id in translations:
            duplicate_ids.add(local_id)
        translations[local_id] = text if isinstance(text, str) else ""
    for local_id, (_, bubble) in enumerate(chunk, start=1):
        text = translations.get(local_id, "").strip()
        if (
            local_id in duplicate_ids
            or not text
            or _looks_like_commentary(text, bubble.get("text", ""))
        ):
            missing_indices.append(offset + local_id - 1)
        else:
            bubble["translation"] = text


def _translate_chunk(
    chunk: list,
    chunk_idx: int,
    offset: int,
    missing_indices: list,
    page_context: str,
    manga_ctx: MangaContext,
    target_lang: str,
    gender_hints: dict,
    errors,
    page_idx: int,
) -> None:
    entries = _build_chunk_entries(chunk, gender_hints)
    prompt = _build_translation_prompt(entries, page_context, manga_ctx)
    system = _translation_system(target_lang)
    raw = _call_translation_llm(prompt, system, len(entries))

    if not raw:
        missing_indices.extend(range(offset, offset + len(chunk)))
        if errors:
            errors.add(
                page_idx,
                "empty_response",
                f"Chunk {chunk_idx + 1} returned no translation",
                bubbles_affected=len(chunk),
            )
        return

    results = parse_json_array(raw)
    if not results:
        if errors:
            errors.add(
                page_idx,
                "json_parse",
                f"Chunk {chunk_idx + 1}: model response did not parse",
                raw_response=raw[:500],
                expected_count=len(chunk),
            )
        missing_indices.extend(range(offset, offset + len(chunk)))
        return

    expected = len(chunk)
    valid = [r for r in results if isinstance(r, dict)]
    if len(valid) != expected:
        logger.debug(
            f"     [chunk {chunk_idx + 1}] count mismatch: got {len(valid)} items, "
            f"expected {expected} → one-by-one fallback"
        )
        if errors:
            errors.add(
                page_idx,
                "count_mismatch",
                f"Chunk {chunk_idx + 1}: model returned {len(valid)} items, expected {expected}",
                expected_count=expected,
                got_count=len(valid),
            )
        missing_indices.extend(range(offset, offset + len(chunk)))
        return

    _apply_chunk_results(results, chunk, offset, missing_indices)


def _translate_persistent(
    bubble: dict,
    page_context: str,
    manga_ctx: MangaContext,
    target_lang: str,
    gender_hints: dict,
    retries: int = 3,
) -> str:
    text = bubble.get("text", "").strip()
    if not text:
        return ""

    speaker = bubble.get("speaker", "unknown")
    gender = bubble.get("gender", "unknown")
    gender_h = gender_hints.get(gender, "gender unknown")
    emotion = bubble.get("emotion_hint", "neutral")
    glossary_prompt = _build_glossary_prompt()

    strategies = [
        (
            "ctx-speaker",
            lambda: (
                f"Translate this manga text to {target_lang}.\n"
                f"Speaker: {speaker} ({gender_h}). Tone/emotion: {emotion}.\n"
                + f"{manga_ctx.to_prompt()}\n{page_context}\n"
                + f"Source: {text}\n\n"
                f"Rules: match the emotional intensity of the original "
                f"('{emotion}' means the translation must feel {emotion}). "
                f"Keep proper nouns, titles, and names unchanged. "
                f"Output ONLY the translated text. No quotes. No explanations."
            ),
        ),
        (
            "ui-style",
            lambda: (
                f"Переведи на русский (только перевод, без пояснений): {text}"
                if target_lang.lower() == "russian"
                else f"Translate to {target_lang} (output translation only): {text}"
            ),
        ),
        (
            "raw-imperative",
            lambda: (
                f"Переведи строку на {target_lang}, только результат: {text}"
                if target_lang.lower() == "russian"
                else f"{target_lang} translation only: {text}"
            ),
        ),
    ]

    refusal_markers = (
        "i cannot",
        "i can't",
        "cannot translate",
        "i am unable",
        "sorry,",
        "as an ai",
        "no translation",
        "не могу",
        "невозможно",
        "к сожалению",
    ) + _META_MARKERS

    for attempt, (strat_label, build_prompt) in enumerate(strategies[:retries], start=1):
        prompt = "\n".join(part for part in (glossary_prompt, build_prompt()) if part)
        if len(prompt.encode("utf-8")) > TRANSLATE_INPUT_BYTES:
            logger.warning("Translation retry exceeds the input budget")
            continue
        try:
            raw = ollama(
                settings().llm_model,
                prompt,
                timeout=60,
                num_predict=TRANSLATE_MAX_TOKENS,
                num_ctx=TRANSLATE_NUM_CTX,
                temperature=0.3,
            )
        except requests.RequestException:
            raise
        except ValueError as e:
            logger.warning(f"     [persistent {attempt} {strat_label}] error: {e}")
            continue

        if not raw:
            logger.warning(f"     [persistent {attempt} {strat_label}] empty response")
            continue

        cleaned = _clean_translation(raw)
        if not cleaned:
            logger.debug(
                f"     [persistent {attempt} {strat_label}] cleanup left nothing; "
                f"raw[:80]={raw[:80]!r}"
            )
            continue
        if any(m in cleaned.lower() for m in refusal_markers):
            logger.debug(f"     [persistent {attempt} {strat_label}] refusal: '{cleaned[:50]}'")
            continue
        logger.debug(f"     [persistent ok @ attempt {attempt} / {strat_label}]")
        return cleaned

    logger.warning(f"     [persistent fail] '{text[:40]}' — all strategies failed")
    return ""


def _clean_translation(raw: str) -> str:
    result = raw.strip()
    if result.startswith("```"):
        result = re.sub(r"^```[a-zA-Z]*\s*", "", result)
        result = result.rstrip("`").strip()
    result = result.strip('"').strip("'").strip()

    prefixes = (
        "translation:",
        "перевод:",
        "russian:",
        "english:",
        "answer:",
        "result:",
        "ответ:",
    )
    for prefix in prefixes:
        if result.lower().startswith(prefix):
            result = result[len(prefix) :].strip().strip('"').strip("'").strip()
            break

    lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
    if not lines:
        return ""

    _meta_intro = ("вот ", "here is", "below is", "please note", "note:")
    first_low = lines[0].lower()
    if any(first_low.startswith(m) for m in _meta_intro) or any(
        first_low.startswith(m) for m in _META_MARKERS
    ):
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
