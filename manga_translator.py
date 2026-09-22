import logging
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context import CharacterArchive, ErrorLog, MangaContext
from fonts import resolve_font
from image_io import SUPPORTED_EXTENSIONS, read_image, write_image
from models import detect_bubbles
from ocr import ocr_region
from page_analysis import (
    analyze_page_full,
    attribute_bubbles,
    detect_character_intro_page,
    extract_character_intros,
)
from rendering import _clip_overlapping_boxes, draw_results
from settings import settings, updated_settings, use_settings
from text_utils import _infer_emotion_tag, natural_key
from translation import translate_batch

logger = logging.getLogger(__name__)


READING_BAND_MIN = 40
READING_BAND_DIVISOR = 25
SFX_OVERLAP_FRAC = 0.3


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
    low_threshold = min(settings().detect_threshold, settings().sfx_threshold)
    detections = detect_bubbles(image_pil, threshold=low_threshold)

    bubbles = [b for b in detections if b["confidence"] >= settings().detect_threshold]
    for b in detections:
        if b["confidence"] >= settings().detect_threshold or b["class"] != "text_free":
            continue
        if not _is_covered_by(b, bubbles, SFX_OVERLAP_FRAC):
            bubbles.append(b)
            logger.debug(f"  [sfx+] text_free @ ({b['x']},{b['y']}) score={b['confidence']:.2f}")

    bubbles = _clip_overlapping_boxes(bubbles, min_area=settings().min_bubble_area)
    band = max(READING_BAND_MIN, img_cv.shape[0] // READING_BAND_DIVISOR)
    return sorted(
        [b for b in bubbles if b["class"] in ("text_bubble", "text_free")],
        key=lambda b: reading_order(b, band),
    )


def _handle_intro_page(
    image_path: str,
    archive: CharacterArchive,
    manga_ctx: MangaContext,
    page_idx: int,
    output_path: str,
    img_cv: np.ndarray,
) -> bool:
    logger.debug("\n── Checking: character gallery? ──")
    if not detect_character_intro_page(image_path):
        return False

    logger.debug("\n── Extracting introductions ──")
    characters_context = extract_character_intros(image_path, archive, page_idx)
    logger.debug(characters_context)
    if characters_context == "CHARACTERS ON THIS PAGE: unknown":
        logger.warning("  [intro detect] extraction failed → treating as regular page")
        return False

    manga_ctx.update("Character introduction page.", page_idx)
    write_image(output_path, img_cv)
    logger.debug(f"\nSaved (unchanged): {output_path}")
    logger.debug("  ⓘ character introduction page — no translation needed")
    return True


def _ocr_bubbles(
    img_cv: np.ndarray, text_bubbles: list[dict], page_idx: int, errors: ErrorLog | None
) -> None:
    logger.debug("\n── OCR ──")
    for i, b in enumerate(text_bubbles):
        b["text"] = ocr_region(
            img_cv, b["x"], b["y"], b["width"], b["height"], idx=i + 1, page_idx=page_idx
        )
        b["emotion_hint"] = _infer_emotion_tag(b.get("text", ""))
        if not b["text"] and errors:
            errors.add(
                page_idx,
                "ocr_empty",
                f"Bubble #{i + 1}: OCR returned no text after two passes",
                bubble_idx=i + 1,
                bbox=f"({b['x']},{b['y']},{b['width']}x{b['height']})",
                crop_index=i + 1,
            )


def _analyze_and_attribute(
    image_path: str,
    text_bubbles: list[dict],
    archive: CharacterArchive,
    manga_ctx: MangaContext,
    page_idx: int,
    fast_mode: bool,
    errors: ErrorLog | None,
    stage,
) -> tuple[str, str, str]:
    if fast_mode:
        logger.debug("\n[fast mode] skipping analyze + attribute stages")
        for b in text_bubbles:
            b["speaker"] = "unknown"
            b["gender"] = "unknown"
        return "CHARACTERS ON THIS PAGE: unknown", "", ""

    logger.debug("\n── Page analysis ──")
    stage("stage_analyze")
    characters_context, page_context, page_summary = analyze_page_full(
        image_path, archive, manga_ctx, page_idx
    )
    logger.debug(characters_context)
    logger.debug(f"  context: {page_context}")
    logger.debug(f"  summary: {page_summary}")
    if characters_context == "CHARACTERS ON THIS PAGE: unknown" and errors:
        errors.add(
            page_idx,
            "character_parse_failed",
            "Character analysis produced no result — JSON did not parse",
            raw_response_snippet=characters_context,
        )

    logger.debug("\n── Attribution ──")
    stage("stage_attribute")
    attribute_bubbles(image_path, text_bubbles, page_context, characters_context, archive)
    for i, b in enumerate(text_bubbles):
        logger.debug(
            f"  [{i + 1}] {b.get('speaker', '?')} ({b.get('gender', '?')}): "
            f"{b.get('text', '')[:40]}"
        )
        if b.get("text") and b.get("speaker") == "unknown" and errors:
            errors.add(
                page_idx,
                "speaker_unknown",
                f"Bubble #{i + 1}: could not determine speaker",
                bubble_idx=i + 1,
                text=b.get("text", "")[:100],
            )

    return characters_context, page_context, page_summary


def _print_page_stats(text_bubbles: list[dict], output_path: str) -> None:
    empty_ocr = sum(1 for b in text_bubbles if not b.get("text"))
    no_translation = sum(1 for b in text_bubbles if b.get("text") and not b.get("translation"))
    err_translation = sum(1 for b in text_bubbles if b.get("translation") == "[error]")
    ok = len(text_bubbles) - empty_ocr - no_translation - err_translation
    logger.debug(f"\nSaved: {output_path}")
    logger.warning(
        f"  ✓ translated: {ok} | ⚠ no translation: {no_translation} | "
        f"✗ OCR empty: {empty_ocr} | ✗ translation error: {err_translation}"
    )


def process_page(
    image_path: str,
    page_idx: int,
    manga_ctx: MangaContext,
    archive: CharacterArchive,
    output_path: str,
    target_lang: str = "Russian",
    debug: bool = False,
    fast_mode: bool = False,
    errors: ErrorLog | None = None,
    on_stage=None,
):
    def stage(key):
        if on_stage:
            try:
                on_stage(page_idx, key)
            except Exception:
                pass

    logger.debug(f"\n{'=' * 60}")
    logger.debug(f"Page {page_idx}: {image_path}")
    if fast_mode:
        logger.debug("[fast mode] skipping page analysis and speaker attribution")
    logger.debug(f"{'=' * 60}")

    stage("stage_detect")
    img_cv = read_image(image_path)
    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    text_bubbles = _detect_text_bubbles(img_cv, image_pil)
    logger.debug(f"Bubbles: {len(text_bubbles)}")

    if not text_bubbles and errors:
        errors.add(page_idx, "no_bubbles", "No text bubbles found on this page", image=image_path)

    if not fast_mode and len(text_bubbles) <= 2:
        if _handle_intro_page(image_path, archive, manga_ctx, page_idx, output_path, img_cv):
            return text_bubbles

    stage("stage_ocr")
    _ocr_bubbles(img_cv, text_bubbles, page_idx, errors)

    _, page_context, page_summary = _analyze_and_attribute(
        image_path,
        text_bubbles,
        archive,
        manga_ctx,
        page_idx,
        fast_mode,
        errors,
        stage,
    )

    logger.debug("\n── Translation ──")
    stage("stage_translate")
    translate_batch(
        text_bubbles,
        page_context,
        manga_ctx,
        target_lang,
        retries=settings().translate_retries,
        errors=errors,
        page_idx=page_idx,
    )
    for i, b in enumerate(text_bubbles):
        logger.debug(f"  [{i + 1}] {b.get('text', '')[:25]} → {b.get('translation', '')[:40]}")

    manga_ctx.update(page_summary, page_idx)

    stage("stage_inpaint")
    annotated = draw_results(
        img_cv,
        text_bubbles,
        debug=debug,
        page_name=os.path.splitext(os.path.basename(output_path))[0],
    )
    write_image(output_path, annotated)

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


def process_directory(
    input_dir: str,
    output_dir: str = "results",
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
    cancel_event=None,
):
    config = updated_settings(
        llm_model=llm_model,
        ollama_url=ollama_url,
        font_path=font_path,
        detect_threshold=detect_threshold,
        sfx_threshold=sfx_threshold,
        min_bubble_area=min_bubble_area,
        max_font_size=max_font_size,
        inpaint_shrink=inpaint_shrink,
        chunk_size=chunk_size,
        translate_retries=translate_retries,
    )
    if not config.llm_model:
        raise ValueError("No LLM model selected. Pass llm_model=...")
    with use_settings(config):
        return _process_directory(
            input_dir,
            output_dir,
            target_lang,
            font_path,
            debug,
            fast_mode,
            error_log_path,
            on_page_done,
            on_start,
            on_finish,
            on_stage,
            cancel_event,
        )


def _process_directory(
    input_dir,
    output_dir,
    target_lang,
    font_path,
    debug,
    fast_mode,
    error_log_path,
    on_page_done,
    on_start,
    on_finish,
    on_stage,
    cancel_event,
):

    resolved = resolve_font(font_path)
    if resolved:
        if resolved == font_path:
            logger.debug(f"[font] Using: {resolved}")
        else:
            logger.warning(f"[font] '{font_path}' not found — auto-selected: {resolved}")
    else:
        logger.warning(
            "[font] ⚠ No usable font found, including system fallbacks. "
            "Will use PIL default (text may render poorly)."
        )

    files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_key,
    )
    if not files:
        raise ValueError(f"No images found in {input_dir}")

    logger.debug(f"Pages found: {len(files)}")
    os.makedirs(output_dir, exist_ok=True)

    manga_ctx = MangaContext()
    archive = CharacterArchive("characters.json")
    errors = ErrorLog(error_log_path)

    if on_start:
        on_start(len(files))

    total_start = time.perf_counter()
    page_times: list[float] = []
    failed = 0

    for page_idx, filename in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            logger.debug("[abort] Translation cancelled by user.")
            break
        input_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}_translated.png")

        page_start = time.perf_counter()
        bubbles_result = []
        try:
            with use_settings(updated_settings(font_path=resolved)):
                bubbles_result = (
                    process_page(
                        input_path,
                        page_idx,
                        manga_ctx,
                        archive,
                        output_path,
                        target_lang,
                        debug=debug,
                        fast_mode=fast_mode,
                        errors=errors,
                        on_stage=on_stage,
                    )
                    or []
                )
            elapsed = time.perf_counter() - page_start
            page_times.append(elapsed)
            logger.debug(f"  ⏱  page processed in {format_duration(elapsed)}")
        except Exception as e:
            failed += 1
            elapsed = time.perf_counter() - page_start
            logger.warning(f"\n[ERROR] {filename}: {e}")
            import traceback

            traceback.print_exc()
            errors.add(
                page_idx,
                "page_failed",
                f"Page not processed: {type(e).__name__}: {e}",
                filename=filename,
                traceback=traceback.format_exc(),
            )

        if on_page_done:
            try:
                on_page_done(page_idx, len(files), filename, output_path, bubbles_result, elapsed)
            except Exception as cb_err:
                logger.debug(f"  [callback warn] on_page_done: {cb_err}")

    total_elapsed = time.perf_counter() - total_start

    errors.save()

    logger.debug(f"\n{'=' * 60}")
    logger.debug("Done!")
    logger.debug(f"  Pages processed:    {len(page_times)} / {len(files)}")
    if failed:
        logger.warning(f"  Errors:             {failed}")
    logger.debug(f"  Total time:         {format_duration(total_elapsed)}")
    if page_times:
        avg = sum(page_times) / len(page_times)
        logger.debug(f"  Avg per page:       {format_duration(avg)}")
        logger.debug(f"  Fastest:            {format_duration(min(page_times))}")
        logger.debug(f"  Slowest:            {format_duration(max(page_times))}")
    logger.debug(f"  Results:            {output_dir}")
    logger.debug(f"  Character archive:  characters.json ({len(archive.characters)} characters)")

    if errors.entries:
        logger.warning(f"\n  ⚠ Issues recorded: {len(errors.entries)}")
        for kind, count in sorted(errors.summary().items(), key=lambda x: -x[1]):
            logger.debug(f"     {kind:30s} {count}")
        logger.debug(f"  Details:            {error_log_path}")
    else:
        logger.debug("  ✓ No issues recorded")

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
    import argparse

    parser = argparse.ArgumentParser(description="Translate manga pages using local models")
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--ollama-url")
    parser.add_argument("--target-lang", default="Russian")
    parser.add_argument("--font-path", default="arial.ttf")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    arguments = vars(parser.parse_args())
    logging.basicConfig(level=logging.DEBUG if arguments["debug"] else logging.WARNING)
    process_directory(**arguments)
