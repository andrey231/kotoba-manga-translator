import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context import CharacterArchive, ErrorLog, MangaContext
from fonts import resolve_font
from image_io import SUPPORTED_EXTENSIONS, read_image, write_image
from llm import model_phase, unload_model
from models import detect_bubbles, release_gpu_memory
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
        image_path, archive, manga_ctx, page_idx, text_bubbles
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


def _stage_callback(page_idx, on_stage):
    def stage(key):
        if on_stage:
            try:
                on_stage(page_idx, key)
            except Exception as error:
                logger.warning("Stage callback failed: %s", error)

    return stage


@dataclass
class _PageState:
    filename: str
    bubbles: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    failed: bool = False
    intro: bool = False


def _prepare_page(image_path, page_idx, errors, on_stage):
    stage = _stage_callback(page_idx, on_stage)
    stage("stage_detect")
    image = read_image(image_path)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    bubbles = _detect_text_bubbles(image, image_pil)
    if not bubbles and errors:
        errors.add(page_idx, "no_bubbles", "No text bubbles found on this page", image=image_path)
    stage("stage_ocr")
    _ocr_bubbles(image, bubbles, page_idx, errors)
    stage("stage_prepared")
    return bubbles


def _translate_page(
    image_path,
    page_idx,
    bubbles,
    manga_ctx,
    archive,
    target_lang,
    fast_mode,
    errors,
    on_stage,
):
    stage = _stage_callback(page_idx, on_stage)
    if not fast_mode and len(bubbles) <= 2:
        stage("stage_analyze")
        if _handle_intro_page(image_path, archive, manga_ctx, page_idx):
            stage("stage_translated")
            return True
    _, page_context, page_summary = _analyze_and_attribute(
        image_path,
        bubbles,
        archive,
        manga_ctx,
        page_idx,
        fast_mode,
        errors,
        stage,
    )
    stage("stage_translate")
    translate_batch(
        bubbles,
        page_context,
        manga_ctx,
        target_lang,
        retries=settings().translate_retries,
        errors=errors,
        page_idx=page_idx,
    )
    manga_ctx.update(page_summary, page_idx)
    stage("stage_translated")
    return False


def _render_page(image_path, page_idx, bubbles, output_path, debug, unchanged, on_stage):
    stage = _stage_callback(page_idx, on_stage)
    stage("stage_inpaint")
    image = read_image(image_path)
    try:
        annotated = (
            image
            if unchanged
            else draw_results(
                image,
                bubbles,
                debug=debug,
                page_name=Path(output_path).stem,
            )
        )
        write_image(output_path, annotated)
    finally:
        for bubble in bubbles:
            bubble.pop("_text_mask", None)


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
    unload_model(settings().llm_model)
    try:
        bubbles = _prepare_page(image_path, page_idx, errors, on_stage)
        with model_phase():
            intro = _translate_page(
                image_path,
                page_idx,
                bubbles,
                manga_ctx,
                archive,
                target_lang,
                fast_mode,
                errors,
                on_stage,
            )
        _render_page(image_path, page_idx, bubbles, output_path, debug, intro, on_stage)
        return bubbles
    finally:
        release_gpu_memory()


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

    pages = sorted(
        [
            _PageState(filename=f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda page: natural_key(page.filename),
    )
    if not pages:
        raise ValueError(f"No images found in {input_dir}")

    logger.debug(f"Pages found: {len(pages)}")
    os.makedirs(output_dir, exist_ok=True)

    manga_ctx = MangaContext()
    archive = CharacterArchive("characters.json")
    errors = ErrorLog(error_log_path)

    if on_start:
        on_start(len(pages))

    total_start = time.perf_counter()
    page_times: list[float] = []
    failed = 0

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    def page_failed(page_idx, page, error):
        nonlocal failed
        if not page.failed:
            failed += 1
        page.failed = True
        logger.warning("Page %s failed: %s", page.filename, error)
        errors.add(
            page_idx, "page_failed", f"{type(error).__name__}: {error}", filename=page.filename
        )

    try:
        if not cancelled():
            unload_model(settings().llm_model)
        with use_settings(updated_settings(font_path=resolved)):
            for page_idx, page in enumerate(pages, start=1):
                if cancelled():
                    break
                page_start = time.perf_counter()
                try:
                    page.bubbles = _prepare_page(
                        os.path.join(input_dir, page.filename),
                        page_idx,
                        errors,
                        on_stage,
                    )
                except Exception as error:
                    page_failed(page_idx, page, error)
                page.elapsed += time.perf_counter() - page_start

            if not cancelled() and any(not page.failed for page in pages):
                with model_phase():
                    for page_idx, page in enumerate(pages, start=1):
                        if cancelled():
                            break
                        if page.failed:
                            continue
                        page_start = time.perf_counter()
                        try:
                            page.intro = _translate_page(
                                os.path.join(input_dir, page.filename),
                                page_idx,
                                page.bubbles,
                                manga_ctx,
                                archive,
                                target_lang,
                                fast_mode,
                                errors,
                                on_stage,
                            )
                        except Exception as error:
                            page_failed(page_idx, page, error)
                        page.elapsed += time.perf_counter() - page_start

            for page_idx, page in enumerate(pages, start=1):
                if cancelled():
                    break
                page_start = time.perf_counter()
                output_path = os.path.join(
                    output_dir, f"{os.path.splitext(page.filename)[0]}_translated.png"
                )
                try:
                    _render_page(
                        os.path.join(input_dir, page.filename),
                        page_idx,
                        page.bubbles,
                        output_path,
                        debug,
                        page.failed or page.intro,
                        on_stage,
                    )
                except Exception as error:
                    page_failed(page_idx, page, error)
                page.elapsed += time.perf_counter() - page_start
                if not page.failed:
                    page_times.append(page.elapsed)
                if on_page_done:
                    try:
                        on_page_done(
                            page_idx,
                            len(pages),
                            page.filename,
                            output_path,
                            page.bubbles if not page.failed else [],
                            page.elapsed,
                        )
                    except Exception as error:
                        logger.warning("Page callback failed: %s", error)
    finally:
        release_gpu_memory()
        errors.save()

    total_elapsed = time.perf_counter() - total_start

    logger.debug(f"\n{'=' * 60}")
    logger.debug("Done!")
    logger.debug(f"  Pages processed:    {len(page_times)} / {len(pages)}")
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
        "total_pages": len(pages),
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
