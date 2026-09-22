import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from bubbles import source_box
from image_io import clip_box, write_image
from models import DEVICE, _ctd_page_mask, get_inpaint_model
from settings import settings

logger = logging.getLogger(__name__)

_LARGE_BUBBLE_PX = 80_000


_INPAINT_MAX_PIXELS = 12_000_000


CTD_MIN_COVERAGE = 0.05


INK_DARK_PERCENTILE = 25


INK_LIGHT_PERCENTILE = 75


BALANCED_LINE_TOLERANCE = 1.3


MIN_EFFECTIVE_BOX = 20


VERTICAL_TEXT_ANGLE = 45


ROTATE_SUPERSAMPLE = 2


MIN_FONT_SIZE = 2


OUTLINE_DELTA = 60


OUTLINE_BLEND_MARGIN = 30


def _largest_subrect(
    bx: int, by: int, bw: int, bh: int, ox: int, oy: int, ow: int, oh: int
) -> tuple[int, int, int, int]:
    ix1 = max(bx, ox)
    iy1 = max(by, oy)
    ix2 = min(bx + bw, ox + ow)
    iy2 = min(by + bh, oy + oh)
    if ix1 >= ix2 or iy1 >= iy2:
        return bx, by, bw, bh

    options: list[tuple[int, int, int, int]] = []
    nw = ox - bx
    if nw > 0:
        options.append((bx, by, nw, bh))
    nx = ox + ow
    nw2 = (bx + bw) - nx
    if nw2 > 0:
        options.append((nx, by, nw2, bh))
    nh = oy - by
    if nh > 0:
        options.append((bx, by, bw, nh))
    ny = oy + oh
    nh2 = (by + bh) - ny
    if nh2 > 0:
        options.append((bx, ny, bw, nh2))
    if not options:
        return bx, by, 0, 0
    return max(options, key=lambda r: r[2] * r[3])


def _clip_overlapping_boxes(bubbles: list[dict], min_area: int = 400) -> list[dict]:
    if len(bubbles) <= 1:
        return bubbles

    order = sorted(
        range(len(bubbles)),
        key=lambda i: (
            bubbles[i]["width"] * bubbles[i]["height"],
            bubbles[i]["y"],
            bubbles[i]["x"],
        ),
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
        logger.debug(f"  [clip] {dropped} bubble(s) dropped (too small after clipping)")
    clipped_count = sum(
        1
        for orig, res in zip(
            sorted(bubbles, key=lambda b: b["width"] * b["height"]),
            result,
        )
        if orig["x"] != res["x"]
        or orig["y"] != res["y"]
        or orig["width"] != res["width"]
        or orig["height"] != res["height"]
    )
    if clipped_count:
        logger.debug(f"  [clip] {clipped_count} bubble(s) clipped to avoid overlap")
    return result


def _binarize_text_mask(crop_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _estimate_text_angle(mask: np.ndarray) -> float:
    connect = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    blobs = cv2.dilate(mask, connect)
    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sin2 = cos2 = 0.0
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw < 1 or rh < 1 or max(rw, rh) / min(rw, rh) < 2.0:
            continue
        box = cv2.boxPoints(rect)
        p1, p2 = max(
            ((box[i], box[(i + 1) % 4]) for i in range(4)),
            key=lambda e: float(np.hypot(*(e[1] - e[0]))),
        )
        d = p2 - p1
        length = float(np.hypot(d[0], d[1]))
        a = float(np.arctan2(d[1], d[0]))
        sin2 += length * np.sin(2 * a)
        cos2 += length * np.cos(2 * a)

    if sin2 == 0.0 and cos2 == 0.0:
        return 0.0
    avg = 0.5 * float(np.degrees(np.arctan2(sin2, cos2)))
    if avg > 90:
        avg -= 180
    if avg <= -90:
        avg += 180
    if abs(avg) > 60:
        return 0.0
    return avg


def _estimate_text_size(mask: np.ndarray) -> int | None:

    rows = (mask > 0).sum(axis=1).astype(float)
    if rows.max() <= 0:
        return None
    active = rows > rows.max() * 0.1
    bands: list[int] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            bands.append(i - start)
            start = None
    if start is not None:
        bands.append(len(active) - start)
    bands = [h for h in bands if h >= 3]
    if not bands:
        return None
    line_h = float(np.median(bands))
    font = int(round(line_h * 1.1))
    return max(MIN_FONT_SIZE, min(font, settings().max_font_size))


def _compute_text_masks(img_cv: np.ndarray, bubbles: list[dict], page_name: str = "") -> None:
    active = [b for b in bubbles if b.get("translation") and b.get("_text_mask") is None]
    if not active:
        return

    page_mask = _ctd_page_mask(img_cv)
    if page_mask is None:
        return

    h_img, w_img = img_cv.shape[:2]
    _dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for b in active:
        x, y, bw, bh = source_box(b)
        x, y, x2, y2 = clip_box(x, y, bw, bh, w_img, h_img)
        if x2 <= x or y2 <= y:
            b["_text_mask"] = None
            continue

        ctd_crop = page_mask[y:y2, x:x2]
        crop_bgr = img_cv[y:y2, x:x2]

        otsu_crop = _binarize_text_mask(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        ctd_dilated = cv2.dilate(ctd_crop, _dilate_k)
        otsu_in_zone = cv2.bitwise_and(otsu_crop, ctd_dilated)
        combined = cv2.bitwise_or(ctd_crop, otsu_in_zone)

        b["_text_mask"] = combined

        b.setdefault("text_angle", _estimate_text_angle(combined))
        b["_text_size"] = _estimate_text_size(combined)

        if settings().mask_debug_dir:
            idx = b.get("idx", id(b))
            overlay = crop_bgr.copy()
            overlay[combined > 0] = (0, 200, 0)
            vis = cv2.addWeighted(crop_bgr, 0.5, overlay, 0.5, 0)
            prefix = f"{page_name}_" if page_name else ""
            dbg_path = os.path.join(settings().mask_debug_dir, f"{prefix}b{idx:03d}.png")
            write_image(dbg_path, vis)
            logger.debug(f"  [ctd-dbg] → {dbg_path}")


def detect_text_color(img_cv: np.ndarray, bubble: dict, default: tuple = (0, 0, 0)) -> tuple:
    ih, iw = img_cv.shape[:2]
    x, y, x2, y2 = clip_box(*source_box(bubble), iw, ih)
    w, h = x2 - x, y2 - y
    if w <= 0 or h <= 0:
        return default
    crop = img_cv[y:y2, x:x2]
    if crop.size == 0:
        return default

    text_mask = bubble.get("_text_mask")
    if text_mask is not None:
        crop_mask = text_mask
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


def detect_text_style(img_cv: np.ndarray, bubble: dict) -> tuple:
    fill = detect_text_color(img_cv, bubble)
    ih, iw = img_cv.shape[:2]
    x, y, x2, y2 = clip_box(*source_box(bubble), iw, ih)
    mask_full = bubble.get("_text_mask")
    if mask_full is None or x2 <= x or y2 <= y:
        return fill, None, 0

    crop = img_cv[y:y2, x:x2]
    mask = mask_full
    if cv2.countNonZero(mask) < 30:
        return fill, None, 0

    width_hint = max(2, min(x2 - x, y2 - y) // 25)

    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_t = cv2.erode(mask, er)
    core = cv2.erode(mask_t, er)
    edge = cv2.subtract(mask_t, core)
    core_px = crop[core > 0]
    edge_px = crop[edge > 0]
    bg_px = crop[mask == 0]
    if len(core_px) >= 15 and len(edge_px) >= 15 and len(bg_px) >= 15:
        core_med = np.median(core_px, axis=0).astype(int)
        edge_med = np.median(edge_px, axis=0).astype(int)
        bg_med = np.median(bg_px, axis=0).astype(int)

        core_b = float(core_med.mean())
        edge_b = float(edge_med.mean())
        bg_b = float(bg_med.mean())
        lo, hi = min(core_b, bg_b), max(core_b, bg_b)
        is_blend = (lo - OUTLINE_BLEND_MARGIN) <= edge_b <= (hi + OUTLINE_BLEND_MARGIN)
        distinct = int(np.abs(core_med - edge_med).sum())
        if distinct > OUTLINE_DELTA and not is_blend:
            cb, cg, cr = core_med
            ob, og, orr = edge_med
            return (int(cr), int(cg), int(cb)), (int(orr), int(og), int(ob)), width_hint

    return fill, None, 0


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1 : h + 1, 1 : w + 1] = mask
    inv = cv2.bitwise_not(padded)
    cv2.floodFill(inv, None, (0, 0), 0)
    holes = inv[1 : h + 1, 1 : w + 1]
    return cv2.bitwise_or(mask, holes)


def build_inpaint_mask(img_cv: np.ndarray, bubbles: list[dict], shrink: int = 1) -> np.ndarray:
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    connect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closing_k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))

    for b in bubbles:
        if not b.get("translation"):
            continue
        x, y, bw, bh = source_box(b)
        x0 = max(0, x + shrink)
        y0 = max(0, y + shrink)
        x1 = min(w, x + bw - shrink)
        y1 = min(h, y + bh - shrink)
        if x1 <= x0 or y1 <= y0:
            continue

        text_mask = b.get("_text_mask")
        if text_mask is not None:
            crop_m = text_mask[y0 - max(0, y) : y1 - max(0, y), x0 - max(0, x) : x1 - max(0, x)]
            coverage = np.count_nonzero(crop_m) / max(crop_m.size, 1)
            if coverage >= CTD_MIN_COVERAGE and crop_m.any():
                closed = cv2.morphologyEx(crop_m, cv2.MORPH_CLOSE, closing_k)
                glyph = cv2.dilate(closed, inpaint_k)

                rows_with_text = np.any(crop_m > 0, axis=1)
                if rows_with_text.any():
                    r_top = int(np.argmax(rows_with_text))
                    r_bot = int(len(rows_with_text) - 1 - np.argmax(rows_with_text[::-1]))
                    sub_gray = img_gray[y0 + r_top : y0 + r_bot + 1, x0:x1]
                    otsu_sub = _binarize_text_mask(sub_gray)
                    otsu_sub = cv2.dilate(
                        otsu_sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    )
                    otsu_full = np.zeros(crop_m.shape, dtype=np.uint8)
                    otsu_full[r_top : r_bot + 1, :] = otsu_sub
                    glyph = cv2.bitwise_or(glyph, otsu_full)

                glyph = _fill_mask_holes(glyph)
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
            contours, _ = cv2.findContours(glyph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    mask_np = build_inpaint_mask(img_cv, bubbles, shrink=settings().inpaint_shrink)

    if not mask_np.any():
        return img_cv.copy()

    model = get_inpaint_model()
    if model is None:
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)

    h_px, w_px = img_cv.shape[:2]
    if h_px * w_px > _INPAINT_MAX_PIXELS:
        logger.debug(f"  [inpaint] page {w_px}x{h_px} exceeds LaMa pixel cap → cv2.inpaint")
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

        if result_cv.shape != img_cv.shape:
            raise ValueError("LaMa returned an unexpected image shape")
        out = img_cv.copy()
        np.copyto(out, result_cv, where=(mask_np > 0)[..., None])
        return out

    except Exception as e:
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.warning(f"  [inpaint ⚠] LaMa failed ({e}), falling back to cv2.inpaint")
        return cv2.inpaint(img_cv, mask_np, 3, cv2.INPAINT_TELEA)


def fit_text_in_box(
    draw,
    text: str,
    box_w: int,
    box_h: int,
    font_path: str | None = None,
    max_size: int | None = None,
) -> tuple:
    if font_path is None:
        font_path = settings().font_path
    upper = (
        settings().max_font_size
        if not max_size
        else max(MIN_FONT_SIZE, min(max_size, settings().max_font_size))
    )
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

    def wrap_greedy(words: list[str], font, allow_hyphenation: bool = False) -> list[str] | None:
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
            elif (
                len(test) <= target_line_chars * BALANCED_LINE_TOLERANCE
                and text_width(test, font) <= usable_w
            ):
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

    def best_wrap_for_size(
        font, allow_hyphenation: bool = False
    ) -> tuple[list[str], int, int, int] | None:
        paragraphs = text.split("\n")

        if len(paragraphs) > 1:
            all_lines: list[str] = []
            for para in paragraphs:
                para_words = para.split()
                if not para_words:
                    all_lines.append("")
                    continue
                wrapped = wrap_greedy(para_words, font, allow_hyphenation=allow_hyphenation)
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
        lo, hi = MIN_FONT_SIZE, upper
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


def _effective_box(
    bx: int, by: int, bw: int, bh: int, smaller_bubbles: list[dict]
) -> tuple[int, int, int, int]:
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


def _render_text_block(
    text: str,
    box_w: int,
    box_h: int,
    color: tuple,
    font_path: str | None = None,
    font_size_override: int | None = None,
    text_align: str = "center",
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    outline_color: tuple | None = None,
    outline_width: int = 0,
    max_size: int | None = None,
) -> Image.Image:
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    eff_font = font_path
    sim_bold = False
    if bold or italic:
        variant = _find_font_variant(font_path or settings().font_path, bold, italic)
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
            draw, text, box_w, box_h, eff_font, max_size=max_size
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

        draw.text(
            (text_x, text_y),
            line,
            fill=rgba_color,
            font=font,
            stroke_fill=stroke_fill,
            stroke_width=stroke_w,
        )

        if underline and line.strip():
            ascent, _ = font.getmetrics()
            uy = text_y + ascent + 1
            thickness = max(1, font.size // 14)
            draw.rectangle([text_x, uy, text_x + line_w, uy + thickness], fill=rgba_color)

        text_y += line_heights[j] + spacing

    return img


def _fit_at_exact_size(
    draw, text: str, box_w: int, box_h: int, font_path: str | None, size: int
) -> tuple:
    if font_path is None:
        font_path = settings().font_path
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        try:
            font = ImageFont.truetype(settings().font_path, size)
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
    return any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in text)


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
        max_size=b.get("_text_size"),
    )


def _render_bubble_block(
    translation: str, color: tuple, style: dict, box: tuple, eff: tuple, text_angle: float
) -> Image.Image:
    bx, by, bw, bh = box
    ex, ey, ew, eh = eff
    off_x, off_y = ex - bx, ey - by

    if abs(text_angle) > VERTICAL_TEXT_ANGLE:
        rot_deg = -90 if text_angle > 0 else 90
        block = _render_text_block(translation, bh, bw, color, **style)
        block = block.rotate(rot_deg, expand=True, resample=Image.Resampling.BICUBIC)
        bw_r, bh_r = block.size
        cx_r = max(0, (bw_r - bw) // 2)
        cy_r = max(0, (bh_r - bh) // 2)
        return block.crop((cx_r + off_x, cy_r + off_y, cx_r + off_x + ew, cy_r + off_y + eh))

    if abs(text_angle) > 1:
        ss = ROTATE_SUPERSAMPLE
        ss_style = dict(style)
        if ss_style.get("max_size"):
            ss_style["max_size"] = ss_style["max_size"] * ss
        big = _render_text_block(translation, bw * ss, bh * ss, color, **ss_style)
        big = big.rotate(-text_angle, expand=True, resample=Image.Resampling.BICUBIC)
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
        font_small = ImageFont.truetype(settings().font_path, 14)
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

        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
        draw.rectangle([(x, y - 22), (x + 18, y)], fill=color)
        draw.text((x + 3, y - 20), str(i + 1), fill=(0, 0, 0), font=font_small)


def draw_results(
    img_cv: np.ndarray, bubbles: list[dict], debug: bool = False, page_name: str = ""
) -> np.ndarray:
    logger.debug("  Segmenting text regions (CTD)...")
    _compute_text_masks(img_cv, bubbles, page_name=page_name)

    for b in bubbles:
        if b.get("translation") and b.get("text_color") is None:
            fill, outline, outline_w = detect_text_style(img_cv, b)
            b["text_color"] = fill
            if outline is not None and not b.get("outline_color"):
                b["outline_color"] = outline
                b["outline_width"] = outline_w

    logger.debug("  Inpainting original text (LaMa)...")
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
        color = b.get("text_color", (0, 0, 0))

        smaller = [s for s in render_order[i + 1 :] if s.get("translation")]
        ex, ey, ew, eh = _effective_box(bx, by, bw, bh, smaller)
        if ew < MIN_EFFECTIVE_BOX or eh < MIN_EFFECTIVE_BOX:
            continue

        text_angle = b.get("text_angle", 0.0)
        if abs(text_angle) > VERTICAL_TEXT_ANGLE and _contains_cjk(b.get("text", "")):
            logger.debug(
                f"  [jp→ru] bubble {i + 1} angle={text_angle:.0f}° japanese → render horizontal"
            )
            text_angle = 0.0

        block = _render_bubble_block(
            translation,
            color,
            _extract_bubble_style(b),
            (bx, by, bw, bh),
            (ex, ey, ew, eh),
            text_angle,
        )
        pil.paste(block, (ex, ey), block)

    if debug:
        _draw_debug_overlay(pil, bubbles)

    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
