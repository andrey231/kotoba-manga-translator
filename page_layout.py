from collections.abc import Sequence
from functools import cmp_to_key

import cv2
import numpy as np

BUBBLE_CONTAIN_FRAC = 0.7
ORPHAN_TEXT_FRAC = 0.3
ORPHAN_INSET_FRAC = 0.12
ORPHAN_MIN_TEXT_PX = 60
ORPHAN_TEXT_PAD = 4

MISSED_KNOWN_PAD = 6
MISSED_MERGE_PX = 15
MISSED_MIN_INK_PX = 120
MISSED_MIN_SIDE = 10
MISSED_MIN_LONG_SIDE = 24
MISSED_GAP_FRAC = 0.6
MISSED_BOX_PAD = 4
MISSED_MAX_REGIONS = 20
MIN_VERIFIED_LETTERS = 2

PAPER_MIN_LEVEL = 150
PAPER_MAX_STD = 18.0
PAPER_INK_DELTA = 40
PAPER_LEAK_FRAC = 0.15
INTERIOR_EDGE_FRAC = 0.03
INTERIOR_MIN_EDGE = 2
ELLIPSE_SHRINK = 0.94

GUTTER_INK_FRAC = 0.003
GUTTER_LOOSE_FRAC = 0.03
BORDER_LINE_FRAC = 0.5
BORDER_SEARCH_PX = 4
GUTTER_MIN_PX = 3
INK_LEVEL = 128
SAME_LINE_OVERLAP = 0.3

_KANA = ((chr(0x3040), chr(0x309F)), (chr(0x30A0), chr(0x30FF)))
_HAN = ((chr(0x3400), chr(0x9FFF)),)
_HANGUL = ((chr(0xAC00), chr(0xD7AF)), (chr(0x1100), chr(0x11FF)))


def _box(region: dict) -> tuple[int, int, int, int]:
    return region["x"], region["y"], region["width"], region["height"]


def _intersection(a: Sequence[int], b: Sequence[int]) -> int:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def attach_bubble_boxes(texts: Sequence[dict], bubbles: Sequence[dict]) -> None:
    for text in texts:
        box = _box(text)
        area = max(1, box[2] * box[3])
        holders = [b for b in bubbles if _intersection(box, _box(b)) >= BUBBLE_CONTAIN_FRAC * area]
        if holders:
            holder = min(holders, key=lambda b: b["width"] * b["height"])
            text["bubble_box"] = list(_box(holder))


def orphan_bubbles(texts: Sequence[dict], bubbles: Sequence[dict]) -> list[dict]:
    return [
        bubble
        for bubble in bubbles
        if not any(
            _intersection(_box(t), _box(bubble)) >= ORPHAN_TEXT_FRAC * t["width"] * t["height"]
            for t in texts
        )
    ]


def texts_in_bubbles(bubbles: Sequence[dict], text_mask: np.ndarray | None) -> list[dict]:
    if text_mask is None:
        return []
    recovered = []
    for bubble in bubbles:
        bx, by, bw, bh = _box(bubble)
        inset_x = int(bw * ORPHAN_INSET_FRAC)
        inset_y = int(bh * ORPHAN_INSET_FRAC)
        x0, y0 = bx + inset_x, by + inset_y
        x1, y1 = bx + bw - inset_x, by + bh - inset_y
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.nonzero(text_mask[y0:y1, x0:x1])
        if len(xs) < ORPHAN_MIN_TEXT_PX:
            continue

        tx0 = max(bx, x0 + int(xs.min()) - ORPHAN_TEXT_PAD)
        ty0 = max(by, y0 + int(ys.min()) - ORPHAN_TEXT_PAD)
        tx1 = min(bx + bw, x0 + int(xs.max()) + 1 + ORPHAN_TEXT_PAD)
        ty1 = min(by + bh, y0 + int(ys.max()) + 1 + ORPHAN_TEXT_PAD)
        recovered.append(
            {
                "class": "text_bubble",
                "x": tx0,
                "y": ty0,
                "width": tx1 - tx0,
                "height": ty1 - ty0,
                "confidence": bubble["confidence"],
                "speaker": "unknown",
                "gender": "unknown",
                "bubble_box": [bx, by, bw, bh],
            }
        )
    return recovered


def _gap(a: Sequence[int], b: Sequence[int]) -> int:
    dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return max(dx, dy)


def _merge_close(boxes: list[list[int]]) -> list[list[int]]:
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                limit = MISSED_GAP_FRAC * max(min(a[2], a[3]), min(b[2], b[3]))
                if _gap(a, b) > limit:
                    continue
                x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                x1 = max(a[0] + a[2], b[0] + b[2])
                y1 = max(a[1] + a[3], b[1] + b[3])
                boxes[i] = [x0, y0, x1 - x0, y1 - y0, a[4] + b[4]]
                del boxes[j]
                merged = True
                break
            if merged:
                break
    return boxes


def missed_text_regions(text_mask: np.ndarray | None, known: Sequence[dict]) -> list[dict]:
    if text_mask is None:
        return []
    free = text_mask.copy()
    for region in known:
        x, y, w, h = _box(region)
        pad = MISSED_KNOWN_PAD
        free[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MISSED_MERGE_PX, MISSED_MERGE_PX))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(free, kernel), connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        ink = int(np.count_nonzero(free[y : y + h, x : x + w]))
        if ink >= MISSED_MIN_INK_PX // 4:
            boxes.append([x, y, w, h, ink])

    ih, iw = text_mask.shape
    regions = []
    for x, y, w, h, ink in sorted(_merge_close(boxes), key=lambda b: -b[4]):
        if ink < MISSED_MIN_INK_PX or min(w, h) < MISSED_MIN_SIDE:
            continue
        if max(w, h) < MISSED_MIN_LONG_SIDE:
            continue
        x0, y0 = max(0, x - MISSED_BOX_PAD), max(0, y - MISSED_BOX_PAD)
        x1, y1 = min(iw, x + w + MISSED_BOX_PAD), min(ih, y + h + MISSED_BOX_PAD)
        regions.append(
            {
                "class": "text_free",
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
                "confidence": 0.0,
                "speaker": "unknown",
                "gender": "unknown",
                "_recovered": True,
            }
        )
        if len(regions) == MISSED_MAX_REGIONS:
            break
    return regions


def is_verified_text(text: str) -> bool:
    return sum(ch.isalnum() for ch in text or "") >= MIN_VERIFIED_LETTERS


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1 : h + 1, 1 : w + 1] = mask
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 255)
    holes = cv2.bitwise_not(outside)[1 : h + 1, 1 : w + 1]
    return cv2.bitwise_or(mask, holes)


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    axes = (max(1, int(w * ELLIPSE_SHRINK / 2)), max(1, int(h * ELLIPSE_SHRINK / 2)))
    cv2.ellipse(mask, (w // 2, h // 2), axes, 0, 0, 360, 255, cv2.FILLED)
    return mask


def bubble_interior(
    img_cv: np.ndarray, bubble_box: Sequence[int], text_box: Sequence[int]
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int]] | None:
    ih, iw = img_cv.shape[:2]
    bx, by, bw, bh = (int(v) for v in bubble_box)
    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(iw, bx + bw), min(ih, by + bh)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = img_cv[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    level = float(np.percentile(gray, 90))
    if level < PAPER_MIN_LEVEL:
        return None
    paper = (gray >= level - PAPER_INK_DELTA).astype(np.uint8) * 255

    tx, ty, tw, th = (int(v) for v in text_box)
    seed = np.zeros_like(paper)
    sx0, sy0 = max(0, tx - x0), max(0, ty - y0)
    sx1, sy1 = min(paper.shape[1], tx + tw - x0), min(paper.shape[0], ty + th - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    seed[sy0:sy1, sx0:sx1] = 255

    count, labels = cv2.connectedComponents(paper, connectivity=4)
    if count < 2:
        return None
    hits = np.bincount(labels[(seed > 0) & (paper > 0)].ravel(), minlength=count)
    hits[0] = 0
    if hits.max() == 0:
        return None
    component = (labels == int(hits.argmax())).astype(np.uint8) * 255

    border = np.concatenate([component[0], component[-1], component[:, 0], component[:, -1]])
    if np.count_nonzero(border) > PAPER_LEAK_FRAC * border.size:
        component = cv2.bitwise_and(component, _ellipse_mask(*component.shape))

    paper_pixels = crop[component > 0]
    if len(paper_pixels) < 50 or float(gray[component > 0].std()) > PAPER_MAX_STD:
        return None

    interior = _fill_holes(component)
    edge = max(INTERIOR_MIN_EDGE, int(round(min(x1 - x0, y1 - y0) * INTERIOR_EDGE_FRAC)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
    interior = cv2.erode(interior, kernel)
    if not interior.any():
        return None
    color = tuple(int(c) for c in np.median(paper_pixels, axis=0))
    return interior, (x0, y0), color


def interior_rect(
    interior: np.ndarray,
    origin: tuple[int, int],
    text_box: Sequence[int],
    blockers: Sequence[Sequence[int]] = (),
) -> tuple[int, int, int, int]:
    ox, oy = origin
    h, w = interior.shape
    inside = interior > 0
    tx, ty, tw, th = (int(v) for v in text_box)
    cx = min(w - 1, max(0, tx + tw // 2 - ox))
    cy = min(h - 1, max(0, ty + th // 2 - oy))
    if not inside[cy, cx]:
        dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
        cy, cx = (int(v) for v in np.unravel_index(int(dist.argmax()), dist.shape))
        if not inside[cy, cx]:
            return tx, ty, tw, th

    left = np.full(h, -1, dtype=np.int64)
    right = np.full(h, -1, dtype=np.int64)
    for row in range(h):
        line = inside[row]
        if not line[cx]:
            continue
        outside_left = np.flatnonzero(~line[:cx])
        outside_right = np.flatnonzero(~line[cx:])
        left[row] = outside_left[-1] + 1 if outside_left.size else 0
        right[row] = cx + outside_right[0] if outside_right.size else w
        for bx, by, bw, bh in blockers:
            if not by - oy <= row < by - oy + bh:
                continue
            if bx - ox + bw <= cx:
                left[row] = max(left[row], bx - ox + bw)
            elif bx - ox > cx:
                right[row] = min(right[row], bx - ox)

    best = (0, cx, cy, cx + 1, cy + 1)
    top_left, top_right = left[cy], right[cy]
    for top in range(cy, -1, -1):
        if left[top] < 0:
            break
        top_left, top_right = max(top_left, left[top]), min(top_right, right[top])
        span_left, span_right = top_left, top_right
        for bottom in range(cy, h):
            if left[bottom] < 0:
                break
            span_left, span_right = max(span_left, left[bottom]), min(span_right, right[bottom])
            if span_right <= span_left:
                break
            area = (span_right - span_left) * (bottom + 1 - top)
            if area > best[0]:
                best = (area, span_left, top, span_right, bottom + 1)
    _, x0, y0, x1, y1 = best
    if (x1 - x0) * (y1 - y0) <= tw * th:
        return tx, ty, tw, th
    return ox + int(x0), oy + int(y0), int(x1 - x0), int(y1 - y0)


def is_right_to_left(texts: Sequence[str], boxes: Sequence[Sequence[int]] = ()) -> bool:
    def has(text: str, ranges) -> bool:
        return any(lo <= ch <= hi for ch in text for lo, hi in ranges)

    filled = [text for text in texts if text and text.strip()]
    if filled:
        if sum(has(text, _KANA) for text in filled) >= 0.3 * len(filled):
            return True
        if sum(has(text, _HANGUL) for text in filled) >= 0.3 * len(filled):
            return False
        if sum(has(text, _HAN) for text in filled) < 0.5 * len(filled):
            return False
    tall = sum(1 for _, _, w, h in boxes if h > 1.3 * w)
    return tall > len(boxes) / 2 if boxes else True


def _gaps(profile: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    loose = (profile <= GUTTER_LOOSE_FRAC) & ~blocked
    runs = []
    start = None
    for i, value in enumerate(loose):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(loose)))

    gaps = []
    for a, b in runs:
        if b - a < GUTTER_MIN_PX or a == 0 or b == len(loose):
            continue
        if profile[a:b].max() <= GUTTER_INK_FRAC:
            gaps.append((a, b))
            continue
        before = profile[max(0, a - BORDER_SEARCH_PX) : a]
        after = profile[b : b + BORDER_SEARCH_PX]
        if before.size and after.size and before.max() >= BORDER_LINE_FRAC <= after.max():
            gaps.append((a, b))
    return gaps


def _segments(length: int, gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    edges = [0] + [v for gap in gaps for v in gap] + [length]
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] > edges[i]]


def _xy_cut(
    ink: np.ndarray,
    centers: np.ndarray,
    boxes: np.ndarray,
    members: np.ndarray,
    rect: tuple[int, int, int, int],
    rtl: bool,
    leaves: list[np.ndarray],
) -> None:
    x0, y0, x1, y1 = rect
    if len(members) <= 1:
        if len(members):
            leaves.append(members)
        return

    sub = ink[y0:y1, x0:x1]
    for axis in (0, 1):
        length = (y1 - y0) if axis == 0 else (x1 - x0)
        profile = sub.mean(axis=1 - axis)
        blocked = np.zeros(length, dtype=bool)
        lo = 1 if axis == 0 else 0
        for i in members:
            start = max(0, int(boxes[i, lo]) - (y0 if axis == 0 else x0))
            stop = min(length, int(boxes[i, lo + 2]) - (y0 if axis == 0 else x0))
            blocked[start:stop] = True
        parts = _segments(length, _gaps(profile, blocked))
        if len(parts) < 2:
            continue
        if axis == 1 and rtl:
            parts.reverse()
        groups = []
        for a, b in parts:
            offset = y0 if axis == 0 else x0
            coord = centers[members, 1 - axis]
            inside = members[(coord >= offset + a) & (coord < offset + b)]
            if len(inside):
                groups.append(((a, b), inside))
        if len(groups) < 2:
            continue
        for (a, b), inside in groups:
            child = (x0, y0 + a, x1, y0 + b) if axis == 0 else (x0 + a, y0, x0 + b, y1)
            _xy_cut(ink, centers, boxes, inside, child, rtl, leaves)
        return
    leaves.append(members)


def _within_panel_order(boxes: np.ndarray, rtl: bool):
    def compare(i: int, j: int) -> int:
        ay0, ay1 = boxes[i, 1], boxes[i, 3]
        by0, by1 = boxes[j, 1], boxes[j, 3]
        overlap = min(ay1, by1) - max(ay0, by0)
        if overlap > SAME_LINE_OVERLAP * min(ay1 - ay0, by1 - by0):
            if rtl:
                return -1 if boxes[i, 2] > boxes[j, 2] else 1
            return -1 if boxes[i, 0] < boxes[j, 0] else 1
        return -1 if (ay0 + ay1) < (by0 + by1) else 1

    return cmp_to_key(compare)


def reading_order(regions: Sequence[dict], img_cv: np.ndarray, rtl: bool) -> list[dict]:
    if len(regions) <= 1:
        return list(regions)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.ndim == 3 else img_cv
    ink = (gray < INK_LEVEL).astype(np.float32)
    boxes = np.array(
        [[r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]] for r in regions],
        dtype=np.int64,
    )
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    h, w = gray.shape
    leaves: list[np.ndarray] = []
    _xy_cut(ink, centers, boxes, np.arange(len(regions)), (0, 0, w, h), rtl, leaves)

    ordered = []
    key = _within_panel_order(boxes, rtl)
    for leaf in leaves:
        ordered.extend(regions[int(i)] for i in sorted(leaf.tolist(), key=key))
    return ordered
