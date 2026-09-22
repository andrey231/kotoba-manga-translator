import math


def public_bubble(bubble: dict, idx: int | None = None) -> dict:
    result = {key: value for key, value in bubble.items() if not key.startswith("_")}
    if idx is not None:
        result["idx"] = idx
    return result


def normalize_bubble(bubble: dict) -> dict:
    aliases = {
        "w": "width",
        "h": "height",
        "_text_color": "text_color",
        "_text_angle": "text_angle",
    }
    for old, new in aliases.items():
        if old in bubble:
            bubble.setdefault(new, bubble.pop(old))
    original_width = bubble.pop("orig_w", bubble.get("width"))
    original_height = bubble.pop("orig_h", bubble.get("height"))
    if original_width != bubble.get("width") or original_height != bubble.get("height"):
        bubble.setdefault("source_box", [bubble["x"], bubble["y"], original_width, original_height])
    return bubble


def source_box(bubble: dict) -> tuple[int, int, int, int]:
    return tuple(
        bubble.get("source_box", (bubble["x"], bubble["y"], bubble["width"], bubble["height"]))
    )


def _finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Bubble dimensions and styles must be finite numbers")
    return result


def _color(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.removeprefix("#")
        if len(value) != 6:
            raise ValueError("Expected a six-digit RGB color")
        return [int(value[i : i + 2], 16) for i in (0, 2, 4)]
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(type(c) is int and 0 <= c <= 255 for c in value)
    ):
        return list(value)
    raise ValueError("Invalid RGB color")


def update_bubble(bubble: dict, update: dict) -> None:
    for name in ("translation", "font_path"):
        if name in update:
            value = update[name]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be text")
            bubble[name] = value or ("" if name == "translation" else None)
    if "font_size" in update:
        value = update["font_size"]
        bubble["font_size"] = max(2, min(512, int(_finite(value)))) if value else None
    for name in ("text_color", "outline_color"):
        if name in update:
            bubble[name] = _color(update[name])
    if "outline_width" in update:
        bubble["outline_width"] = max(0, min(32, int(_finite(update["outline_width"] or 0))))
    for name in ("bold", "italic", "underline"):
        if name in update:
            if not isinstance(update[name], bool):
                raise ValueError(f"{name} must be a boolean")
            bubble[name] = update[name]
    if "text_align" in update:
        if update["text_align"] not in ("left", "center", "right"):
            raise ValueError("Invalid text alignment")
        bubble["text_align"] = update["text_align"]
    if "text_angle" in update:
        bubble["text_angle"] = (_finite(update["text_angle"] or 0) + 180) % 360 - 180
    if any(name in update for name in ("box_cx", "box_cy", "box_sw", "box_sh")):
        original_box = source_box(bubble)
        base_width, base_height = original_box[2:]
        cx = _finite(update.get("box_cx", bubble["x"] + bubble["width"] / 2))
        cy = _finite(update.get("box_cy", bubble["y"] + bubble["height"] / 2))
        sw = max(0.05, min(100, _finite(update.get("box_sw", bubble["width"] / base_width))))
        sh = max(0.05, min(100, _finite(update.get("box_sh", bubble["height"] / base_height))))
        width = max(4, min(16384, round(base_width * sw)))
        height = max(4, min(16384, round(base_height * sh)))
        bubble.update(
            x=max(0, round(cx - width / 2)),
            y=max(0, round(cy - height / 2)),
            width=width,
            height=height,
        )
        if (bubble["x"], bubble["y"], width, height) != original_box:
            bubble["source_box"] = list(original_box)
        else:
            bubble.pop("source_box", None)
        bubble.pop("_text_mask", None)
