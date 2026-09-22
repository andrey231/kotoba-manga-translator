import json
import re


def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`+", "", text)

    def _circled(m):
        n = int(m.group(1))
        if 1 <= n <= 20:
            return chr(0x2460 + n - 1)
        return m.group(0)

    text = re.sub(r"\$\s*\\?textcircled\{(\d+)\}\s*\$", _circled, text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_emotion_tag(text: str) -> str:
    t = text.replace("！", "!").replace("？", "?").replace("…", "...")

    has_double_excl = bool(re.search(r"!{2,}", t))
    has_excl = "!" in t
    has_question = "?" in t
    has_mixed = bool(re.search(r"[?!][!?]", t))
    has_ellipsis = "..." in t or "…" in text
    has_repetition = bool(re.search(r"(.{2,4})\1", text))
    all_caps = text.replace(" ", "").replace("\n", "").isupper() and len(text.strip()) > 2

    if has_double_excl or (has_excl and has_repetition):
        return "excited/shouting"
    if has_mixed:
        return "shocked/alarmed"
    if has_ellipsis and not has_excl:
        return "hesitant/trailing-off"
    if has_excl:
        return "emphatic"
    if has_question:
        return "questioning"
    if all_caps:
        return "loud/sfx"
    return "neutral"


def parse_json_array(text: str) -> list:
    if not text:
        return []

    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

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
        candidate = cleaned[start:]
    else:
        candidate = cleaned[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r",\s*([\]\}])", r"\1", candidate)
    last_close = repaired.rfind("}")
    if last_close > 0:
        repaired = repaired[: last_close + 1] + "]"
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return []


def natural_key(filename: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", filename)]
