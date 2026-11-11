from collections.abc import Sequence

SFX_KIND = "sfx"
MAX_SFX_LETTERS = 6
MIN_KANA_SHARE = 0.8

_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_KATAKANA_SHIFT = 0x60
_KANA_RANGES = ((0x3041, 0x3096), (0x30A1, 0x30FA))
_PROLONG = "ー〜～~"
_SMALL_TSU = "っッ"
_DROPPED = _PROLONG + _SMALL_TSU + "!?！？….。、・♥♡ "

_MEANINGS = {
    "ヌ": "looming up: something silently rises or appears and towers over",
    "ゴ": "ominous rumble, menacing pressure",
    "ド": "dramatic heavy impact or emphasis",
    "ドン": "boom, heavy impact, dramatic emphasis",
    "ドーン": "boom, big dramatic impact",
    "バーン": "bang, grand dramatic reveal",
    "ジャーン": "ta-da, triumphant reveal",
    "ザ": "quick step or swish of movement",
    "ザワ": "uneasy murmur, unsettled stir",
    "ガヤ": "crowd noise, chatter",
    "ワイ": "lively chatter",
    "バ": "sudden swift movement",
    "バン": "bang, slam",
    "ガ": "grab or sharp hit",
    "ガシ": "firm grab",
    "ガタ": "rattle, clatter, sudden jolt",
    "ガチャ": "click of a door or mechanism",
    "カチ": "click",
    "ピ": "beep",
    "ピタ": "abrupt stop",
    "ピク": "twitch",
    "ビク": "startled flinch",
    "ギク": "guilty start, being caught off guard",
    "ドキ": "heartbeat, nervousness",
    "シン": "dead silence",
    "ジ": "intent stare",
    "チラ": "quick glance",
    "ニコ": "smile",
    "ニヤ": "smirk",
    "ムカ": "flash of irritation",
    "イラ": "annoyance",
    "ハ": "sudden realization or gasp",
    "ヒュ": "whoosh of wind",
    "ビュン": "whiz past",
    "シュ": "swish",
    "ス": "smooth quiet movement, quietly reaching or appearing",
    "ズ": "slide, sinking, or a heavy push",
    "ズブ": "stab, sinking in",
    "ズン": "heavy thud, gloom settling",
    "ズザ": "skidding slide",
    "ドサ": "thud of a fall",
    "バタ": "collapse, flop",
    "パチ": "snap, clap, or blink",
    "パン": "pop or slap",
    "バキ": "crack, hard punch",
    "ボキ": "snap of breaking",
    "ゴン": "bonk on the head",
    "ガン": "clang, hard hit",
    "ドカ": "crash, kick",
    "ドゴ": "heavy blow",
    "グ": "clench, push hard",
    "ギュ": "tight squeeze or hug",
    "ギロ": "glare",
    "キ": "sharp glare or shrill frustrated cry",
    "プイ": "turning away in a huff",
    "プ": "stifled laugh, pfft",
    "フ": "scoff, short exhale",
    "ハァ": "sigh or panting",
    "ゼェ": "heavy panting",
    "ゴク": "gulp",
    "ペコ": "bow",
    "ポン": "pat or pop",
    "ポタ": "drip",
    "ザー": "pouring rain",
    "ゴロ": "rolling or thunder",
    "ユラ": "swaying, wavering, ominous rise",
    "スタ": "brisk walking",
    "トコ": "small quick steps",
    "ダ": "dash",
    "ダダ": "running or rapid fire",
    "ブ": "spurt, burst, buzz",
    "カ": "flash, flare-up of light or anger",
    "キラ": "sparkle",
    "ピカ": "flash, shine",
    "ガーン": "shock",
    "ズーン": "gloom, crushing disappointment",
    "シュン": "downcast",
}


def _katakana(text: str) -> str:
    return "".join(
        chr(ord(ch) + _KATAKANA_SHIFT) if _HIRAGANA_START <= ord(ch) <= _HIRAGANA_END else ch
        for ch in text
    )


def _is_kana(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _KANA_RANGES) or ch in _PROLONG


def _repeated_unit(text: str) -> str:
    for size in range(1, len(text) // 2 + 1):
        unit = text[:size]
        if len(text) % size == 0 and unit * (len(text) // size) == text:
            return unit
    return text


def _base(text: str) -> str:
    kept = "".join(ch for ch in _katakana(text) if ch not in _DROPPED)
    return _repeated_unit(kept)


def sound_effect_meaning(text: str) -> str | None:
    base = _base(text)
    if not base:
        return None
    full = "".join(ch for ch in _katakana(text) if ch not in "!?！？…. ")
    for key in (full, _repeated_unit(full), base, base[:2], base[:1]):
        if key in _MEANINGS:
            return _MEANINGS[key]
    return None


def is_sound_effect(text: str, in_bubble: bool) -> bool:
    if in_bubble or not text:
        return False
    letters = [ch for ch in text if ch.isalnum() or ch in _PROLONG]
    if not 1 <= len(letters) <= MAX_SFX_LETTERS:
        return False
    if sum(_is_kana(ch) for ch in letters) < MIN_KANA_SHARE * len(letters):
        return False
    katakana = _katakana(text)
    marked = any(ch in _SMALL_TSU or ch in _PROLONG for ch in text)
    repeated = len(_base(text)) < len("".join(ch for ch in katakana if ch not in _DROPPED))
    only_katakana = all(not (_HIRAGANA_START <= ord(ch) <= _HIRAGANA_END) for ch in letters)
    return marked or repeated or only_katakana or sound_effect_meaning(text) is not None


def mark_sound_effects(regions: Sequence[dict]) -> None:
    for region in regions:
        if is_sound_effect(region.get("text", ""), bool(region.get("bubble_box"))):
            region["kind"] = SFX_KIND
        else:
            region.pop("kind", None)
