import os
from pathlib import Path

from PIL import ImageFont


def system_font_dirs() -> list[str]:
    dirs: list[str] = []
    if os.name == "nt":
        dirs.append("C:/Windows/Fonts")
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    else:
        dirs += [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
        dirs += [
            "/Library/Fonts",
            "/System/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    dirs.append(str(Path.cwd()))
    return [d for d in dirs if os.path.isdir(d)]


def resolve_font(requested: str) -> str | None:
    try:
        ImageFont.truetype(requested, 12)
        return requested
    except OSError:
        pass
    candidates = [
        "Ace 2.0 BB Cyr.ttf",
        "animeace2_bld.ttf",
        "animeace2_reg.ttf",
        "CCWildWords.ttf",
        "wildwords.ttf",
        "arialbd.ttf",
        "ARIALBD.TTF",
        "calibrib.ttf",
        "verdanab.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]
    sys_dirs = system_font_dirs()
    bold_kw = ("bold", "bd", "black", "heavy")
    for d in sys_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for name in files:
                if any(kw in name.lower() for kw in bold_kw):
                    candidates.append(os.path.join(root, name))
            break

    for cand in candidates:
        try:
            ImageFont.truetype(cand, 12)
            return cand
        except OSError:
            continue
    return None
