import logging
import re

from context import CharacterArchive, MangaContext
from image_io import read_image
from llm import ollama
from settings import settings
from text_utils import parse_json_array

logger = logging.getLogger(__name__)


def detect_character_intro_page(image_path: str) -> bool:
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

    raw = ollama(settings().llm_model, prompt, image_path, timeout=120).strip().upper()
    is_intro = raw.startswith("YES")
    logger.debug(f"  [intro detect] {raw[:30]} → {'gallery page' if is_intro else 'regular page'}")
    return is_intro


def extract_character_intros(image_path: str, archive: CharacterArchive, page_idx: int) -> str:
    logger.debug("  Extracting character introductions...")

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

    raw = ollama(settings().llm_model, prompt, image_path)
    chars = parse_json_array(raw)

    if not chars:
        logger.debug(f"  [warn] character intros did not parse: {raw[:200]}")
        return "CHARACTERS ON THIS PAGE: unknown"

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
            f"- {marker} {c.get('name', '?')} | {c.get('gender', '?')} | {c.get('notes', '')}"
        )
    return "\n".join(lines)


def analyze_page_full(
    image_path: str, archive: CharacterArchive, manga_ctx: MangaContext, page_idx: int
) -> tuple[str, str, str]:
    logger.debug("  Analyzing page (characters + scene)...")

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

    raw = ollama(settings().llm_model, prompt, image_path)

    chars_match = re.search(
        r"===\s*CHARACTERS\s*===(.*?)===\s*SCENE\s*===",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    chars_section = chars_match.group(1) if chars_match else raw
    chars = parse_json_array(chars_section)

    if not chars:
        logger.debug(f"  [warn] characters did not parse: {chars_section[:200]}")
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

    scene_match = re.search(
        r"===\s*SCENE\s*===(.*?)(?:===\s*END\s*===|\Z)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    scene_section = scene_match.group(1) if scene_match else raw

    context_match = re.search(
        r"CONTEXT:\s*(.+?)\s*(?:SUMMARY:|===|\Z)",
        scene_section,
        re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r"SUMMARY:\s*(.+?)\s*(?:===|\Z)",
        scene_section,
        re.DOTALL | re.IGNORECASE,
    )

    page_context = context_match.group(1).strip() if context_match else scene_section[:300].strip()
    page_summary = summary_match.group(1).strip() if summary_match else page_context[:120]

    page_context = page_context[:800]
    page_summary = page_summary.split("\n")[0][:200]

    return characters_context, page_context, page_summary


def deduplicate_characters(chars: list, archive: CharacterArchive) -> list:
    result = []
    for char in chars:
        if not isinstance(char, dict):
            continue
        cid = char.get("id", "")
        if not isinstance(cid, str) or not cid.strip():
            continue
        if cid in archive.characters:
            known = archive.characters[cid]
            char.update(name=known["name"], gender=known.get("gender", "unknown"))
            char["is_new"] = False
            result.append(char)
            continue

        if char.get("is_new", True):
            match = find_similar_in_archive(char.get("appearance", "").lower(), archive)
            if match:
                existing_id, existing = match
                logger.debug(
                    f"  [dedup] '{char.get('name')}' → '{existing['name']}' (id={existing_id})"
                )
                char.update(
                    {
                        "id": existing_id,
                        "name": existing["name"],
                        "gender": existing["gender"],
                        "is_new": False,
                    }
                )
        result.append(char)
    return result


def find_similar_in_archive(appearance: str, archive: CharacterArchive) -> tuple | None:
    keywords = extract_appearance_keywords(appearance)
    if not keywords:
        return None

    new_distinct = extract_distinctive_features(appearance)

    best_match, best_score = None, 0
    for cid, c in archive.characters.items():
        archived_appearance = c.get("appearance", "").lower()
        archived_keywords = extract_appearance_keywords(archived_appearance)
        common = keywords & archived_keywords

        min_size = min(len(keywords), len(archived_keywords))
        threshold = max(3, min_size // 2)

        if len(common) < threshold:
            continue

        archived_distinct = extract_distinctive_features(archived_appearance)
        if features_conflict(new_distinct, archived_distinct):
            logger.debug(
                f"  [dedup-skip] '{c['name']}' rejected: "
                f"волосы {new_distinct['hair_colors']} vs {archived_distinct['hair_colors']}"
            )
            continue

        if len(common) > best_score:
            best_score = len(common)
            best_match = (cid, c)

    return best_match


def extract_appearance_keywords(text: str) -> set:
    stopwords = {
        "a",
        "an",
        "the",
        "with",
        "and",
        "or",
        "is",
        "are",
        "has",
        "have",
        "wearing",
        "looking",
        "man",
        "woman",
        "person",
        "character",
        "young",
        "old",
        "tall",
        "short",
        "small",
        "large",
        "none",
        "partially",
        "covered",
        "hair",
        "eyes",
        "face",
        "expression",
        "style",
        "kimono",
        "shirt",
        "pants",
        "dress",
        "clothes",
        "clothing",
        "patterned",
    }
    return {w for w in re.findall(r"\b[a-z]+\b", text) if w not in stopwords and len(w) > 2}


HAIR_COLORS = {
    "dark",
    "black",
    "brown",
    "blonde",
    "blond",
    "yellow",
    "light",
    "white",
    "silver",
    "gray",
    "grey",
    "red",
    "ginger",
    "auburn",
    "pink",
    "blue",
    "green",
    "purple",
    "orange",
}
HAIR_STYLES = {
    "long",
    "short",
    "bob",
    "ponytail",
    "twintails",
    "braid",
    "braids",
    "curly",
    "straight",
    "wavy",
    "spiky",
    "bangs",
    "fringe",
}


def extract_distinctive_features(text: str) -> dict:
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    return {
        "hair_colors": words & HAIR_COLORS,
        "hair_styles": words & HAIR_STYLES,
    }


def features_conflict(a: dict, b: dict) -> bool:
    if a["hair_colors"] and b["hair_colors"]:
        if not (a["hair_colors"] & b["hair_colors"]):
            return True
    return False


def attribute_bubbles(
    image_path: str,
    bubbles: list[dict],
    page_context: str,
    characters_context: str,
    archive: CharacterArchive,
) -> list[dict]:
    logger.debug("  Attributing speech bubbles...")
    bubble_list = "\n".join(
        f"[{i + 1}] pos=({b['x']},{b['y']}) size={b['width']}x{b['height']} "
        f'text="{b.get("text", "").replace(chr(10), " ")}"'
        for i, b in enumerate(bubbles)
    )

    page_height, page_width = read_image(image_path).shape[:2]
    prompt = f"""You are analyzing a manga page.
Bubble coordinates refer to the original {page_width}x{page_height} page, before image resizing.

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

    raw = ollama(settings().llm_model, prompt, image_path)
    for attr in parse_json_array(raw):
        if not isinstance(attr, dict) or type(attr.get("bubble")) is not int:
            continue
        idx = attr["bubble"] - 1
        if 0 <= idx < len(bubbles):
            speaker = attr.get("speaker", "unknown")
            gender = attr.get("gender", "unknown")
            known = archive.find_character(speaker)
            if known:
                speaker = known["name"]
                gender = known["gender"]
            bubbles[idx]["speaker"] = speaker
            bubbles[idx]["gender"] = gender

    return bubbles
