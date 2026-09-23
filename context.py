import json
import logging
import os
import re
import time
from collections import deque

from storage import write_json

logger = logging.getLogger(__name__)


class CharacterArchive:
    def __init__(self, path: str = "characters.json"):
        self.path = path
        self.characters: dict = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.characters = json.load(f)
            logger.debug(f"  Loaded archive: {len(self.characters)} character(s)")

    def save(self):
        write_json(self.path, self.characters)

    def to_prompt(self) -> str:
        if not self.characters:
            return "CHARACTER ARCHIVE: (empty, this may be the first page)"
        lines = ["CHARACTER ARCHIVE (these characters have been seen before):"]
        for cid, c in self.characters.items():
            lines.append(
                f"- ID={cid} | {c['name']} | gender={c['gender']} | "
                f"appearance: {c['appearance']} | notes: {c.get('notes', '')} | "
                f"first seen: page {c['first_seen']}"
            )
        return "\n".join(lines)

    def _unique_name(self, proposed: str, cid: str) -> str:
        existing_names = {c["name"].lower() for c in self.characters.values()}
        if proposed.lower() not in existing_names:
            return proposed

        base_words = set(re.findall(r"[a-z]+", proposed.lower()))
        id_words = re.findall(r"[a-z]+", cid.lower())
        extras = [w for w in id_words if w not in base_words and len(w) > 2]
        if extras:
            candidate = f"{proposed} ({' '.join(extras)})"
            if candidate.lower() not in existing_names:
                return candidate

        for i in range(2, 100):
            candidate = f"{proposed} #{i}"
            if candidate.lower() not in existing_names:
                return candidate
        return proposed

    def update_from_json(self, data: list, page_idx: int):
        for char in data:
            cid = char.get("id", "").strip()
            if not cid:
                continue
            if cid in self.characters:
                new_notes = char.get("notes", "")
                if new_notes and new_notes not in self.characters[cid].get("notes", ""):
                    self.characters[cid]["notes"] = (
                        self.characters[cid].get("notes", "") + "; " + new_notes
                    ).strip("; ")
            else:
                raw_name = char.get("name", cid)
                unique_name = self._unique_name(raw_name, cid)
                if unique_name != raw_name:
                    logger.debug(f"  [archive] Name '{raw_name}' already taken → '{unique_name}'")

                self.characters[cid] = {
                    "name": unique_name,
                    "gender": char.get("gender", "unknown"),
                    "appearance": char.get("appearance", ""),
                    "notes": char.get("notes", ""),
                    "first_seen": page_idx,
                }
                logger.debug(f"  [archive] New character: {unique_name}")
        self.save()

    def find_character(self, description: str) -> dict | None:
        if not description:
            return None
        desc_lower = description.lower()
        for cid, c in self.characters.items():
            if c["name"].lower() in desc_lower or cid.lower() in desc_lower:
                return c
        return None


class MangaContext:
    def __init__(self):
        self.page_summaries: deque[tuple[int, str]] = deque(maxlen=5)

    def update(self, summary: str, page_idx: int | None = None):
        if not summary or not summary.strip():
            return
        if page_idx is None:
            page_idx = self.page_summaries[-1][0] + 1 if self.page_summaries else 1
        self.page_summaries.append((page_idx, summary))

    def to_prompt(self) -> str:
        if not self.page_summaries:
            return ""
        lines = "\n".join(f"Page {page}: {summary}" for page, summary in self.page_summaries)
        return f"STORY SO FAR:\n{lines}"


class ErrorLog:
    def __init__(self, path: str = "errors.log"):
        self.path = path
        self.entries: list[dict] = []
        if os.path.exists(self.path):
            os.remove(self.path)

    def add(self, page: int, kind: str, message: str, **details):
        entry = {
            "page": page,
            "kind": kind,
            "message": message,
            **details,
        }
        self.entries.append(entry)

    def summary(self) -> dict:
        counts: dict = {}
        for e in self.entries:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        return counts

    def save(self):
        if not self.entries:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"# Error log — {len(self.entries)} entries total\n")
            f.write(f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            by_page: dict = {}
            for e in self.entries:
                by_page.setdefault(e["page"], []).append(e)

            for page in sorted(by_page):
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Page {page}\n")
                f.write(f"{'=' * 60}\n")
                for e in by_page[page]:
                    f.write(f"\n  [{e['kind']}] {e['message']}\n")
                    extras = {k: v for k, v in e.items() if k not in ("page", "kind", "message")}
                    if extras:
                        for k, v in extras.items():
                            v_str = str(v)
                            if len(v_str) > 300:
                                v_str = v_str[:300] + "... [truncated]"
                            f.write(f"    {k}: {v_str}\n")
