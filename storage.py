import json
from pathlib import Path


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
