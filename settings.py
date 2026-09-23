import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit, urlunsplit


def ollama_endpoint(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    parsed = urlsplit(value)
    if not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("Invalid Ollama URL")
    path = parsed.path.removesuffix("/api/generate").rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/api/generate", "", ""))


@dataclass(frozen=True)
class Settings:
    llm_model: str = ""
    ollama_url: str = field(
        default_factory=lambda: ollama_endpoint(
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        )
    )
    font_path: str | None = "arial.ttf"
    detect_threshold: float = 0.5
    sfx_threshold: float = 0.3
    min_bubble_area: int = 400
    max_font_size: int = 90
    inpaint_shrink: int = 1
    chunk_size: int = 5
    translate_retries: int = 3
    llm_debug: bool = False
    crops_dir: str | None = None
    mask_debug_dir: str | None = None
    glossary: tuple = ()

    def __post_init__(self):
        if not 0 <= self.detect_threshold <= 1 or not 0 <= self.sfx_threshold <= 1:
            raise ValueError("Detection thresholds must be between 0 and 1")
        for name in ("min_bubble_area", "max_font_size", "chunk_size", "translate_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.inpaint_shrink) is not int or self.inpaint_shrink < 0:
            raise ValueError("inpaint_shrink must be a non-negative integer")


_SETTINGS = ContextVar("translation_settings", default=Settings())


def settings() -> Settings:
    return _SETTINGS.get()


@contextmanager
def use_settings(config: Settings):
    token = _SETTINGS.set(config)
    try:
        yield config
    finally:
        _SETTINGS.reset(token)


def updated_settings(**values) -> Settings:
    if values.get("ollama_url"):
        values["ollama_url"] = ollama_endpoint(values["ollama_url"])
    return replace(settings(), **{key: value for key, value in values.items() if value is not None})
