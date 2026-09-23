import time
from contextlib import contextmanager
from contextvars import ContextVar

import requests

from image_io import encode_image, read_image, to_pil
from models import release_gpu_memory
from settings import settings

OLLAMA_IMAGE_MAX = 1024
_PHASE_ACTIVE = ContextVar("ollama_phase_active", default=False)


def unload_model(model_name: str) -> None:
    if not model_name:
        return
    response = requests.post(
        settings().ollama_url,
        json={"model": model_name, "keep_alive": 0, "stream": False},
        timeout=(10, 30),
    )
    try:
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise ValueError(f"Ollama model unload failed: {body['error']}")
    finally:
        response.close()


@contextmanager
def model_phase():
    if _PHASE_ACTIVE.get():
        yield
        return
    release_gpu_memory()
    token = _PHASE_ACTIVE.set(True)
    try:
        yield
    finally:
        _PHASE_ACTIVE.reset(token)
        unload_model(settings().llm_model)


def _prep_ollama_image(path: str) -> str:
    return encode_image(to_pil(read_image(path)), max_side=OLLAMA_IMAGE_MAX)


_OLLAMA_CALL_NUM = 0


def _log_llm_call(
    call_num: int, model: str, prompt: str, response: str, opts: dict, has_image: bool
) -> None:
    sep = "─" * 76
    print(f"\n{sep}")
    print(
        f"[LLM call #{call_num}] model={model} options={opts} image={'yes' if has_image else 'no'}"
    )
    print(f"{sep}")
    print(f"PROMPT ({len(prompt)} chars):")
    print(prompt)
    print(f"{sep}")
    print(f"RESPONSE ({len(response)} chars):")
    if response:
        print(response)
    else:
        print("(empty response)")
    print(f"{sep}\n")


def ollama(
    model_name: str,
    prompt: str,
    image_path: str = None,
    timeout: int = 120,
    num_predict: int = 6000,
    temperature: float = 0.1,
    system: str | None = None,
    fmt=None,
    stop: list[str] | None = None,
    num_ctx: int = 8192,
    repeat_penalty: float | None = None,
    think: bool | str = False,
) -> str:
    global _OLLAMA_CALL_NUM

    base_opts = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx is not None:
        base_opts["num_ctx"] = num_ctx
    if stop:
        base_opts["stop"] = stop
    if repeat_penalty is not None:
        base_opts["repeat_penalty"] = repeat_penalty

    encoded_image = _prep_ollama_image(image_path) if image_path else None

    if not _PHASE_ACTIVE.get():
        release_gpu_memory()
    _OLLAMA_CALL_NUM += 1
    call_num = _OLLAMA_CALL_NUM
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": base_opts,
    }
    if _PHASE_ACTIVE.get():
        payload["keep_alive"] = -1
    if system:
        payload["system"] = system
    if fmt is not None:
        payload["format"] = fmt
    if image_path:
        payload["images"] = [encoded_image]
    started = time.perf_counter()
    if settings().llm_debug:
        print(f"[LLM call #{call_num}] started model={model_name} timeout={timeout}s")
    r = requests.post(settings().ollama_url, json=payload, timeout=(10, timeout))
    try:
        r.raise_for_status()
        body = r.json()
    finally:
        r.close()
    if body.get("error"):
        raise ValueError(f"Ollama request failed: {body['error']}")
    response = body.get("response", "")
    if not isinstance(response, str):
        raise ValueError("Ollama response must be text")
    response = response.strip()
    if settings().llm_debug:
        _log_llm_call(call_num, model_name, prompt, response, base_opts, bool(image_path))
        metrics = {
            key: body.get(key)
            for key in (
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "eval_count",
                "done_reason",
            )
        }
        print(
            f"[LLM call #{call_num}] elapsed={time.perf_counter() - started:.3f}s metrics={metrics}"
        )
    if body.get("done_reason") == "length":
        raise ValueError("Ollama output reached the token limit; the response is incomplete")
    if not response:
        raise ValueError(
            "Ollama returned no final response "
            f"(thinking_chars={len(body.get('thinking', ''))}, done_reason={body.get('done_reason')})"
        )
    return response
