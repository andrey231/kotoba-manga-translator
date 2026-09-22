import requests

from image_io import encode_image, read_image, to_pil
from settings import settings

OLLAMA_IMAGE_MAX = 1024


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
    timeout: int = 800,
    num_predict: int = 6000,
    temperature: float = 0.1,
    system: str | None = None,
    fmt=None,
    stop: list[str] | None = None,
    num_ctx: int | None = None,
    repeat_penalty: float | None = None,
) -> str:
    global _OLLAMA_CALL_NUM
    import random

    base_opts = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx is not None:
        base_opts["num_ctx"] = num_ctx
    if stop:
        base_opts["stop"] = stop
    if repeat_penalty is not None:
        base_opts["repeat_penalty"] = repeat_penalty

    encoded_image = _prep_ollama_image(image_path) if image_path else None

    def _call(opts: dict, model: str) -> str:
        global _OLLAMA_CALL_NUM
        _OLLAMA_CALL_NUM += 1
        call_num = _OLLAMA_CALL_NUM
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if fmt is not None:
            payload["format"] = fmt
        if opts:
            payload["options"] = opts
        if image_path:
            payload["images"] = [encoded_image]
        r = requests.post(settings().ollama_url, json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise ValueError(f"Ollama request failed: {body['error']}")
        response = body.get("response", "")
        if not isinstance(response, str):
            raise ValueError("Ollama response must be text")
        response = response.strip()
        if settings().llm_debug:
            _log_llm_call(call_num, model, prompt, response, opts, bool(image_path))
        return response

    response = _call(dict(base_opts), model_name)
    if response:
        return response

    response = _call(dict(base_opts, seed=random.randint(1, 100000)), model_name)
    if response:
        return response

    return ""
