import argparse
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests

import manga_translator
import models
from context import CharacterArchive, ErrorLog, MangaContext
from fonts import resolve_font
from settings import Settings, use_settings


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _run(arguments):
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "image": str(Path(arguments.image).resolve()),
        "model": arguments.model,
        "stages": [],
        "requests": [],
        "gpu_before": _gpu(),
        "overrides": {
            "release_gpu": arguments.release_gpu,
            "num_ctx": arguments.num_ctx,
            "no_thinking": arguments.no_thinking,
        },
    }
    original_post = requests.post
    current_stage = None
    stage_started = started

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        _write(output / "timings.json", report)

    def on_stage(page_idx, key):
        nonlocal current_stage, stage_started
        now = time.perf_counter()
        if current_stage is not None:
            report["stages"].append({"stage": current_stage, "seconds": now - stage_started})
        current_stage, stage_started = key, now
        report["active_stage"] = key
        save()
        logging.getLogger(__name__).debug("Stage started: %s", key)

    def timed_post(url, **kwargs):
        request_started = time.perf_counter()
        payload = kwargs.get("json", {})
        if url.endswith("/api/generate"):
            if arguments.release_gpu:
                models.release_gpu_memory()
            if arguments.num_ctx is not None:
                payload.setdefault("options", {})["num_ctx"] = arguments.num_ctx
            if arguments.no_thinking:
                payload["think"] = False
        record = {
            "stage": current_stage,
            "url": url,
            "timeout": kwargs.get("timeout"),
            "payload": {k: v for k, v in payload.items() if k != "images"},
            "image_count": len(payload.get("images", [])),
            "gpu": _gpu(),
        }
        report["requests"].append(record)
        save()
        logging.getLogger(__name__).debug("Request started: %s", record)
        try:
            response = original_post(url, **kwargs)
            record["status"] = response.status_code
            if not kwargs.get("stream"):
                body = response.json()
                _write(output / f"response_{len(report['requests']):02d}.json", body)
                record["metrics"] = {
                    k: v
                    for k, v in body.items()
                    if k.endswith("duration") or k.endswith("count") or k == "done_reason"
                }
                record["response_chars"] = len(body.get("response", ""))
                record["thinking_chars"] = len(body.get("thinking", ""))
            return response
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["seconds"] = time.perf_counter() - request_started
            save()
            logging.getLogger(__name__).debug("Request finished: %.3fs", record["seconds"])

    errors = ErrorLog(str(output / "errors.log"))
    config = Settings(
        llm_model=arguments.model, llm_debug=arguments.verbose, font_path=resolve_font("arial.ttf")
    )
    with (output / "pipeline.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            logging.basicConfig(
                level=logging.DEBUG if arguments.verbose else logging.WARNING,
                stream=log,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                force=True,
            )
            try:
                with use_settings(config), patch.object(requests, "post", side_effect=timed_post):
                    bubbles = manga_translator.process_page(
                        image_path=arguments.image,
                        page_idx=1,
                        manga_ctx=MangaContext(),
                        archive=CharacterArchive(str(output / "characters.json")),
                        output_path=str(output / "translated.png"),
                        errors=errors,
                        on_stage=on_stage,
                    )
                _write(
                    output / "bubbles.json",
                    [{k: v for k, v in b.items() if not k.startswith("_")} for b in bubbles],
                )
                report["status"] = "completed"
            except Exception as error:
                report["status"] = "failed"
                report["error"] = f"{type(error).__name__}: {error}"
                logging.exception("Page profiling failed")
            finally:
                if current_stage is not None:
                    report["stages"].append(
                        {"stage": current_stage, "seconds": time.perf_counter() - stage_started}
                    )
                report["active_stage"] = None
                report["gpu_after"] = _gpu()
                errors.save()
                save()
    if arguments.verbose:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    return report["status"] != "completed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a complete manga page pipeline")
    parser.add_argument("image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--release-gpu", action="store_true")
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--no-thinking", action="store_true")
    sys.exit(_run(parser.parse_args()))
