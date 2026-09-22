import asyncio
import io
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from dataclasses import fields
from functools import wraps
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import manga_translator as mt
from bubbles import normalize_bubble, public_bubble, update_bubble
from context import MangaContext
from fonts import system_font_dirs
from image_io import SUPPORTED_EXTENSIONS, clip_box, read_image, to_pil, write_image
from models import detect_bubbles
from ocr import ocr_region
from rendering import draw_results
from settings import Settings, settings, use_settings
from storage import write_json
from text_utils import _infer_emotion_tag, natural_key
from translation import translate_batch

BASE_DIR = Path("web_data")
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
JOBS_DIR = BASE_DIR / "jobs"
GLOSSARY_FILE = BASE_DIR / "glossary.json"

for d in (UPLOADS_DIR, RESULTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


_MAX_EXTRACT_BYTES = 4 * 1024**3

_JOB_ID_RE = re.compile(r"[0-9a-f]{6,32}")


def _safe_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id")
    return job_id


def _load_glossary() -> list:
    if GLOSSARY_FILE.exists():
        try:
            return json.loads(GLOSSARY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_glossary(entries: list) -> None:
    write_json(GLOSSARY_FILE, entries)


JOBS: dict = {}


MODEL_GATE = asyncio.Lock()


def _serialize_model_work(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        async with MODEL_GATE:
            return await function(*args, **kwargs)

    return wrapped


def _unique_upload_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    index = 2
    while candidate.exists() or any(path.stem == candidate.stem for path in directory.iterdir()):
        candidate = directory / f"{Path(name).stem}_{index}{Path(name).suffix}"
        index += 1
    return candidate


def _load_pages(job_id: str) -> list:
    pages = json.loads((JOBS_DIR / f"{job_id}.json").read_text(encoding="utf-8"))
    for page in pages:
        for bubble in page["bubbles"]:
            normalize_bubble(bubble)
    return pages


def _save_pages(job_id: str, pages: list) -> None:
    for page in pages:
        page["bubbles"] = [public_bubble(bubble) for bubble in page["bubbles"]]
    path = JOBS_DIR / f"{job_id}.json"
    write_json(path, pages)


def _load_config(job_id: str) -> dict:
    config_path = JOBS_DIR / f"{job_id}_config.json"
    return json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}


async def _run_configured(job_id: str, function, *args, **kwargs):
    config = _load_config(job_id)
    allowed = {field.name for field in fields(Settings)}
    values = {key: value for key, value in config.items() if key in allowed and value is not None}
    from settings import ollama_endpoint

    if values.get("ollama_url"):
        values["ollama_url"] = ollama_endpoint(values["ollama_url"])
    values["glossary"] = tuple(_load_glossary())
    result_dir = RESULTS_DIR / job_id
    if config.get("debug"):
        values["crops_dir"] = str(result_dir / "crops")
    if config.get("mask_debug"):
        directory = result_dir / "mask_debug"
        directory.mkdir(exist_ok=True)
        values["mask_debug_dir"] = str(directory)
    with use_settings(Settings(**values)):
        return await asyncio.to_thread(function, *args, **kwargs)


app = FastAPI(title="Kotoba — Manga Translator")

app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "web_ui.html").read_text(encoding="utf-8"))


OLLAMA_HOST = settings().ollama_url.removesuffix("/api/generate")


def _fetch_ollama_models() -> tuple[list, str | None]:
    import requests as _req

    url = f"{OLLAMA_HOST}/api/tags"
    try:
        r = _req.get(url, timeout=5)
    except _req.exceptions.ConnectionError as e:
        msg = (
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is running (try 'ollama list' in terminal). "
            f"Details: {e}"
        )
        print(f"[/api/models] {msg}")
        return [], msg
    except Exception as e:
        msg = f"Unexpected error reaching {url}: {type(e).__name__}: {e}"
        print(f"[/api/models] {msg}")
        return [], msg

    if r.status_code != 200:
        msg = f"Ollama returned HTTP {r.status_code} from {url}: {r.text[:200]}"
        print(f"[/api/models] {msg}")
        return [], msg

    try:
        data = r.json()
    except Exception as e:
        msg = f"Ollama response is not valid JSON: {e}; body: {r.text[:200]}"
        print(f"[/api/models] {msg}")
        return [], msg

    models = data.get("models", [])
    print(f"[/api/models] Ollama at {url}: {len(models)} model(s) installed")
    return models, None


KNOWN_MULTIMODAL = (
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "minicpm",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen-vl",
    "llama3.2-vision",
    "llama4",
    "pixtral",
    "molmo",
    "gemma3",
    "gemma4",
    "phi3.5-vision",
    "phi-3-vision",
    "phi3-vision",
    "phi4-vision",
    "internvl",
    "cogvlm",
    "yi-vl",
)
MULTIMODAL_FAMILIES = {"clip", "mllama", "llava", "gemma3", "gemma4"}
OCR_FAMILIES = {"glmocr"}
OCR_NAME_HINTS = ("glm-ocr", "tesseract", "paddleocr", "easyocr")
NEVER_MULTIMODAL = ("gemma:", "gemma2:", "gemma2-", "phi3:", "phi3-mini", "phi:")


def _is_ocr(name: str, family: str, families: set) -> bool:
    name_lower = name.lower()
    if any(hint in name_lower for hint in OCR_NAME_HINTS):
        return True
    if family.lower() in OCR_FAMILIES:
        return True
    if families & OCR_FAMILIES:
        return True
    return False


def _is_multimodal(name: str, family: str, families: set) -> bool:
    name_lower = name.lower()
    if any(name_lower.startswith(p) for p in NEVER_MULTIMODAL):
        return False
    if any(known in name_lower for known in KNOWN_MULTIMODAL):
        return True
    if "gemma-4" in name_lower or "gemma-3" in name_lower:
        return True
    if family.lower() in MULTIMODAL_FAMILIES:
        return True
    if families & MULTIMODAL_FAMILIES:
        return True
    return False


@app.get("/api/models")
async def list_models():
    models_raw, error = _fetch_ollama_models()
    out = []
    for m in models_raw:
        name = m.get("name", "")
        details = m.get("details") or {}
        families = set(f.lower() for f in (details.get("families") or []))
        family = details.get("family") or ""
        size_bytes = m.get("size", 0)

        ocr = _is_ocr(name, family, families)
        is_multi = _is_multimodal(name, family, families) if not ocr else False

        out.append(
            {
                "name": name,
                "size_gb": round(size_bytes / (1024**3), 1),
                "family": family,
                "families": sorted(families),
                "multimodal": is_multi,
                "ocr": ocr,
            }
        )
    out.sort(key=lambda m: (m["ocr"], not m["multimodal"], m["name"]))
    return {
        "models": out,
        "default": settings().llm_model,
        "ollama_host": OLLAMA_HOST,
        "error": error,
    }


@app.get("/api/fonts")
async def list_fonts():
    from PIL import ImageFont

    fonts: list[dict] = []
    seen: set[str] = set()
    for root_dir in system_font_dirs():
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.lower().endswith((".ttf", ".otf")):
                    continue
                path = os.path.join(dirpath, fname)
                if path in seen:
                    continue
                seen.add(path)
                try:
                    pil_font = ImageFont.truetype(path, 12)
                    family, style = pil_font.getname()
                    label = f"{family} {style}" if style not in ("Regular", "") else family
                    fonts.append(
                        {
                            "file": fname,
                            "path": path,
                            "family": family,
                            "style": style,
                            "display": label,
                        }
                    )
                except Exception:
                    pass

    fonts.sort(key=lambda f: f["display"].lower())
    return {"fonts": fonts}


@app.get("/api/models/debug")
async def debug_models():
    import requests as _req

    url = f"{OLLAMA_HOST}/api/tags"
    try:
        r = _req.get(url, timeout=5)
        return {
            "ollama_host": OLLAMA_HOST,
            "url_queried": url,
            "http_status": r.status_code,
            "raw_response": r.json() if r.status_code == 200 else r.text,
        }
    except Exception as e:
        return {
            "ollama_host": OLLAMA_HOST,
            "url_queried": url,
            "error": f"{type(e).__name__}: {e}",
        }


@app.post("/api/upload")
async def upload_chapter(
    files: list[UploadFile] = File(...),
    target_lang: str = Form("Russian"),
    font_path: str = Form("arial.ttf"),
    debug: bool = Form(False),
    llm_model: str = Form(""),
    llm_debug: bool = Form(False),
    fast_mode: bool = Form(False),
    mask_debug: bool = Form(False),
    ollama_url: str = Form(""),
    detect_threshold: float = Form(0.5),
    sfx_threshold: float = Form(0.3),
    min_bubble_area: int = Form(400),
    max_font_size: int = Form(90),
    inpaint_shrink: int = Form(1),
    chunk_size: int = Form(5),
    translate_retries: int = Form(3),
):
    try:
        Settings(
            detect_threshold=detect_threshold,
            sfx_threshold=sfx_threshold,
            min_bubble_area=min_bubble_area,
            max_font_size=max_font_size,
            inpaint_shrink=inpaint_shrink,
            chunk_size=chunk_size,
            translate_retries=translate_retries,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    job_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    extracted_total = 0
    for f in files:
        upload_name = (f.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        ext = Path(upload_name).suffix.lower()
        if ext in {".zip", ".cbz"}:
            data = await f.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for entry in sorted(zf.infolist(), key=lambda e: e.filename):
                    if entry.is_dir():
                        continue
                    name = entry.filename.replace("\\", "/").rsplit("/", 1)[-1]
                    if name.startswith("._") or entry.filename.startswith("__MACOSX"):
                        continue
                    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    extracted_total += entry.file_size
                    if extracted_total > _MAX_EXTRACT_BYTES:
                        raise HTTPException(
                            413,
                            "Archive too large when decompressed "
                            f"(> {_MAX_EXTRACT_BYTES // (1024**3)} GB)",
                        )
                    _unique_upload_path(upload_dir, name).write_bytes(zf.read(entry))
        elif ext in SUPPORTED_EXTENSIONS:
            target = _unique_upload_path(upload_dir, upload_name)
            with open(target, "wb") as out:
                shutil.copyfileobj(f.file, out)

    saved = sorted(upload_dir.iterdir(), key=lambda p: natural_key(p.name))
    if not saved:
        raise HTTPException(400, "Не загружено ни одной валидной картинки")

    JOBS[job_id] = {
        "status": "ready",
        "config": {
            "target_lang": target_lang,
            "font_path": font_path,
            "debug": debug,
            "llm_model": llm_model.strip() or None,
            "llm_debug": llm_debug,
            "fast_mode": fast_mode,
            "mask_debug": mask_debug,
            "ollama_url": ollama_url.strip() or None,
            "detect_threshold": detect_threshold,
            "sfx_threshold": sfx_threshold,
            "min_bubble_area": min_bubble_area,
            "max_font_size": max_font_size,
            "inpaint_shrink": inpaint_shrink,
            "chunk_size": chunk_size,
            "translate_retries": translate_retries,
        },
        "stats": None,
        "queue": asyncio.Queue(),
        "task": None,
        "cancel_event": threading.Event(),
    }
    write_json(JOBS_DIR / f"{job_id}_config.json", JOBS[job_id].pop("config"))
    return {"job_id": job_id, "total_pages": len(saved), "filenames": [p.name for p in saved]}


@app.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in JOBS:
        await websocket.send_json({"type": "error", "message": "Unknown job_id"})
        await websocket.close()
        return

    job = JOBS[job_id]
    loop = asyncio.get_running_loop()

    def emit(payload: dict):
        loop.call_soon_threadsafe(job["queue"].put_nowait, payload)

    def on_start(total):
        job["status"] = "running"
        emit({"type": "start", "total": total})

    def on_page_done(page_idx, total, filename, output_path, bubbles, elapsed):
        try:
            rel = Path(output_path).resolve().relative_to(BASE_DIR.resolve())
            url = f"/files/{rel.as_posix()}"
        except Exception:
            url = None
        page_data = {
            "page": page_idx,
            "filename": filename,
            "url": url,
            "elapsed": elapsed,
            "bubbles": [public_bubble(b, i + 1) for i, b in enumerate(bubbles)],
        }
        job_path = JOBS_DIR / f"{job_id}.json"
        pages = _load_pages(job_id) if job_path.exists() else []
        pages.append(page_data)
        _save_pages(job_id, pages)
        emit({"type": "page_done", **page_data})

    def on_finish(stats):
        job["status"] = "cancelled" if job["cancel_event"].is_set() else "done"
        job["stats"] = stats
        emit({"type": "finish", "stats": stats})

    def on_stage(page_idx, stage_key):
        emit({"type": "stage", "page": page_idx, "stage_key": stage_key})

    async def run_job():
        cfg = _load_config(job_id)
        upload_dir = UPLOADS_DIR / job_id
        result_dir = RESULTS_DIR / job_id

        async with MODEL_GATE:
            if job["cancel_event"].is_set():
                return
            try:
                await _run_configured(
                    job_id,
                    mt.process_directory,
                    input_dir=str(upload_dir),
                    output_dir=str(result_dir),
                    target_lang=cfg["target_lang"],
                    font_path=cfg["font_path"],
                    debug=cfg["debug"],
                    fast_mode=cfg.get("fast_mode", False),
                    llm_model=cfg.get("llm_model"),
                    ollama_url=cfg.get("ollama_url"),
                    detect_threshold=cfg.get("detect_threshold"),
                    sfx_threshold=cfg.get("sfx_threshold"),
                    min_bubble_area=cfg.get("min_bubble_area"),
                    max_font_size=cfg.get("max_font_size"),
                    inpaint_shrink=cfg.get("inpaint_shrink"),
                    chunk_size=cfg.get("chunk_size"),
                    translate_retries=cfg.get("translate_retries"),
                    error_log_path=str(JOBS_DIR / f"{job_id}_errors.log"),
                    on_start=on_start,
                    on_page_done=on_page_done,
                    on_finish=on_finish,
                    on_stage=on_stage,
                    cancel_event=job["cancel_event"],
                )

            except Exception as error:
                job["status"] = "failed"
                emit({"type": "error", "message": str(error)})

    try:
        while True:
            recv_task = asyncio.create_task(websocket.receive_json())
            queue_task = asyncio.create_task(job["queue"].get())
            done, pending = await asyncio.wait(
                {recv_task, queue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in pending:
                t.cancel()

            if recv_task in done:
                try:
                    msg = recv_task.result()
                except Exception:
                    break
                if msg.get("action") == "start" and job["status"] == "ready":
                    job["status"] = "queued"
                    job["task"] = asyncio.create_task(run_job())
                elif msg.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})

            if queue_task in done:
                event = queue_task.result()
                await websocket.send_json(event)
                if event.get("type") in ("finish", "error"):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if job["status"] in ("queued", "running"):
            job["cancel_event"].set()


@app.post("/api/job/{job_id}/abort")
async def abort_job(job_id: str):
    _safe_id(job_id)
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    JOBS[job_id]["cancel_event"].set()
    JOBS[job_id]["status"] = "cancelled"
    return {"ok": True}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")
    return _load_pages(job_id)


@app.post("/api/job/{job_id}/page/{page_idx}/render")
@_serialize_model_work
async def re_render_page(job_id: str, page_idx: int, payload: dict):
    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = _load_pages(job_id)
    page = next((p for p in pages if p["page"] == page_idx), None)
    if not page:
        raise HTTPException(404, "Page not found")

    try:
        updates = {bubble["idx"]: bubble for bubble in payload.get("bubbles", [])}
        for bubble in page["bubbles"]:
            if bubble["idx"] in updates:
                update_bubble(bubble, updates[bubble["idx"]])
    except (TypeError, KeyError, ValueError) as error:
        raise HTTPException(400, str(error)) from error

    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    src_path = upload_dir / page["filename"]

    try:
        img_cv = read_image(str(src_path))
    except ValueError as e:
        raise HTTPException(400, str(e))

    annotated = await _run_configured(job_id, draw_results, img_cv, page["bubbles"], debug=False)

    out_path = result_dir / f"{Path(page['filename']).stem}_translated.png"
    write_image(str(out_path), annotated)

    _save_pages(job_id, pages)

    return {
        "ok": True,
        "url": f"/files/results/{job_id}/{out_path.name}",
        "bubbles": page["bubbles"],
    }


@app.post("/api/job/{job_id}/page/{page_idx}/detect-region")
@_serialize_model_work
async def detect_region(job_id: str, page_idx: int, payload: dict):
    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = _load_pages(job_id)
    page = next((p for p in pages if p["page"] == page_idx), None)
    if not page:
        raise HTTPException(404, "Page not found")

    rx = int(payload.get("x", 0))
    ry = int(payload.get("y", 0))
    rw = int(payload.get("w", 0))
    rh = int(payload.get("h", 0))
    if rw < 10 or rh < 10:
        raise HTTPException(400, "Region too small (min 10×10 px)")

    cfg = _load_config(job_id)
    target_lang = cfg.get("target_lang", "Russian")
    font_path = cfg.get("font_path", settings().font_path)
    detect_threshold = float(cfg.get("detect_threshold", settings().detect_threshold))
    min_bubble_area = int(cfg.get("min_bubble_area", settings().min_bubble_area))
    translate_retries = int(cfg.get("translate_retries", settings().translate_retries))

    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    src_path = upload_dir / page["filename"]

    def _run():

        img_cv = read_image(str(src_path))

        ih, iw = img_cv.shape[:2]
        x0, y0, x1, y1 = clip_box(rx, ry, rw, rh, iw, ih)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Region lies outside the image")
        crop_pil = to_pil(img_cv[y0:y1, x0:x1])
        raw = detect_bubbles(crop_pil, threshold=detect_threshold)
        raw = [bubble for bubble in raw if bubble["class"] in ("text_bubble", "text_free")]
        for bubble in raw:
            bubble["x"] += x0
            bubble["y"] += y0

        raw = [b for b in raw if b["width"] * b["height"] >= min_bubble_area]
        if not raw:
            return {"ok": True, "new_bubbles": [], "url": None}

        max_idx = max((b["idx"] for b in page["bubbles"]), default=0)
        for i, b in enumerate(raw):
            b.update(
                idx=max_idx + i + 1,
                text="",
                translation="",
                speaker="unknown",
                gender="unknown",
                font_path=font_path,
                font_size=None,
                text_color=None,
            )

        for b in raw:
            b["text"] = ocr_region(
                img_cv,
                b["x"],
                b["y"],
                b["width"],
                b["height"],
                b["idx"],
                page_idx,
            )
            b["emotion_hint"] = _infer_emotion_tag(b.get("text", ""))

        text_bubbles = [b for b in raw if b.get("text", "").strip()]

        dummy_ctx = MangaContext()
        if text_bubbles:
            translate_batch(
                text_bubbles,
                "",
                dummy_ctx,
                target_lang,
                retries=translate_retries,
            )

        out_path = result_dir / f"{Path(page['filename']).stem}_translated.png"
        page["bubbles"].extend(raw)
        annotated = draw_results(img_cv, page["bubbles"], debug=False)
        write_image(str(out_path), annotated)
        new_page_bubbles = [public_bubble(bubble) for bubble in raw]
        _save_pages(job_id, pages)

        try:
            rel = out_path.resolve().relative_to(BASE_DIR.resolve())
            url = f"/files/{rel.as_posix()}"
        except Exception:
            url = None

        return {"ok": True, "new_bubbles": new_page_bubbles, "url": url}

    try:
        return await _run_configured(job_id, _run)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/job/{job_id}/export")
async def export_job(job_id: str, fmt: str = "zip"):
    if fmt not in ("zip", "cbz"):
        raise HTTPException(400, "fmt must be 'zip' or 'cbz'")

    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = _load_pages(job_id)
    pages_sorted = sorted(pages, key=lambda p: p["page"])
    result_dir = RESULTS_DIR / job_id

    if not result_dir.exists() or not any(result_dir.iterdir()):
        raise HTTPException(404, "No rendered pages found")

    import io
    import zipfile

    buf = io.BytesIO()
    pad = len(str(len(pages_sorted)))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in pages_sorted:
            src_name = Path(p["filename"]).stem
            src_path = result_dir / f"{src_name}_translated.png"
            if not src_path.exists():
                continue
            arc_name = f"{p['page']:0{pad}d}_{src_name}.png"
            zf.write(src_path, arcname=arc_name)

    buf.seek(0)
    content = buf.getvalue()
    if not content:
        raise HTTPException(404, "No translated pages in this job")

    filename = f"translation_{job_id[:8]}.{fmt}"
    media_type = "application/zip" if fmt == "zip" else "application/vnd.comicbook+zip"

    from fastapi.responses import Response

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        },
    )


@app.get("/api/characters")
async def get_characters():
    path = Path("characters.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.put("/api/characters")
@_serialize_model_work
async def save_characters(data: dict):
    write_json("characters.json", data)
    return {"ok": True, "count": len(data)}


@app.delete("/api/characters")
@_serialize_model_work
async def clear_characters():
    path = Path("characters.json")
    if path.exists():
        write_json(path, {})
    return {"ok": True}


@app.delete("/api/characters/{char_id}")
@_serialize_model_work
async def delete_character(char_id: str):
    path = Path("characters.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if char_id not in data:
        raise HTTPException(404, "Character not found")
    del data[char_id]
    write_json(path, data)
    return {"ok": True}


@app.get("/api/glossary")
async def get_glossary():
    return _load_glossary()


@app.post("/api/glossary")
async def add_glossary_entry(payload: dict):
    source = payload.get("source", "").strip()
    target = payload.get("target", "").strip()
    if not source or not target:
        raise HTTPException(400, "source and target are required")
    entries = _load_glossary()
    entries.append({"source": source, "target": target, "note": payload.get("note", "").strip()})
    _save_glossary(entries)
    return {"ok": True, "count": len(entries)}


@app.put("/api/glossary/{idx}")
async def update_glossary_entry(idx: int, payload: dict):
    entries = _load_glossary()
    if idx < 0 or idx >= len(entries):
        raise HTTPException(404, "Entry not found")
    source = payload.get("source", "").strip()
    target = payload.get("target", "").strip()
    if not source or not target:
        raise HTTPException(400, "source and target are required")
    entries[idx] = {"source": source, "target": target, "note": payload.get("note", "").strip()}
    _save_glossary(entries)
    return {"ok": True}


@app.delete("/api/glossary/{idx}")
async def delete_glossary_entry(idx: int):
    entries = _load_glossary()
    if idx < 0 or idx >= len(entries):
        raise HTTPException(404, "Entry not found")
    entries.pop(idx)
    _save_glossary(entries)
    return {"ok": True, "count": len(entries)}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("KOTOBA_HOST", "127.0.0.1")
    port = int(os.environ.get("KOTOBA_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
