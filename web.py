"""
web.py
Web interface for manga_translator.py

Run:
    uvicorn web:app --host 127.0.0.1 --port 8000

Open http://localhost:8000 in your browser.
"""

import io
import os
import re
import json
import shutil
import asyncio
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import manga_translator as mt


BASE_DIR = Path("web_data")
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
JOBS_DIR = BASE_DIR / "jobs"
GLOSSARY_FILE = BASE_DIR / "glossary.json"

for d in (UPLOADS_DIR, RESULTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


_MAX_EXTRACT_BYTES = 4 * 1024 ** 3

_JOB_ID_RE = re.compile(r"[0-9a-f]{6,32}")


def _safe_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id")
    return job_id


def _hex_to_rgb(value: str | None) -> list[int] | None:
    h = (value or "").lstrip("#")
    if len(h) != 6:
        return None
    try:
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return None


def _load_glossary() -> list:
    if GLOSSARY_FILE.exists():
        try:
            return json.loads(GLOSSARY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_glossary(entries: list) -> None:
    GLOSSARY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    mt.GLOSSARY = entries


mt.GLOSSARY = _load_glossary()

JOBS: dict = {}


app = FastAPI(title="Kotoba — Manga Translator")

app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "web_ui.html").read_text(encoding="utf-8"))


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"


def _fetch_ollama_models() -> tuple[list, str | None]:
    import requests as _req
    url = f"{OLLAMA_HOST}/api/tags"
    try:
        r = _req.get(url, timeout=5)
    except _req.exceptions.ConnectionError as e:
        msg = (f"Cannot connect to Ollama at {url}. "
               "Make sure Ollama is running (try 'ollama list' in terminal). "
               f"Details: {e}")
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
    "llava", "bakllava", "moondream", "minicpm-v", "minicpm",
    "qwen2-vl", "qwen2.5-vl", "qwen-vl",
    "llama3.2-vision", "llama4",
    "pixtral", "molmo",
    "gemma3", "gemma4",
    "phi3.5-vision", "phi-3-vision", "phi3-vision", "phi4-vision",
    "internvl", "cogvlm", "yi-vl",
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
        family = (details.get("family") or "")
        size_bytes = m.get("size", 0)

        ocr = _is_ocr(name, family, families)
        is_multi = _is_multimodal(name, family, families) if not ocr else False

        out.append({
            "name": name,
            "size_gb": round(size_bytes / (1024 ** 3), 1),
            "family": family,
            "families": sorted(families),
            "multimodal": is_multi,
            "ocr": ocr,
        })
    out.sort(key=lambda m: (m["ocr"], not m["multimodal"], m["name"]))
    return {
        "models": out,
        "default": mt.LLM_MODEL,
        "ollama_host": OLLAMA_HOST,
        "error": error,
    }


def _system_font_dirs() -> list[str]:
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


@app.get("/api/fonts")
async def list_fonts():
    from PIL import ImageFont

    fonts: list[dict] = []
    seen: set[str] = set()
    for root_dir in _system_font_dirs():
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
                    fonts.append({"file": fname, "path": path,
                                  "family": family, "style": style, "display": label})
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
    job_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    extracted_total = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext in {'.zip', '.cbz'}:
            data = await f.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for entry in sorted(zf.infolist(), key=lambda e: e.filename):
                    if entry.is_dir():
                        continue
                    name = Path(entry.filename).name
                    if name.startswith('._') or entry.filename.startswith('__MACOSX'):
                        continue
                    if Path(name).suffix.lower() not in mt.SUPPORTED_EXTENSIONS:
                        continue
                    extracted_total += entry.file_size
                    if extracted_total > _MAX_EXTRACT_BYTES:
                        raise HTTPException(
                            413, "Archive too large when decompressed "
                                 f"(> {_MAX_EXTRACT_BYTES // (1024**3)} GB)")
                    (upload_dir / name).write_bytes(zf.read(entry))
        elif ext in mt.SUPPORTED_EXTENSIONS:
            target = upload_dir / f.filename
            with open(target, "wb") as out:
                shutil.copyfileobj(f.file, out)

    saved = sorted(upload_dir.iterdir(), key=lambda p: mt.natural_key(p.name))
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
        "total": len(saved),
        "completed": 0,
        "pages": [],
        "stats": None,
        "queue": asyncio.Queue(),
        "cancel_event": threading.Event(),
    }
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
        job["completed"] = page_idx
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
            "bubbles": [
                {
                    "idx": i + 1,
                    "x": b.get("x"), "y": b.get("y"),
                    "w": b.get("width"), "h": b.get("height"),
                    "orig_w": b.get("width"), "orig_h": b.get("height"),
                    "text": b.get("text", ""),
                    "translation": b.get("translation", ""),
                    "speaker": b.get("speaker", "unknown"),
                    "gender": b.get("gender", "unknown"),
                    "font_path": b.get("font_path"),
                    "font_size": b.get("font_size"),
                    "text_color": list(b["_text_color"]) if b.get("_text_color") else None,
                    "class": b.get("class", "text_bubble"),
                    "_text_angle": b.get("_text_angle", 0.0),
                }
                for i, b in enumerate(bubbles)
            ],
        }
        job["pages"].append(page_data)
        (JOBS_DIR / f"{job_id}.json").write_text(
            json.dumps(job["pages"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit({"type": "page_done", **page_data})

    def on_finish(stats):
        job["status"] = "done"
        job["stats"] = stats
        emit({"type": "finish", "stats": stats})

    def on_stage(page_idx, stage_key):
        emit({"type": "stage", "page": page_idx, "stage_key": stage_key})

    async def run_job():
        cfg = job["config"]
        upload_dir = UPLOADS_DIR / job_id
        result_dir = RESULTS_DIR / job_id

        mt.DEBUG_LLM = bool(cfg.get("llm_debug"))
        if mt.DEBUG_LLM:
            print("\n*** VERBOSE LLM LOGGING ENABLED — every prompt/response will be printed ***\n")

        crops_dir = result_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        mt.CROPS_DIR = str(crops_dir)

        if cfg.get("mask_debug"):
            mask_dbg_dir = result_dir / "mask_debug"
            mask_dbg_dir.mkdir(exist_ok=True)
            mt.MASK_DEBUG_DIR = str(mask_dbg_dir)
            print(f"[mask-dbg] Mask debug enabled → {mask_dbg_dir}")
        else:
            mt.MASK_DEBUG_DIR = None

        await asyncio.to_thread(
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

    runner_task: Optional[asyncio.Task] = None
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
                    runner_task = asyncio.create_task(run_job())
                elif msg.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})

            if queue_task in done:
                event = queue_task.result()
                await websocket.send_json(event)
                if event.get("type") == "finish":
                    break
    except WebSocketDisconnect:
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
    return json.loads(job_file.read_text(encoding="utf-8"))


@app.post("/api/job/{job_id}/page/{page_idx}/render")
async def re_render_page(job_id: str, page_idx: int, payload: dict):
    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = json.loads(job_file.read_text(encoding="utf-8"))
    page = next((p for p in pages if p["page"] == page_idx), None)
    if not page:
        raise HTTPException(404, "Page not found")

    updates = {b["idx"]: b for b in payload.get("bubbles", []) if "idx" in b}
    for b in page["bubbles"]:
        if b["idx"] in updates:
            u = updates[b["idx"]]
            if "translation" in u:
                b["translation"] = u.get("translation") or ""
            if "font_path" in u:
                b["font_path"] = u["font_path"] if u["font_path"] else None
            if "font_size" in u:
                b["font_size"] = u["font_size"] if u["font_size"] else None
            if "text_color" in u:
                rgb = _hex_to_rgb(u.get("text_color"))
                if rgb is not None:
                    b["text_color"] = rgb
            if "outline_color" in u:
                b["outline_color"] = _hex_to_rgb(u.get("outline_color"))
            if "outline_width" in u:
                b["outline_width"] = int(u.get("outline_width") or 0)
            for flag in ("bold", "italic", "underline"):
                if flag in u:
                    b[flag] = bool(u[flag])
            if "text_align" in u:
                b["text_align"] = u.get("text_align") or "center"
            if "text_angle" in u:
                b["_text_angle"] = float(u.get("text_angle") or 0)
            if any(k in u for k in ("box_cx", "box_cy", "box_sw", "box_sh")):
                cx = float(u.get("box_cx", b["x"] + b["w"] / 2))
                cy = float(u.get("box_cy", b["y"] + b["h"] / 2))
                sw = max(0.05, float(u.get("box_sw") or 1.0))
                sh = max(0.05, float(u.get("box_sh") or 1.0))
                base_w = b.get("orig_w") or b["w"]
                base_h = b.get("orig_h") or b["h"]
                nw = max(4, round(base_w * sw))
                nh = max(4, round(base_h * sh))
                b["x"] = max(0, round(cx - nw / 2))
                b["y"] = max(0, round(cy - nh / 2))
                b["w"] = nw
                b["h"] = nh

    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    src_path = upload_dir / page["filename"]

    import cv2
    try:
        img_cv = mt.read_image(str(src_path))
    except ValueError as e:
        raise HTTPException(400, str(e))

    bubbles_for_draw = [
        {
            "x": b["x"], "y": b["y"],
            "width": b["w"], "height": b["h"],
            "text": b.get("text", ""),
            "translation": b.get("translation", ""),
            "speaker": b.get("speaker", ""),
            "gender": b.get("gender", ""),
            "font_path": b.get("font_path"),
            "font_size": b.get("font_size"),
            "_text_color": tuple(b["text_color"]) if b.get("text_color") else None,
            "class": b.get("class", "text_bubble"),
            "bold": b.get("bold", False),
            "italic": b.get("italic", False),
            "underline": b.get("underline", False),
            "text_align": b.get("text_align", "center"),
            "outline_color": b.get("outline_color"),
            "outline_width": b.get("outline_width", 0),
            "_text_angle": b.get("_text_angle", 0.0),
        }
        for b in page["bubbles"]
    ]
    annotated = mt.draw_results(img_cv, bubbles_for_draw, debug=False)

    out_path = result_dir / f"{Path(page['filename']).stem}_translated.png"
    cv2.imwrite(str(out_path), annotated)

    job_file.write_text(json.dumps(pages, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    return {"ok": True, "url": f"/files/results/{job_id}/{out_path.name}"}


@app.post("/api/job/{job_id}/page/{page_idx}/detect-region")
async def detect_region(job_id: str, page_idx: int, payload: dict):
    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = json.loads(job_file.read_text(encoding="utf-8"))
    page = next((p for p in pages if p["page"] == page_idx), None)
    if not page:
        raise HTTPException(404, "Page not found")

    rx = int(payload.get("x", 0))
    ry = int(payload.get("y", 0))
    rw = int(payload.get("w", 0))
    rh = int(payload.get("h", 0))
    if rw < 10 or rh < 10:
        raise HTTPException(400, "Region too small (min 10×10 px)")

    cfg = JOBS.get(job_id, {}).get("config", {})
    target_lang        = cfg.get("target_lang", "Russian")
    font_path          = cfg.get("font_path", mt.DEFAULT_FONT)
    detect_threshold   = float(cfg.get("detect_threshold", mt.DETECT_THRESHOLD))
    min_bubble_area    = int(cfg.get("min_bubble_area", mt.MIN_BUBBLE_AREA))
    translate_retries  = int(cfg.get("translate_retries", mt.TRANSLATE_RETRIES))

    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    src_path   = upload_dir / page["filename"]

    def _run():
        import cv2
        from PIL import Image as PILImage

        img_cv = mt.read_image(str(src_path))

        ih, iw = img_cv.shape[:2]
        x0 = max(0, min(rx, iw - 1))
        y0 = max(0, min(ry, ih - 1))
        x1 = min(iw, x0 + rw)
        y1 = min(ih, y0 + rh)
        cw, ch = x1 - x0, y1 - y0

        DET = 640
        crop_cv = img_cv[y0:y1, x0:x1]
        resized = cv2.resize(crop_cv, (DET, DET), interpolation=cv2.INTER_LINEAR)
        crop_pil = PILImage.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        raw = mt.detect_bubbles(crop_pil, threshold=detect_threshold)
        if not raw:
            return {"ok": True, "new_bubbles": [], "url": None}

        sx, sy = cw / DET, ch / DET
        for b in raw:
            b["x"]      = max(0, min(round(b["x"]      * sx) + x0, iw - 1))
            b["y"]      = max(0, min(round(b["y"]      * sy) + y0, ih - 1))
            b["width"]  = max(1, min(round(b["width"]  * sx), iw - b["x"]))
            b["height"] = max(1, min(round(b["height"] * sy), ih - b["y"]))

        raw = [b for b in raw if b["width"] * b["height"] >= min_bubble_area]
        if not raw:
            return {"ok": True, "new_bubbles": [], "url": None}

        max_idx = max((b["idx"] for b in page["bubbles"]), default=0)
        for i, b in enumerate(raw):
            b.update(idx=max_idx + i + 1, text="", translation="",
                     speaker="unknown", gender="unknown",
                     font_path=font_path, font_size=None,
                     _text_color=None, _sam2_mask=None, _text_angle=0.0)

        crops_dir = result_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        mt.CROPS_DIR = str(crops_dir)

        for b in raw:
            b["text"] = mt.ocr_region(
                img_cv, b["x"], b["y"], b["width"], b["height"],
                b["idx"], page_idx,
            )
            b["emotion_hint"] = mt._infer_emotion_tag(b.get("text", ""))

        text_bubbles = [b for b in raw if b.get("text", "").strip()]

        for b in text_bubbles:
            b["translation"] = b["text"]
        mt._compute_text_masks(img_cv, text_bubbles)
        for b in text_bubbles:
            b["translation"] = ""
            if b.get("_text_color") is None:
                b["_text_color"] = mt.detect_text_color(img_cv, b)

        dummy_ctx = mt.MangaContext()
        if text_bubbles:
            mt.translate_batch(
                text_bubbles, "", dummy_ctx, target_lang,
                retries=translate_retries,
            )

        out_path = result_dir / f"{Path(page['filename']).stem}_translated.png"
        if out_path.exists():
            base_cv = cv2.imread(str(out_path))
            if base_cv is None:
                base_cv = img_cv.copy()
        else:
            base_cv = img_cv.copy()

        bubbles_for_draw = [
            {
                "x": b["x"], "y": b["y"],
                "width": b["width"], "height": b["height"],
                "text": b.get("text", ""),
                "translation": b.get("translation", ""),
                "speaker": b.get("speaker", ""),
                "gender": b.get("gender", ""),
                "font_path": b.get("font_path"),
                "font_size": None,
                "_text_color": b.get("_text_color"),
                "_sam2_mask":  b.get("_sam2_mask"),
                "class": b.get("class", "text_bubble"),
                "bold": False, "italic": False, "underline": False,
                "text_align": "center",
                "outline_color": None, "outline_width": 0,
                "_text_angle": b.get("_text_angle", 0.0),
            }
            for b in raw if b.get("translation")
        ]

        if bubbles_for_draw:
            annotated = mt.draw_results(base_cv, bubbles_for_draw, debug=False)
            cv2.imwrite(str(out_path), annotated)

        def _color(c):
            return list(c) if c else None

        new_page_bubbles = [
            {
                "idx": b["idx"],
                "x": b["x"], "y": b["y"],
                "w": b["width"], "h": b["height"],
                "orig_w": b["width"], "orig_h": b["height"],
                "text": b.get("text", ""),
                "translation": b.get("translation", ""),
                "speaker": b.get("speaker", "unknown"),
                "gender": b.get("gender", "unknown"),
                "font_path": None,
                "font_size": None,
                "text_color": _color(b.get("_text_color")),
                "class": b.get("class", "text_bubble"),
                "_text_angle": b.get("_text_angle", 0.0),
            }
            for b in raw
        ]

        page["bubbles"].extend(new_page_bubbles)
        job_file.write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            rel = out_path.resolve().relative_to(BASE_DIR.resolve())
            url = f"/files/{rel.as_posix()}"
        except Exception:
            url = None

        return {"ok": True, "new_bubbles": new_page_bubbles, "url": url}

    try:
        return await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(500, str(e))


@app.get("/api/job/{job_id}/export")
async def export_job(job_id: str, fmt: str = "zip"):
    if fmt not in ("zip", "cbz"):
        raise HTTPException(400, "fmt must be 'zip' or 'cbz'")

    _safe_id(job_id)
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = json.loads(job_file.read_text(encoding="utf-8"))
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
async def save_characters(data: dict):
    Path("characters.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "count": len(data)}


@app.delete("/api/characters")
async def clear_characters():
    path = Path("characters.json")
    if path.exists():
        path.write_text("{}", encoding="utf-8")
    return {"ok": True}


@app.delete("/api/characters/{char_id}")
async def delete_character(char_id: str):
    path = Path("characters.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if char_id not in data:
        raise HTTPException(404, "Character not found")
    del data[char_id]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                     encoding="utf-8")
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
