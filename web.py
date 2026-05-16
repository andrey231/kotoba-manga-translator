"""
web.py
Веб-интерфейс для manga_translator.py

Запуск:
    uvicorn web:app --host 0.0.0.0 --port 8000

Откройте http://localhost:8000 в браузере.
"""

import os
import json
import shutil
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import manga_translator as mt


# ─── состояние и пути ─────────────────────────────────────────────────────────

BASE_DIR = Path("web_data")
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
JOBS_DIR = BASE_DIR / "jobs"          # bubbles.json по каждому job_id

for d in (UPLOADS_DIR, RESULTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Активные джобы: job_id → {"status", "pages", "stats", "ws_queue"}
JOBS: dict = {}


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Kotoba — Manga Translator")

# Раздаём результаты статикой — браузер сможет тянуть картинки напрямую
app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")


# ─── главная страница ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "web_ui.html").read_text(encoding="utf-8"))


# ─── список доступных LLM-моделей из Ollama ──────────────────────────────────

# URL Ollama можно переопределить переменной окружения OLLAMA_HOST
# (Ollama использует ту же переменную). По умолчанию — localhost:11434.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"


def _fetch_ollama_models() -> tuple[list, str | None]:
    """
    Запрашивает /api/tags у Ollama и возвращает (список_моделей, ошибка_или_None).
    Если что-то пошло не так — пишет в консоль развёрнутую диагностику.
    """
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


# Имена которые ТОЧНО мультимодальные — фильтр должен пропускать их без вопросов
KNOWN_MULTIMODAL = (
    "llava", "bakllava", "moondream", "minicpm-v", "minicpm",
    "qwen2-vl", "qwen2.5-vl", "qwen-vl",
    "llama3.2-vision", "llama4",
    "pixtral", "molmo",
    "gemma3", "gemma4",  # обе поддерживают vision
    "phi3.5-vision", "phi-3-vision", "phi3-vision", "phi4-vision",
    "internvl", "cogvlm", "yi-vl",
)
# Семейства из ollama show — означают что у модели есть vision-проектор
MULTIMODAL_FAMILIES = {"clip", "mllama", "llava", "gemma3", "gemma4"}
# OCR-модели не подходят для перевода/анализа сцены, прячем их
OCR_FAMILIES = {"glmocr"}
OCR_NAME_HINTS = ("glm-ocr", "tesseract", "paddleocr", "easyocr")
# Не-мультимодальные модели которые могут случайно матчиться по подстроке "gemma" или "phi"
NEVER_MULTIMODAL = ("gemma:", "gemma2:", "gemma2-", "phi3:", "phi3-mini", "phi:")


def _is_ocr(name: str, family: str, families: set) -> bool:
    """OCR-модели исключаем из списка LLM."""
    name_lower = name.lower()
    if any(hint in name_lower for hint in OCR_NAME_HINTS):
        return True
    if family.lower() in OCR_FAMILIES:
        return True
    if families & OCR_FAMILIES:
        return True
    return False


def _is_multimodal(name: str, family: str, families: set) -> bool:
    """Решает мультимодальная модель или нет, на основе имени + метаданных Ollama."""
    name_lower = name.lower()
    # Жёсткое исключение
    if any(name_lower.startswith(p) for p in NEVER_MULTIMODAL):
        return False
    # Жёсткое включение по имени (subscring match, чтобы ловить
    # huihui_ai/gemma-4-abliterated:26b, ollama пользовательских сборок и т.п.)
    if any(known in name_lower for known in KNOWN_MULTIMODAL):
        return True
    # gemma-4 как имя файла без 'gemma4' — например 'gemma-4-abliterated'
    if "gemma-4" in name_lower or "gemma-3" in name_lower:
        return True
    # Метаданные Ollama: family/families указывают на vision
    if family.lower() in MULTIMODAL_FAMILIES:
        return True
    if families & MULTIMODAL_FAMILIES:
        return True
    return False


@app.get("/api/models")
async def list_models():
    """Список Ollama-моделей с пометкой какие мультимодальные / OCR."""
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
    # Multimodal first, then "other" LLMs, OCR last
    out.sort(key=lambda m: (m["ocr"], not m["multimodal"], m["name"]))
    return {
        "models": out,
        "default": mt.LLM_MODEL,
        "ollama_host": OLLAMA_HOST,
        "error": error,
    }


@app.get("/api/models/debug")
async def debug_models():
    """
    Возвращает СЫРОЙ ответ Ollama для диагностики.
    Открой http://localhost:8000/api/models/debug в браузере чтобы посмотреть.
    """
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


# ─── загрузка главы ──────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_chapter(
    files: list[UploadFile] = File(...),
    target_lang: str = Form("Russian"),
    font_path: str = Form("arial.ttf"),
    debug: bool = Form(False),
    llm_model: str = Form(""),
    llm_debug: bool = Form(False),
    fast_mode: bool = Form(False),
):
    """
    Создаёт job, сохраняет файлы, возвращает job_id.
    Сам процесс перевода запустится отдельно через POST /api/start/{job_id}.
    """
    job_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in mt.SUPPORTED_EXTENSIONS:
            continue
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
        },
        "total": len(saved),
        "completed": 0,
        "pages": [],
        "stats": None,
        "queue": asyncio.Queue(),
    }
    return {"job_id": job_id, "total_pages": len(saved)}


# ─── запуск перевода и WebSocket прогресс ────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """
    Открывается клиентом сразу после upload.
    Клиент шлёт {"action": "start"} — запускаем перевод.
    Сервер шлёт события {type: "start"|"page_done"|"finish"|"error"|"log"}.
    """
    await websocket.accept()
    if job_id not in JOBS:
        await websocket.send_json({"type": "error", "message": "Unknown job_id"})
        await websocket.close()
        return

    job = JOBS[job_id]
    loop = asyncio.get_running_loop()

    # Коллбэки бегают в синхронном потоке (process_directory блокирующий) —
    # нужно перебрасывать события в asyncio через call_soon_threadsafe
    def emit(payload: dict):
        loop.call_soon_threadsafe(job["queue"].put_nowait, payload)

    def on_start(total):
        job["status"] = "running"
        emit({"type": "start", "total": total})

    def on_page_done(page_idx, total, filename, output_path, bubbles, elapsed):
        job["completed"] = page_idx
        # Превращаем абсолютный путь в URL для браузера
        try:
            rel = Path(output_path).resolve().relative_to(BASE_DIR.resolve())
            url = f"/files/{rel.as_posix()}"
        except Exception:
            url = None
        # Минимальное представление баблов для редактора
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
                    "text": b.get("text", ""),
                    "translation": b.get("translation", ""),
                    "speaker": b.get("speaker", "unknown"),
                    "gender": b.get("gender", "unknown"),
                    "font_path": b.get("font_path"),    # per-bubble override (None = use default)
                    "font_size": b.get("font_size"),    # per-bubble override (None = auto-fit)
                }
                for i, b in enumerate(bubbles)
            ],
        }
        job["pages"].append(page_data)
        # Сохраняем по job_id, чтобы редактор мог потом править
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
        """Запускает синхронный перевод в отдельном потоке."""
        cfg = job["config"]
        upload_dir = UPLOADS_DIR / job_id
        result_dir = RESULTS_DIR / job_id

        # Включаем подробный лог LLM для этого джоба если запросили
        mt.DEBUG_LLM = bool(cfg.get("llm_debug"))
        if mt.DEBUG_LLM:
            print("\n*** VERBOSE LLM LOGGING ENABLED — every prompt/response will be printed ***\n")

        await asyncio.to_thread(
            mt.process_directory,
            input_dir=str(upload_dir),
            output_dir=str(result_dir),
            target_lang=cfg["target_lang"],
            font_path=cfg["font_path"],
            debug=cfg["debug"],
            fast_mode=cfg.get("fast_mode", False),
            llm_model=cfg.get("llm_model"),
            error_log_path=str(JOBS_DIR / f"{job_id}_errors.log"),
            on_start=on_start,
            on_page_done=on_page_done,
            on_finish=on_finish,
            on_stage=on_stage,
        )

    runner_task: Optional[asyncio.Task] = None
    try:
        while True:
            # Ждём либо команды от клиента, либо событий из очереди
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
        pass
    finally:
        if runner_task and not runner_task.done():
            # Даём джобу довыполниться в фоне — он сам зачистит state
            pass


# ─── редактор: правка переводов ──────────────────────────────────────────────

@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    """Возвращает все страницы джоба с баблами для редактора."""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")
    return json.loads(job_file.read_text(encoding="utf-8"))


@app.post("/api/job/{job_id}/page/{page_idx}/render")
async def re_render_page(job_id: str, page_idx: int, payload: dict):
    """
    Перерисовывает страницу с обновлёнными переводами.
    payload: {"bubbles": [{"idx": 1, "translation": "новый перевод"}, ...]}
    """
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = json.loads(job_file.read_text(encoding="utf-8"))
    page = next((p for p in pages if p["page"] == page_idx), None)
    if not page:
        raise HTTPException(404, "Page not found")

    # Применяем обновления баблов (translation, font_path, font_size).
    # Только указанные в payload поля затрагиваются — остальные сохраняются.
    updates = {b["idx"]: b for b in payload.get("bubbles", []) if "idx" in b}
    for b in page["bubbles"]:
        if b["idx"] in updates:
            u = updates[b["idx"]]
            if "translation" in u:
                b["translation"] = u.get("translation") or ""
            if "font_path" in u:
                # None или пустая строка — сбрасываем override
                b["font_path"] = u["font_path"] if u["font_path"] else None
            if "font_size" in u:
                b["font_size"] = u["font_size"] if u["font_size"] else None

    # Перерисовываем страницу
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    src_path = upload_dir / page["filename"]

    import cv2
    import numpy as np
    img_cv = cv2.imread(str(src_path))
    if img_cv is None:
        img_cv = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8),
                              cv2.IMREAD_COLOR)

    # Восстанавливаем структуру bubbles в формате draw_results
    bubbles_for_draw = [
        {
            "x": b["x"], "y": b["y"],
            "width": b["w"], "height": b["h"],
            "text": b.get("text", ""),
            "translation": b.get("translation", ""),
            "class": "text_bubble",   # для цвета рамки, влияет только в debug
            "speaker": b.get("speaker", ""),
            "gender": b.get("gender", ""),
            # Per-bubble overrides — могут быть None
            "font_path": b.get("font_path"),
            "font_size": b.get("font_size"),
        }
        for b in page["bubbles"]
    ]
    annotated = mt.draw_results(img_cv, bubbles_for_draw, debug=False)

    out_path = result_dir / f"{Path(page['filename']).stem}_translated.png"
    cv2.imwrite(str(out_path), annotated)

    # Сохраняем обновлённые баблы обратно
    job_file.write_text(json.dumps(pages, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    return {"ok": True, "url": f"/files/results/{job_id}/{out_path.name}"}


# ─── архив персонажей ─────────────────────────────────────────────────────────

@app.get("/api/job/{job_id}/export")
async def export_job(job_id: str, fmt: str = "zip"):
    """
    Собирает все переведённые страницы джоба в архив (zip или cbz).
    fmt: 'zip' (по умолчанию) или 'cbz' (тот же zip с расширением .cbz —
    стандарт для манги/комиксов, открывается ридерами вроде CDisplayEx).
    """
    if fmt not in ("zip", "cbz"):
        raise HTTPException(400, "fmt must be 'zip' or 'cbz'")

    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    pages = json.loads(job_file.read_text(encoding="utf-8"))
    pages_sorted = sorted(pages, key=lambda p: p["page"])
    result_dir = RESULTS_DIR / job_id

    if not result_dir.exists() or not any(result_dir.iterdir()):
        raise HTTPException(404, "No rendered pages found")

    # Используем in-memory буфер; для главы из 50-100 страниц это ~20-50MB
    import io
    import zipfile

    buf = io.BytesIO()
    pad = len(str(len(pages_sorted)))   # для 100 страниц → 3-значное паддинг
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in pages_sorted:
            # Имя файла в архиве: 001_page.png, 002_page.png, ...
            # Ридеры манги сортируют именно так
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

    # Имя файла для скачивания
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
    """Возвращает содержимое characters.json."""
    path = Path("characters.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.put("/api/characters")
async def save_characters(data: dict):
    """Сохраняет characters.json после редактирования."""
    Path("characters.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "count": len(data)}


@app.delete("/api/characters")
async def clear_characters():
    """Очищает весь архив персонажей."""
    path = Path("characters.json")
    if path.exists():
        path.write_text("{}", encoding="utf-8")
    return {"ok": True}


@app.delete("/api/characters/{char_id}")
async def delete_character(char_id: str):
    """Удаляет одного персонажа из архива."""
    path = Path("characters.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if char_id not in data:
        raise HTTPException(404, "Character not found")
    del data[char_id]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
