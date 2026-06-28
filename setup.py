"""
setup.py — launcher запускаемый из run.bat / run.sh.

Зависимости УЖЕ установлены в venv через run.bat (там портативная установка
по принципу ComfyUI: всё в подпапке venv/, изолированно от системы).
Этот скрипт только:
  1. Проверяет что Ollama запущена и есть подходящие модели (warning, не fatal)
  2. Запускает uvicorn web:app

Если вы запускаете этот файл вручную (минуя run.bat), убедитесь что находитесь
в venv с уже установленными зависимостями.
"""

import sys

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
DIM = "\033[2m"

OLLAMA_URL = "http://localhost:11434"


def info(msg):  print(f"{BLUE}[setup]{RESET} {msg}")
def ok(msg):    print(f"{GREEN}[ ok ]{RESET} {msg}")
def warn(msg):  print(f"{YELLOW}[warn]{RESET} {msg}")
def err(msg):   print(f"{RED}[err ]{RESET} {msg}")


def check_python():
    if sys.version_info < (3, 10):
        err(f"Python 3.10+ required, got {sys.version.split()[0]}")
        sys.exit(1)
    ok(f"Python {sys.version.split()[0]}")


def check_ollama():
    """Проверяет что Ollama запущена и есть хотя бы одна мультимодальная модель."""
    info("Checking Ollama...")
    try:
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = r.json()
    except Exception as e:
        warn(f"Ollama is not reachable at {OLLAMA_URL} ({e})")
        warn("Install from https://ollama.com and start it before translating.")
        warn("You can still launch the web UI to explore the interface.")
        return

    models = data.get("models", [])
    if not models:
        warn("Ollama is running but no models installed.")
        warn("Pull a multimodal model, for example:")
        print(f"   {DIM}ollama pull gemma3:27b{RESET}")
        print(f"   {DIM}ollama pull gemma4:26b{RESET}")
        print(f"   {DIM}ollama pull llava:13b{RESET}")
        warn("Also recommended for OCR:")
        print(f"   {DIM}ollama pull glm-ocr{RESET}")
        return

    # Простая эвристика: имена с известными vision-маркерами
    hints = ("vl", "vision", "llava", "gemma", "qwen", "minicpm",
             "llama4", "pixtral", "molmo", "phi")
    multimodal = [m["name"] for m in models
                   if any(h in m["name"].lower() for h in hints)
                   and "ocr" not in m["name"].lower()]

    if multimodal:
        ok(f"Ollama OK — multimodal models: {', '.join(multimodal[:5])}"
           + (f", +{len(multimodal)-5} more" if len(multimodal) > 5 else ""))
    else:
        warn(f"Ollama has {len(models)} model(s) but none look multimodal.")
        warn("Translation needs a vision-capable model. Try:")
        print(f"   {DIM}ollama pull gemma4:26b{RESET}")

    # OCR-модель
    has_ocr = any("ocr" in m["name"].lower() for m in models)
    if not has_ocr:
        warn("No OCR model detected (glm-ocr recommended). Pull with:")
        print(f"   {DIM}ollama pull glm-ocr{RESET}")


def launch_server():
    info("Starting web server...")
    print(f"{DIM}{'─' * 60}{RESET}")
    print(f"  Opening {GREEN}http://localhost:8000{RESET} in your browser...")
    print(f"  Press Ctrl+C to stop")
    print(f"{DIM}{'─' * 60}{RESET}")

    try:
        import uvicorn
    except ImportError:
        err("uvicorn not installed in this Python environment.")
        err("Run via run.bat (Windows) or run.sh (Linux/Mac) to set up venv automatically.")
        sys.exit(1)

    # КРИТИЧНО: embeddable Python не добавляет cwd в sys.path автоматически,
    # как делает обычный python. Без этого uvicorn.run("web:app", ...) падает
    # с "Could not import module 'web'". Добавляем директорию этого скрипта
    # в sys.path и меняем рабочую директорию на неё.
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    os.chdir(script_dir)

    # Открываем браузер из отдельного потока: дожидаемся пока порт начнёт
    # принимать соединения, потом запускаем системный браузер.
    # Если сделать это до uvicorn.run, страница покажет "сервер недоступен";
    # если использовать фиксированный sleep — на медленных машинах не хватит.
    import threading, socket, time, webbrowser

    def open_browser():
        url = "http://localhost:8000"
        # Ждём пока порт начнёт отвечать (макс. 30 секунд)
        for _ in range(60):
            try:
                with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            warn(f"Server didn't start listening in 30s — open {url} manually")
            return
        try:
            webbrowser.open(url)
        except Exception as e:
            warn(f"Couldn't open browser: {e}. Open {url} manually.")

    threading.Thread(target=open_browser, daemon=True).start()

    # Слушаем только localhost по умолчанию (нет аутентификации). Открыть в
    # локальную сеть осознанно: KOTOBA_HOST=0.0.0.0 перед запуском.
    host = os.environ.get("KOTOBA_HOST", "127.0.0.1")
    try:
        uvicorn.run("web:app", host=host, port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n" + DIM + "Server stopped." + RESET)


def main():
    print(f"\n{BLUE}═══ Kotoba — Manga Translator ═══{RESET}\n")
    check_python()
    check_ollama()
    print()
    launch_server()


if __name__ == "__main__":
    main()
