#!/usr/bin/env bash
# Kotoba — manga translator launcher (Linux / macOS)
#
# При первом запуске скачивает portable Python (python-build-standalone от
# Astral / Indygreg — официальные standalone сборки CPython) в подпапку
# python_embed/, ставит туда все зависимости. После этого системный Python
# не требуется. Аналог ComfyUI portable / Windows embeddable.

set -e

cd "$(dirname "$0")"

PY_VERSION="3.11.9"
PY_DIR="python_embed"
PY_EXE="$PY_DIR/bin/python3"
DEPS_MARKER="$PY_DIR/.deps_installed"

# python-build-standalone хостит готовые сборки для разных платформ
PBS_RELEASE="20240415"   # дата релиза python-build-standalone
PBS_BASE="https://github.com/indygreg/python-build-standalone/releases/download/$PBS_RELEASE"

# ─── 1. Установка portable Python если его нет ────────────────────────────
if [ ! -x "$PY_EXE" ]; then
    echo
    echo "[setup] First run — downloading portable Python $PY_VERSION"
    echo "This is a one-time setup."
    echo

    # Определяем платформу
    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$OS-$ARCH" in
        Linux-x86_64)
            PY_ARCHIVE="cpython-${PY_VERSION}+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"
            ;;
        Linux-aarch64)
            PY_ARCHIVE="cpython-${PY_VERSION}+${PBS_RELEASE}-aarch64-unknown-linux-gnu-install_only.tar.gz"
            ;;
        Darwin-x86_64)
            PY_ARCHIVE="cpython-${PY_VERSION}+${PBS_RELEASE}-x86_64-apple-darwin-install_only.tar.gz"
            ;;
        Darwin-arm64)
            PY_ARCHIVE="cpython-${PY_VERSION}+${PBS_RELEASE}-aarch64-apple-darwin-install_only.tar.gz"
            ;;
        *)
            echo "[ERROR] Unsupported platform: $OS-$ARCH"
            echo "Supported: Linux x86_64/aarch64, macOS x86_64/arm64"
            exit 1
            ;;
    esac

    # Скачиваем (curl или wget)
    if command -v curl &> /dev/null; then
        curl -L -o python_embed.tar.gz "$PBS_BASE/$PY_ARCHIVE"
    elif command -v wget &> /dev/null; then
        wget -O python_embed.tar.gz "$PBS_BASE/$PY_ARCHIVE"
    else
        echo "[ERROR] Neither curl nor wget found. Install one of them and retry."
        exit 1
    fi

    # Распаковываем (архив содержит папку python/ внутри)
    mkdir -p "$PY_DIR"
    tar -xzf python_embed.tar.gz -C "$PY_DIR" --strip-components=1
    rm python_embed.tar.gz

    if [ ! -x "$PY_EXE" ]; then
        echo "[ERROR] Python binary not found after extraction at $PY_EXE"
        exit 1
    fi

    echo
    echo "[setup] Portable Python $PY_VERSION ready."
    echo
fi

# ─── 2. Установка зависимостей ────────────────────────────────────────────
NEED_INSTALL=""
if [ ! -f "$DEPS_MARKER" ]; then
    NEED_INSTALL=1
elif [ "requirements.txt" -nt "$DEPS_MARKER" ]; then
    NEED_INSTALL=1
fi

if [ -n "$NEED_INSTALL" ]; then
    echo "[setup] Installing/updating dependencies into portable Python..."
    echo "This will take several minutes on first run."
    echo

    if ! "$PY_EXE" -m pip install --upgrade pip; then
        echo
        echo "[ERROR] pip upgrade failed."
        exit 1
    fi

    # --upgrade гарантирует замену CPU-сборки torch на CUDA-сборку
    if ! "$PY_EXE" -m pip install --upgrade -r requirements.txt; then
        echo
        echo "[ERROR] Dependency installation failed."
        exit 1
    fi

    touch "$DEPS_MARKER"
    echo
    echo "[setup] All dependencies installed."
    echo
fi

# ─── 3. Запуск сервера ─────────────────────────────────────────────────────
"$PY_EXE" setup.py
