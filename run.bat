@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul
title Kotoba — Manga Translator

cd /d "%~dp0"

set "PY_VERSION=3.11.9"
set "PY_DIR=python_embed"
set "PY_EXE=%PY_DIR%\python.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-embed-amd64.zip"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "DEPS_MARKER=%PY_DIR%\.deps_installed"

REM --- 1. Install portable Python if missing ---
if exist "%PY_EXE%" goto :have_python

echo.
echo [setup] First run - downloading portable Python %PY_VERSION%
echo One-time setup. Approx 10 MB download.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%PY_URL%' -OutFile 'python_embed.zip' -UseBasicParsing"
if errorlevel 1 goto :download_failed

powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath '%PY_DIR%' -Force"
if errorlevel 1 goto :extract_failed

del python_embed.zip

REM Patch python*._pth to enable site-packages.
REM Embeddable distribution ships with an isolated config; we need to uncomment
REM 'import site' and add Lib\site-packages so installed packages are visible.
set "PATCH=%PY_DIR%\_patch_pth.py"
> "%PATCH%" echo import glob, sys, os
>> "%PATCH%" echo def fix^(p^):
>> "%PATCH%" echo  lines=[^('import site' if l.strip^(^)=='#import site' else l^) for l in open^(p^).read^(^).splitlines^(^)]
>> "%PATCH%" echo  lines.append^('Lib\\site-packages'^) if 'Lib\\site-packages' not in lines else None
>> "%PATCH%" echo  open^(p,'w'^).write^('\n'.join^(lines^)+'\n'^)
>> "%PATCH%" echo [fix^(p^) for p in glob.glob^(os.path.join^(sys.argv[1], 'python*._pth'^)^)]

"%PY_EXE%" "%PATCH%" "%PY_DIR%"
if errorlevel 1 goto :patch_failed
del "%PATCH%"

echo [setup] Installing pip into portable Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%GET_PIP_URL%' -OutFile '%PY_DIR%\get-pip.py' -UseBasicParsing"
if errorlevel 1 goto :pip_bootstrap_failed

"%PY_EXE%" "%PY_DIR%\get-pip.py" --no-warn-script-location
if errorlevel 1 goto :pip_install_failed
del "%PY_DIR%\get-pip.py"

echo.
echo [setup] Portable Python %PY_VERSION% ready.
echo.

:have_python

REM --- 2. Install dependencies if missing or requirements changed ---
set "NEED_INSTALL="
if not exist "%DEPS_MARKER%" set "NEED_INSTALL=1"
if exist "%DEPS_MARKER%" call :check_req_mtime

if not defined NEED_INSTALL goto :deps_ok

echo [setup] Installing dependencies into portable Python...
echo This will take several minutes on first run.
echo.

"%PY_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :pip_deps_failed

REM --upgrade гарантирует что CPU-сборка torch будет заменена на CUDA-сборку
REM (или наоборот) если изменился --extra-index-url в requirements.txt
"%PY_EXE%" -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :pip_deps_failed

echo. > "%DEPS_MARKER%"
echo.
echo [setup] All dependencies installed.
echo.

:deps_ok

REM --- 3. Launch server ---
"%PY_EXE%" setup.py
if errorlevel 1 goto :server_failed
pause
exit /b 0


REM --- Helper: compare mtimes, sets NEED_INSTALL if requirements.txt is newer ---
:check_req_mtime
"%PY_EXE%" -c "import os,sys; sys.exit(0 if os.path.getmtime('requirements.txt') <= os.path.getmtime(sys.argv[1]) else 1)" "%DEPS_MARKER%"
if errorlevel 1 set "NEED_INSTALL=1"
exit /b 0


REM --- Error handlers ---
:download_failed
echo [ERROR] Failed to download Python.
echo Check internet connection, firewall, antivirus.
pause
exit /b 1

:extract_failed
echo [ERROR] Failed to extract Python archive.
pause
exit /b 1

:patch_failed
echo [ERROR] Failed to patch Python path config.
pause
exit /b 1

:pip_bootstrap_failed
echo [ERROR] Failed to download get-pip.py
pause
exit /b 1

:pip_install_failed
echo [ERROR] Failed to install pip into portable Python.
pause
exit /b 1

:pip_deps_failed
echo.
echo [ERROR] Dependency installation failed.
echo Common causes:
echo   - No internet connection
echo   - Antivirus blocking pip
echo   - Corporate proxy needing configuration
echo.
pause
exit /b 1

:server_failed
echo.
echo [ERROR] Server exited with an error. See messages above.
pause
exit /b 1
