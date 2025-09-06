@echo off
setlocal enabledelayedexpansion

REM filepath: c:\Users\jussi\source\repos\node_audio\start-windows.bat
REM Automatic installation and startup script for Windows 10/11.

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "PY_REQUIRED_MAJOR=3"
set "PY_REQUIRED_MINOR=11"

set "FLAG_CHECK_ONLY=false"
set "FLAG_RECREATE_VENV=false"
set "FLAG_WITH_EXTRAS=false"
set "APP_ARGS="
set "ERROR_COUNT=0"
set "SUMMARY_ERRORS="

REM --- Argument Parsing ---
:PARSE_ARGS_LOOP
if "%~1"=="" goto :END_PARSE_ARGS
if /i "%~1"=="--check-only" (
    set "FLAG_CHECK_ONLY=true"
    shift
    goto :PARSE_ARGS_LOOP
)
if /i "%~1"=="--recreate-venv" (
    set "FLAG_RECREATE_VENV=true"
    shift
    goto :PARSE_ARGS_LOOP
)
if /i "%~1"=="--with-extras" (
    set "FLAG_WITH_EXTRAS=true"
    shift
    goto :PARSE_ARGS_LOOP
)
if /i "%~1"=="--clean" (
    set "APP_ARGS=%APP_ARGS% --clean"
    shift
    goto :PARSE_ARGS_LOOP
)
if /i "%~1"=="-h" (
    call :usage
    exit /b 0
)
if /i "%~1"=="--help" (
    call :usage
    exit /b 0
)
echo [FAIL] Unknown option: %1
call :usage
exit /b 2
:END_PARSE_ARGS

REM --- Main Execution ---
call :check_python || set "PYTHON_CHECK_FAILED=1"
if defined PYTHON_CHECK_FAILED (
    echo [INFO] Python not available. Will offer to install in auto-fix step.
)

call :ensure_venv || set "VENV_FAILED=1"
if defined VENV_FAILED (
    echo [INFO] Virtual environment missing. Will offer to create in auto-fix step.
)

call :ensure_core_requirements || set "CORE_REQ_FAILED=1"
call :verify_runtime || set "RUNTIME_FAILED=1"

if "%FLAG_WITH_EXTRAS%"=="true" (
    call :install_extras_auto || set "EXTRAS_FAILED=1"
)

if "%FLAG_CHECK_ONLY%"=="true" (
    call :summary_and_exit
)

if %ERROR_COUNT% gtr 0 (
    call :auto_fix_flow
)

:LAUNCH_APP
echo [INFO] Launching application...
call "%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%main.py" %APP_ARGS%

endlocal
exit /b 0

REM --- Subroutines ---

:info
echo [INFO] %*
goto :eof

:ok
echo [OK] %*
goto :eof

:fail
echo [FAIL] %*
goto :eof

:add_error
set /a ERROR_COUNT+=1
set "SUMMARY_ERRORS=!SUMMARY_ERRORS! - %~1^

"
goto :eof

:usage
echo Usage: start-windows.bat [options]
echo.
echo Options:
echo   --check-only       Run checks only; do not modify system or install
echo   --recreate-venv    Recreate .venv from scratch
echo   --with-extras      Install extras (auto-detect NVIDIA GPU; Windows/Linux)
echo   --clean            Pass --clean to the application on launch
echo   -h, --help         Show this help
goto :eof

:check_python
call :info "Checking Python version"
REM Prefer Python launcher (py), fallback to python
set "PY_CMD="
where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    call :fail "Python not found in PATH. Install Python 3.11+."
    echo Suggested fix: Install from python.org or run 'winget install Python.Python.3.11'
    echo Also ensure Settings ^> Apps ^> App execution aliases: disable the Microsoft Store 'python' alias.
    call :add_error "Install Python 3.11+ (e.g., 'winget install Python.Python.3.11')"
    exit /b 1
)

REM Query version string "MAJOR.MINOR" robustly (avoid %% formatting issues)
set "PY_VER="
for /f %%v in ('%PY_CMD% -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))" 2^>nul') do set "PY_VER=%%v"

if not defined PY_VER (
    call :fail "Unable to execute Python interpreter via %PY_CMD%."
    echo Suggested fix: Ensure a real Python installation is present and disable the Microsoft Store alias.
    call :add_error "Fix Python PATH / install Python 3.11+"
    exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if %PY_MAJOR% lss %PY_REQUIRED_MAJOR% (
    set "VERSION_OK=0"
) else if %PY_MAJOR% equ %PY_REQUIRED_MAJOR% (
    if %PY_MINOR% lss %PY_REQUIRED_MINOR% (
        set "VERSION_OK=0"
    ) else (
        set "VERSION_OK=1"
    )
) else (
    set "VERSION_OK=1"
)

if "%VERSION_OK%"=="0" (
    call :fail "Python %PY_MAJOR%.%PY_MINOR% found, need >= %PY_REQUIRED_MAJOR%.%PY_REQUIRED_MINOR%"
    echo Suggested fix: Upgrade Python. 'winget install Python.Python.3.11'
    call :add_error "Upgrade Python to >= %PY_REQUIRED_MAJOR%.%PY_REQUIRED_MINOR%"
    exit /b 1
)
call :ok "Python %PY_MAJOR%.%PY_MINOR% (%PY_CMD%)"
exit /b 0

:ensure_venv
call :info "Ensuring virtual environment at %VENV_DIR%"
if "%FLAG_RECREATE_VENV%"=="true" (
    if exist "%VENV_DIR%\" (
        if "%FLAG_CHECK_ONLY%"=="true" (
            call :fail "--recreate-venv requested during --check-only"
            call :add_error "Rerun without --check-only to recreate venv"
            exit /b 1
        )
        call :info "Recreating venv..."
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_DIR%\" (
    if "%FLAG_CHECK_ONLY%"=="true" (
        call :fail "venv missing at %VENV_DIR%"
        echo Suggested fix: %PY_CMD% -m venv %VENV_DIR%
        call :add_error "Create venv: %PY_CMD% -m venv %VENV_DIR%"
        exit /b 1
    )
    call :info "Creating venv..."
    %PY_CMD% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        call :fail "Failed to create virtual environment."
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
call :ok "Venv active"
call :info "Upgrading pip..."
"%VENV_PY%" -m pip install --upgrade pip >nul
exit /b 0

:ensure_core_requirements
call :info "Checking core Python dependencies"
if "%FLAG_CHECK_ONLY%"=="true" (
    "%VENV_PY%" -c "import sys; missing = []; modules = ['PySide6', 'sounddevice', 'numpy', 'cffi', 'pyperclip']; [missing.append(m) for m in modules if not __import__(m, fromlist=['__name__'])]; sys.exit(1) if missing else sys.exit(0)"
    if %errorlevel% equ 0 (
        call :ok "Core deps present"
    ) else (
        call :fail "Core deps missing"
        echo Suggested fix: pip install -r requirements.txt
        call :add_error "Install core deps: pip install -r requirements.txt"
        exit /b 1
    )
) else (
    call :info "Installing/updating core dependencies from requirements.txt..."
    "%VENV_PY%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    if %errorlevel% neq 0 (
        call :fail "Failed to install core requirements."
        call :add_error "Rerun pip install -r requirements.txt"
        exit /b 1
    )
    call :ok "Core dependencies installed."
)
exit /b 0

:verify_runtime
call :info "Verifying PySide6 and sounddevice runtime"
"%VENV_PY%" -c "import sys, traceback; print('Checking PySide6...'); import PySide6, PySide6.QtWidgets; print('Checking sounddevice...'); import sounddevice as sd; sd.query_devices(); print('OK')" >nul 2>nul
if %errorlevel% equ 0 (
    call :ok "Runtime OK"
) else (
    call :fail "Runtime check failed. PySide6 or sounddevice may have issues."
    echo Suggested fix: Reinstall dependencies.
    echo   pip install --force-reinstall -r requirements.txt
    call :add_error "Fix runtime: pip install --force-reinstall -r requirements.txt"
    exit /b 1
)
exit /b 0

:install_extras_auto
call :info "Installing extras (auto GPU detect)"
REM Detect NVIDIA GPU via nvidia-smi
set "GPU_AVAILABLE=0"
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    nvidia-smi -L >nul 2>nul
    if %errorlevel% equ 0 (
        set "GPU_AVAILABLE=1"
    )
)

REM In check-only mode, just report what would be installed and exit
if "%FLAG_CHECK_ONLY%"=="true" (
    if "%GPU_AVAILABLE%"=="1" (
        call :info "NVIDIA GPU detected (check-only). Would install: torch==2.5.1+cu121, torchaudio==2.5.1+cu121 from cu121 index"
        call :info "Then would install remaining extras (keeping onnxruntime-gpu), excluding torch/torchaudio lines"
    ) else (
        call :info "No NVIDIA GPU detected (check-only). Would install: torch==2.5.* and torchaudio==2.5.* from CPU index"
        call :info "Then would install remaining extras excluding torch/torchaudio and onnxruntime-gpu (CPU-safe)"
    )
    exit /b 0
)

if "%GPU_AVAILABLE%"=="1" (
    call :info "NVIDIA GPU detected; installing CUDA 12.1 wheels for torch/torchaudio"
    "%VENV_PY%" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchaudio==2.5.1+cu121
    if %errorlevel% neq 0 (
        call :fail "CUDA wheel install failed; falling back to CPU wheels"
        "%VENV_PY%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.* torchaudio==2.5.*
    )
    REM Install the rest, excluding torch/torchaudio lines (onnxruntime-gpu kept for GPU systems)
    set "FILTERED_REQ=%TEMP%\node_audio_additional_no_torch.txt"
    findstr /V /C:"torch==" /C:"torchaudio==" "%PROJECT_DIR%additional_requirements.txt" > "%FILTERED_REQ%"
    call :info "Installing other extras from additional_requirements.txt (excluding torch/torchaudio)"
    pip install -r "%FILTERED_REQ%" 1>nul
) else (
    call :info "No NVIDIA GPU detected; installing CPU wheels for torch/torchaudio"
    "%VENV_PY%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.* torchaudio==2.5.*
    REM CPU-safe: also exclude onnxruntime-gpu from extras
    set "FILTERED_REQ=%TEMP%\node_audio_additional_cpu_safe.txt"
    findstr /V /C:"torch==" /C:"torchaudio==" /C:"onnxruntime-gpu==" "%PROJECT_DIR%additional_requirements.txt" > "%FILTERED_REQ%"
    call :info "Installing other extras from additional_requirements.txt (excluding torch/torchaudio and onnxruntime-gpu)"
    "%VENV_PY%" -m pip install -r "%FILTERED_REQ%" 1>nul
)

call :ok "Extras install attempted"
exit /b 0

:auto_fix_flow
if "%FLAG_CHECK_ONLY%"=="true" (
    call :summary_and_exit
)
echo.
echo [FAIL] Some checks failed.
set "CHOICE="
set /p CHOICE=Attempt automatic install/fix now? [Y/N]: 
if /i "!CHOICE!"=="Y" (
    REM Attempt to fix issues
    if defined PYTHON_CHECK_FAILED (
        echo [INFO] Attempting to install Python 3.11 via winget...
        where winget >nul 2>nul
        if %errorlevel% equ 0 (
            winget install -e --id Python.Python.3.11 -s winget
        ) else (
            echo [INFO] winget not available. Please install Python manually from https://www.python.org/downloads/windows/
        )
    )
    REM Re-run venv creation and installs
    call :ensure_venv || set "VENV_FAILED=1"
    call :ensure_core_requirements || set "CORE_REQ_FAILED=1"
    if defined RUNTIME_FAILED (
        echo [INFO] Reinstalling core requirements to attempt to fix runtime...
        "%VENV_PY%" -m pip install --force-reinstall -r "%PROJECT_DIR%requirements.txt"
    )
    if "%FLAG_WITH_EXTRAS%"=="true" (
        call :install_extras_auto || set "EXTRAS_FAILED=1"
    )
    REM Reset errors and re-run checks
    set "ERROR_COUNT=0"
    set "SUMMARY_ERRORS="
    set "PYTHON_CHECK_FAILED="
    set "VENV_FAILED="
    set "CORE_REQ_FAILED="
    set "RUNTIME_FAILED="
    call :check_python || set "PYTHON_CHECK_FAILED=1"
    call :ensure_venv || set "VENV_FAILED=1"
    call :ensure_core_requirements || set "CORE_REQ_FAILED=1"
    call :verify_runtime || set "RUNTIME_FAILED=1"
    if %ERROR_COUNT% gtr 0 (
        call :summary_and_exit
    ) else (
        goto :LAUNCH_APP
    )
) else (
    echo.
    echo [INFO] Copy-paste these commands to fix the issues:
    echo.!SUMMARY_ERRORS!
    call :summary_and_exit
)

:summary_and_exit
if %ERROR_COUNT% gtr 0 (
    echo.
    call :fail "Some checks failed. Suggested fixes:"
    echo.!SUMMARY_ERRORS!
    exit /b 1
) else (
    call :ok "All checks passed"
    exit /b 0
)
goto
