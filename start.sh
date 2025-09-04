#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PY_REQUIRED_MAJOR=3
PY_REQUIRED_MINOR=11

FLAG_CHECK_ONLY=false
FLAG_RECREATE_VENV=false
FLAG_WITH_EXTRAS=false
APP_ARGS=()

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_RESET='\033[0m'

function info() { echo -e "${COLOR_YELLOW}[INFO]${COLOR_RESET} $*"; }
function ok() { echo -e "${COLOR_GREEN}[OK]${COLOR_RESET} $*"; }
function fail() { echo -e "${COLOR_RED}[FAIL]${COLOR_RESET} $*"; }

function usage() {
  cat <<USAGE
Usage: ./start.sh [options]

Options:
  --check-only       Run checks only; do not modify system or install
  --recreate-venv    Recreate .venv from scratch
  --with-extras      Install CPU-safe extras from additional_requirements.txt
  --clean            Pass --clean to the application on launch
  -h, --help         Show this help
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --check-only) FLAG_CHECK_ONLY=true ;;
    --recreate-venv) FLAG_RECREATE_VENV=true ;;
    --with-extras) FLAG_WITH_EXTRAS=true ;;
    --clean) APP_ARGS+=("--clean") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 2 ;;
  esac
done

SUMMARY_ERRORS=()

function add_error() {
  SUMMARY_ERRORS+=("$1")
}

function check_python() {
  info "Checking Python version"
  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found. Install Python 3.11+."
    echo "Suggested fix: brew install python@3.11"
    add_error "Install Python 3.11 (brew install python@3.11)"
    return 1
  fi
  local ver
  ver=$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')
  local major minor
  major=${ver%%.*}
  minor=${ver##*.}
  if [[ $major -lt $PY_REQUIRED_MAJOR || ( $major -eq $PY_REQUIRED_MAJOR && $minor -lt $PY_REQUIRED_MINOR ) ]]; then
    fail "Python $ver found, need >= ${PY_REQUIRED_MAJOR}.${PY_REQUIRED_MINOR}"
    echo "Suggested fix: brew install python@3.11 && brew link python@3.11"
    add_error "Upgrade Python to >= ${PY_REQUIRED_MAJOR}.${PY_REQUIRED_MINOR}"
    return 1
  fi
  ok "Python $ver"
}

function ensure_venv() {
  info "Ensuring virtual environment at $VENV_DIR"
  if $FLAG_RECREATE_VENV && [[ -d "$VENV_DIR" ]]; then
    if $FLAG_CHECK_ONLY; then
      fail "--recreate-venv requested during --check-only"
      add_error "Rerun without --check-only to recreate venv"
      return 1
    fi
    rm -rf "$VENV_DIR"
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    if $FLAG_CHECK_ONLY; then
      fail "venv missing at $VENV_DIR"
      echo "Suggested fix: python3 -m venv $VENV_DIR"
      add_error "Create venv: python3 -m venv $VENV_DIR"
      return 1
    fi
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  ok "Venv active: $(command -v python)"
  python -m pip install --upgrade pip >/dev/null
}

function check_homebrew() {
  info "Checking Homebrew"
  if ! command -v brew >/dev/null 2>&1; then
    fail "Homebrew not found"
    echo "Suggested fix: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    add_error "Install Homebrew"
    return 1
  fi
  ok "Homebrew present"
}

function ensure_audio_libs() {
  info "Checking PortAudio (required) and libsndfile (optional)"
  local need_install=false
  if ! brew list --versions portaudio >/dev/null; then
    if $FLAG_CHECK_ONLY; then
      fail "portaudio not installed"
      echo "Suggested fix: brew install portaudio"
      add_error "Install portaudio: brew install portaudio"
    else
      brew install portaudio >/dev/null
    fi
    need_install=true
  fi
  if ! brew list --versions libsndfile >/dev/null; then
    if $FLAG_CHECK_ONLY; then
      info "libsndfile not installed (optional)"
    else
      brew install libsndfile >/dev/null || true
    fi
  fi
  if [[ "$need_install" == false ]]; then
    ok "Audio libs OK"
  fi
}

function ensure_core_requirements() {
  info "Checking core Python dependencies"
  if $FLAG_CHECK_ONLY; then
    # Try importing key modules to infer install status
    if python - <<'PY'
import sys
missing = []
for m in ["PySide6", "sounddevice", "numpy", "cffi", "pyperclip"]:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print("MISSING:", " ".join(missing))
    sys.exit(1)
print("OK")
PY
    then
      ok "Core deps present"
    else
      fail "Core deps missing"
      echo "Suggested fix: pip install -r requirements.txt"
      add_error "Install core deps: pip install -r requirements.txt"
      return 1
    fi
  else
    pip install -r "$PROJECT_DIR/requirements.txt"
  fi
}

function verify_runtime() {
  info "Verifying PySide6 and sounddevice runtime"
  if ! python - <<'PY'
import sys
import traceback
try:
  import PySide6
  from PySide6 import QtWidgets
except Exception as e:
  print("PySide6 import error:", e)
  traceback.print_exc()
  raise SystemExit(2)
try:
  import sounddevice as sd
  sd.query_devices()
except Exception as e:
  print("sounddevice error:", e)
  traceback.print_exc()
  raise SystemExit(3)
print("OK")
PY
  then
    fail "Runtime check failed"
    echo "Suggested fix: Ensure PortAudio installed and reinstall deps"
    echo "  brew install portaudio"
    echo "  pip install --force-reinstall -r requirements.txt"
    add_error "Fix runtime: portaudio + reinstall deps"
    return 1
  fi
  ok "Runtime OK"
}

function install_extras_cpu_safe() {
  info "Installing CPU-safe extras (macOS)"
  if $FLAG_CHECK_ONLY; then
    info "Skipping install due to --check-only"
    return 0
  fi
  # Install CPU builds of torch/torchaudio and the rest excluding CUDA-pinned wheels
  pip install "torch==2.5.*" "torchaudio==2.5.*" || true
  # shellcheck disable=SC2002
  grep -v -E '^(torch|torchaudio)=' "$PROJECT_DIR/additional_requirements.txt" | pip install -r /dev/stdin || true
  ok "Extras install attempted"
}

function summary_and_exit() {
  if (( ${#SUMMARY_ERRORS[@]} > 0 )); then
    echo
    fail "Some checks failed. Copy-paste these fixes:" 
    for e in "${SUMMARY_ERRORS[@]}"; do
      echo " - $e"
    done
    exit 1
  else
    ok "All checks passed"
    exit 0
  fi
}

# Run checks/setup
check_python || true
check_homebrew || true
ensure_venv || true
ensure_audio_libs || true
ensure_core_requirements || true
verify_runtime || true

if $FLAG_WITH_EXTRAS; then
  install_extras_cpu_safe || true
fi

if $FLAG_CHECK_ONLY; then
  summary_and_exit
fi

if (( ${#SUMMARY_ERRORS[@]} > 0 )); then
  summary_and_exit
fi

# Launch app
info "Launching application"
# Handle empty APP_ARGS safely under set -u
if (( ${#APP_ARGS[@]:-0} > 0 )); then
  exec python "$PROJECT_DIR/main.py" "${APP_ARGS[@]}"
else
  exec python "$PROJECT_DIR/main.py"
fi


