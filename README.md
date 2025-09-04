# node_audio

Node-based realtime audio processing.

## Quick start

Below are two supported development setups. Pick one: Conda (recommended) or Python venv. Both will isolate dependencies and let you run the app.

### Prerequisites

- Python 3.11
- Git
- Recommended on macOS: Homebrew (`brew`)

### System libraries (audio)

Some platforms need PortAudio (for `sounddevice`) and optionally libsndfile (for `soundfile`, used by optional plugins):

- macOS: `brew install portaudio libsndfile`
- Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y libportaudio2 libsndfile1`
- Windows: none required (wheels include binaries)

## Option A: Conda (recommended)

```bash
# Create and activate environment
conda create -n node-audio python=3.11 -y
conda activate node-audio

# (macOS/Linux) Ensure audio libs are available in this env
conda install -c conda-forge portaudio libsndfile -y

# Install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional: install extra features (plugins under `additional_plugins/`).

- NVIDIA GPU (Linux/Windows):

```bash
pip install -r additional_requirements.txt
```

- macOS or CPU-only (no CUDA): install PyTorch/torchaudio appropriate for your platform and the rest of the extras without the CUDA-pinned wheels:

```bash
# Install CPU builds of PyTorch/torchaudio
pip install "torch==2.5.*" "torchaudio==2.5.*"

# Install remaining extras excluding torch/torchaudio lines
grep -v -E '^(torch|torchaudio)=' additional_requirements.txt | pip install -r /dev/stdin
```

## Option B: Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# Install core deps
pip install -r requirements.txt

# Optional: extras (see the same platform notes as in Option A)
# NVIDIA GPU (Linux/Windows):
# pip install -r additional_requirements.txt

# macOS or CPU-only:
# pip install "torch==2.5.*" "torchaudio==2.5.*"
# grep -v -E '^(torch|torchaudio)=' additional_requirements.txt | pip install -r /dev/stdin
```

## Run the app

```bash
python main.py
```

Notes:

- On macOS, the first run may prompt for microphone permissions. Grant access so audio I/O works.
- If you installed only `requirements.txt`, you get the core UI and audio graph engine. Features in `additional_plugins/` (e.g., RVC, YouTube, certain spectral effects) require the optional installs.

## Automated setup and checks (macOS)

Use `start.sh` to automatically verify and set up the environment, and then start the app.

What it checks/does:

- Python version is 3.11+ and a project venv exists and is active
- Homebrew is available
- Audio libraries installed: `portaudio` (required), `libsndfile` (optional but recommended)
- Core Python deps installed from `requirements.txt`
- Verifies `PySide6` and `sounddevice` imports and enumerates audio devices
- Optionally installs extras from `additional_requirements.txt` (CPU-safe subset on macOS)

Usage:

```bash
# Make executable once
chmod +x start.sh

# 1) Dry-run checks only (no changes). Prints copy-paste fixes if anything fails.
./start.sh --check-only

# 2) Auto-setup and start the app (installs any missing pieces)
./start.sh

# Options:
#   --check-only         Do not modify system; only report status and fixes
#   --recreate-venv      Recreate .venv from scratch
#   --with-extras        Install CPU-safe extras from additional_requirements.txt
#   --clean              Pass --clean to the app on launch
```

Expected outcome:

- A green summary with “OK” for each check. If any step fails in `--check-only`, the script prints explicit commands you can copy and paste to fix the issue.
- When all checks pass (and not in `--check-only`), the app launches.

## Why two requirements files?

- `requirements.txt`: Minimal, cross-platform core runtime (GUI via `PySide6`, audio I/O via `sounddevice`, etc.).
- `additional_requirements.txt`: Optional, heavier dependencies for extra plugins (e.g., PyTorch, ONNX, YouTube). Some entries are CUDA-specific (e.g., `torch==2.5.1+cu121`), which are not available on macOS. Use the platform notes above.

## Troubleshooting

- PortAudio errors (e.g., "PortAudio library not found"):
  - macOS: `brew install portaudio`
  - Conda: `conda install -c conda-forge portaudio`

- Qt platform plugin errors (`xcb`, `cocoa`):
  - Reinstall `PySide6`: `pip install --force-reinstall PySide6==6.9.2`

- PyTorch install conflicts on macOS:
  - Do not install CUDA-pinned wheels from `additional_requirements.txt`.
  - Install CPU/MPS builds as shown above, then install the rest of the extras excluding `torch`/`torchaudio` lines.

## Development tips

- Keep your env active while working (`conda activate node-audio` or `source .venv/bin/activate`).
- To update dependencies, adjust the appropriate requirements file(s) and reinstall.
- To reset the env quickly:

```bash
pip freeze > /tmp/old.txt && xargs -a /tmp/old.txt -r pip uninstall -y
pip install -r requirements.txt
```
he you ennallyt s