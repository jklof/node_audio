#!/usr/bin/env bash

set -euo pipefail

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--system|--user] [--dry-run] [--codesign]

Build the VST3 plugins and install them to the standard macOS VST3 folder.

Options:
  --system   Install to /Library/Audio/Plug-Ins/VST3 (requires sudo if not writable)
  --user     Install to ~/Library/Audio/Plug-Ins/VST3 (default)
  --dry-run  Show actions without copying files
  --codesign Ad-hoc codesign the bundles (helps some DAWs on macOS)
  -h, --help Show this help and exit
EOF
}

DEST_USER="$HOME/Library/Audio/Plug-Ins/VST3"
DEST_SYSTEM="/Library/Audio/Plug-Ins/VST3"
DEST="$DEST_USER"
DRY_RUN=0
CODESIGN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      DEST="$DEST_SYSTEM"
      shift
      ;;
    --user)
      DEST="$DEST_USER"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --codesign)
      CODESIGN=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building and bundling VST3 plugins in workspace at: $SCRIPT_DIR"

if [[ "$DRY_RUN" -eq 0 ]]; then
c  cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release
  cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release
else
  echo "[dry-run] cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release" >&2
  echo "[dry-run] cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release" >&2
fi

TARGET_DIR="$SCRIPT_DIR/target/bundled"

# Determine bundle paths (directories ending in .vst3) produced by nih-plug bundler
find_bundle() {
  local pattern="$1"
  local path
  path="${TARGET_DIR}/${pattern}.vst3"
  if [[ -d "$path" ]]; then
    echo "$path"
    return 0
  fi
  # Fallback: search by prefix in case of variant suffixes
  path=$(find "$TARGET_DIR" -maxdepth 1 -type d -name "${pattern}*.vst3" | head -n1 || true)
  if [[ -n "${path}" && -d "${path}" ]]; then
    echo "$path"
    return 0
  fi
  return 1
}

FX_BUNDLE=$(find_bundle "node_audio_vst_fx" || true)
INST_BUNDLE=$(find_bundle "node_audio_vst_instrument" || true)

# Fallback: search by plugin display names if bundler uses them
if [[ -z "$FX_BUNDLE" ]]; then
  FX_BUNDLE=$(find "$TARGET_DIR" -maxdepth 1 -type d -name "*NodeAudio*FX*.vst3" | head -n1 || true)
fi
if [[ -z "$INST_BUNDLE" ]]; then
  INST_BUNDLE=$(find "$TARGET_DIR" -maxdepth 1 -type d -name "*NodeAudio*Instrument*.vst3" | head -n1 || true)
fi

if [[ -z "$FX_BUNDLE" || ! -d "$FX_BUNDLE" ]]; then
  echo "Error: Could not find bundled FX plugin in $TARGET_DIR" >&2
  exit 1
fi
if [[ -z "$INST_BUNDLE" || ! -d "$INST_BUNDLE" ]]; then
  echo "Error: Could not find bundled Instrument plugin in $TARGET_DIR" >&2
  exit 1
fi

echo "Found bundled plugins:"
echo "  FX:         $FX_BUNDLE"
echo "  Instrument: $INST_BUNDLE"

echo "Installing to: $DEST"

SUDO=""
if [[ "$DEST" == "$DEST_SYSTEM" && ! -w "$DEST_SYSTEM" ]]; then
  SUDO="sudo"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  $SUDO mkdir -p "$DEST"
else
  echo "[dry-run] mkdir -p \"$DEST\""
fi

install_bundle() {
  local bundle_path="$1"
  local dest_dir="$2"
  local name
  name="$(basename "$bundle_path")"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    $SUDO rm -rf "$dest_dir/$name"
    $SUDO cp -R "$bundle_path" "$dest_dir/"
  else
    echo "[dry-run] rm -rf \"$dest_dir/$name\""
    echo "[dry-run] cp -R \"$bundle_path\" \"$dest_dir/\""
  fi
}

install_bundle "$FX_BUNDLE" "$DEST"
install_bundle "$INST_BUNDLE" "$DEST"

echo "Done. If your DAW is open, rescan plugins or restart it."


