#!/usr/bin/env bash
# Run chat_htm with the Qt6 GUI debugger inside a container.
#
# Usage:
#   ./gui/run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml
#   ./gui/run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml --theme dark
#   ./gui/run_gui.sh --input myfile.txt --config configs/default_text.yaml --log
#   ./gui/run_gui.sh --help
#
# The container mounts the entire chat_htm repo at /work/chat_htm, builds with
# HTM_FLOW_WITH_GUI=ON, then runs the chat_htm binary with --gui plus any args
# you pass to this script.
#
# Prerequisites:
#   1. Build the image first:  ./gui/build_image.sh
#   2. Or reuse htm_flow's image: IMAGE_NAME=htm_flow_gui:qt6 ./gui/run_gui.sh ...
set -euo pipefail

# Handle --help locally for quick reference
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run chat_htm with the Qt6 GUI debugger in a container.

Usage: ./gui/run_gui.sh [chat_htm options]

All arguments are forwarded to the chat_htm binary (--gui is added automatically).
Paths to --input and --config should be relative to the chat_htm project root.
Theme can be selected with --theme light|dark (or YAML gui.theme in the config file).

Examples:
  ./gui/run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml
  ./gui/run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml --theme dark
  ./gui/run_gui.sh --input tests/test_data/alphabet.txt --config configs/default_text.yaml --log
  ./gui/run_gui.sh --input mydata.txt --config configs/default_text.yaml --epochs 10

Building the image:
  ./gui/build_image.sh

Environment:
  IMAGE_NAME   Container image name (default: chat_htm_gui:qt6)
EOF
  exit 0
fi

IMAGE_NAME=${IMAGE_NAME:-chat_htm_gui:qt6}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Auto-build the image if it doesn't exist yet
if ! podman image exists "$IMAGE_NAME" 2>/dev/null; then
  # Also check if htm_flow's image exists and reuse it
  if podman image exists "htm_flow_gui:qt6" 2>/dev/null; then
    echo "Reusing existing htm_flow_gui:qt6 image."
    IMAGE_NAME="htm_flow_gui:qt6"
  else
    echo "Container image not found. Building '$IMAGE_NAME' (first time only)..."
    "$SCRIPT_DIR/build_image.sh"
  fi
fi

UID_NUM=$(id -u)

COMMON_ARGS=(
  --rm
  --userns=keep-id
  -v "$REPO_ROOT:/work/chat_htm:Z"
  -w /work/chat_htm
  -e QT_X11_NO_MITSHM=1
)

DISPLAY_ARGS=()

if [[ -n "${WAYLAND_DISPLAY-}" && -d "/run/user/$UID_NUM" ]]; then
  DISPLAY_ARGS+=(
    -e WAYLAND_DISPLAY="$WAYLAND_DISPLAY"
    -e XDG_RUNTIME_DIR="/run/user/$UID_NUM"
    -v "/run/user/$UID_NUM:/run/user/$UID_NUM:Z"
    -e QT_QPA_PLATFORM=wayland
  )
elif [[ -n "${DISPLAY-}" && -d /tmp/.X11-unix ]]; then
  DISPLAY_ARGS+=(
    -e DISPLAY="$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix:Z
    -e QT_QPA_PLATFORM=xcb
  )
else
  echo "No WAYLAND_DISPLAY or DISPLAY detected; cannot show GUI." >&2
  exit 1
fi

# Optional GPU accel
if [[ -d /dev/dri ]]; then
  DISPLAY_ARGS+=(--device /dev/dri)
fi

# Forward all script arguments to chat_htm (--gui is added automatically)
CHAT_HTM_ARGS="${*:-}"

podman run "${COMMON_ARGS[@]}" "${DISPLAY_ARGS[@]}" "$IMAGE_NAME" bash -lc "
  set -euo pipefail

  # Ensure htm_flow dependencies are present
  cd htm_flow
  if [[ ! -d include/taskflow ]]; then
    echo 'Taskflow headers not found; running htm_flow/setup.sh'
    ./setup.sh
  fi
  cd ..

  # Build chat_htm with GUI support
  cmake -S . -B build_qt -DHTM_FLOW_WITH_GUI=ON -DCHAT_HTM_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release 2>/dev/null
  cmake --build build_qt -j 2>/dev/null

  echo '--- Launching chat_htm GUI ---'
  ./build_qt/chat_htm $CHAT_HTM_ARGS --gui
"
