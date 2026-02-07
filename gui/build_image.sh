#!/usr/bin/env bash
# Build the container image for running chat_htm with the Qt6 GUI.
#
# This reuses htm_flow's Containerfile (same base: Debian trixie + Qt6 + cmake).
# If htm_flow already built the image you can skip this step.
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-chat_htm_gui:qt6}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONTAINERFILE="$REPO_ROOT/htm_flow/htm_gui/Containerfile"

if [[ ! -f "$CONTAINERFILE" ]]; then
  echo "Containerfile not found at $CONTAINERFILE" >&2
  echo "Make sure the htm_flow submodule is initialised:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

podman build -f "$CONTAINERFILE" -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "Built image: $IMAGE_NAME"
