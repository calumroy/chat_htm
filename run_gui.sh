#!/usr/bin/env bash
# Shortcut to run chat_htm with the Qt6 GUI in a container.
# No local Qt6 installation needed -- everything runs inside podman.
#
# Usage:
#   ./run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml
#   ./run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml --theme dark
#   ./run_gui.sh --input tests/test_data/alphabet.txt --config configs/default_text.yaml --log
#   ./run_gui.sh -h
#
# The container image is built automatically on first run.
exec "$(dirname "$0")/gui/run_gui.sh" "$@"
