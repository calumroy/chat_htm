#!/usr/bin/env bash
set -euo pipefail

# Build script for chat_htm.
#
# Defaults: Release build, tests ON, GUI OFF.
# Run with no arguments for a quick release build.
#
# Usage:
#   ./build.sh                     # Release build (default)
#   ./build.sh Debug               # Debug build
#   ./build.sh Release GUI         # Release with Qt6 GUI
#   ./build.sh Debug GUI           # Debug with Qt6 GUI
#   ./build.sh RelWithDebInfo      # Optimised + debug symbols
#   ./build.sh clean               # Delete the build directory
#   ./build.sh -h                  # Show help

display_help() {
  cat <<'EOF'
Usage: ./build.sh [BUILD_TYPE] [OPTIONS...]

BUILD_TYPE (first positional arg, default: Release):
  Release          Optimised build
  Debug            Debug symbols, no optimisation
  RelWithDebInfo   Optimised with debug symbols

OPTIONS (case-insensitive, any order after BUILD_TYPE):
  GUI              Enable the Qt6 GUI debugger (requires Qt6)
  NOTESTS          Skip building the test suite
  CLEAN            Same as passing "clean" as the first argument

COMMANDS:
  clean            Delete the build/ directory and exit
  -h, --help       Show this help and exit

Examples:
  ./build.sh                          # Quick release build
  ./build.sh Debug                    # Debug build
  ./build.sh Release GUI              # Release + GUI
  ./build.sh Debug GUI NOTESTS        # Debug + GUI, skip tests
  ./build.sh clean                    # Wipe build dir
EOF
  exit 0
}

# --- Parse arguments ---

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  display_help
fi

if [[ "${1:-}" == "clean" ]]; then
  echo "Removing build/ directory..."
  rm -rf build
  echo "Done."
  exit 0
fi

# First arg = build type (if it looks like one), otherwise default to Release
BUILD_TYPE="Release"
START_IDX=1
case "${1:-}" in
  Release|Debug|RelWithDebInfo)
    BUILD_TYPE="$1"
    START_IDX=2
    ;;
esac

# Remaining args = options
GUI_FLAG="-DHTM_FLOW_WITH_GUI=OFF"
TESTS_FLAG="-DCHAT_HTM_BUILD_TESTS=ON"

for arg in "${@:$START_IDX}"; do
  lower=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    gui)     GUI_FLAG="-DHTM_FLOW_WITH_GUI=ON" ;;
    notests) TESTS_FLAG="-DCHAT_HTM_BUILD_TESTS=OFF" ;;
    clean)   rm -rf build; echo "Removed build/"; exit 0 ;;
    *)       echo "Unknown option: $arg"; display_help ;;
  esac
done

# --- Ensure htm_flow dependencies ---

if [[ ! -d htm_flow/include/taskflow ]]; then
  echo "htm_flow dependencies missing; running htm_flow/setup.sh..."
  (cd htm_flow && ./setup.sh)
fi

# --- Build ---

mkdir -p build
cd build

echo "Build type:  $BUILD_TYPE"
echo "GUI:         ${GUI_FLAG##*=}"
echo "Tests:       ${TESTS_FLAG##*=}"
echo ""

if ! cmake .. \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  "$GUI_FLAG" \
  "$TESTS_FLAG" \
  -DBUILD_TESTS=OFF 2>&1; then

  # Check if this was a Qt6 / GUI failure
  if [[ "$GUI_FLAG" == *"ON"* ]]; then
    echo ""
    echo "============================================================"
    echo "  CMAKE FAILED -- likely because Qt6 is not installed."
    echo ""
    echo "  You do NOT need Qt6 locally to use the GUI."
    echo "  Use the container-based GUI instead:"
    echo ""
    echo "    ./run_gui.sh --input tests/test_data/hello_world.txt \\"
    echo "                 --config configs/small_text.yaml"
    echo ""
    echo "  Or build without GUI:  ./build.sh"
    echo "============================================================"
  fi
  exit 1
fi

cmake --build . -j"$(nproc)"

echo ""
echo "Build complete.  Binary: build/chat_htm"
if [[ "$TESTS_FLAG" == *"ON"* ]]; then
  echo "Run tests:   cd build && ctest --output-on-failure"
fi
if [[ "$GUI_FLAG" == *"OFF"* ]]; then
  echo "GUI:         Use ./run_gui.sh for the visual debugger (no Qt6 needed)"
fi
