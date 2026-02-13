# Chat HTM

Teach an HTM (Hierarchical Temporal Memory) network to learn and predict patterns in simple text.

## Overview

This project uses [htm_flow](https://github.com/calumroy/htm_flow) as a library to build HTM-based models that learn temporal patterns in text. Characters are encoded into Sparse Distributed Representations (SDRs) and fed to a configurable multi-layer HTM network, which learns to predict the next character in a sequence.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed explanation of how everything works.

## Setup

1. Clone the repository with submodules:
```bash
git clone --recurse-submodules <repo-url>
cd chat_htm
```

Or if already cloned:
```bash
git submodule update --init --recursive
```

2. Build:
```bash
./build.sh
```

That handles dependencies, cmake, and compilation in one step. See `./build.sh -h` for options (e.g. `./build.sh Debug`, `./build.sh Release GUI`).

## Usage

All examples assume you are in the project root (not `build/`).

```bash
# Feed a text file to the HTM network
./build/chat_htm --input tests/test_data/hello_world.txt --config configs/small_text.yaml

# With logging to see progress and prediction accuracy
./build/chat_htm --input tests/test_data/hello_world.txt --config configs/default_text.yaml --epochs 10 --log

# Simple repeating pattern (good for seeing the HTM learn quickly)
./build/chat_htm --input tests/test_data/simple_patterns.txt --config configs/small_text.yaml --epochs 5 --log

# Word-row mode: one word per step, one row per letter position
./build/chat_htm --input tests/test_data/simple_sentences.txt --config configs/word_rows_text.yaml --epochs 20 --log

# Process a specific number of steps
./build/chat_htm --input tests/test_data/alphabet.txt --config configs/small_text.yaml --steps 500

# List available configs
./build/chat_htm --list-configs
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--input FILE` | Path to a text file (required) |
| `--config FILE` | Path to a YAML config file (required) |
| `--steps N` | Number of character steps (default: entire file) |
| `--epochs N` | Number of passes through the file (default: 1) |
| `--gui` | Launch the htm_gui visual debugger |
| `--theme MODE` | GUI theme: `light` or `dark` (overrides YAML `gui.theme`) |
| `--log` | Print per-step progress and accuracy |
| `--list-configs` | List YAML configs in `configs/` |

## Configuration

YAML config files in `configs/` control both the text encoding and HTM network parameters:

- **`configs/small_text.yaml`** -- Single-layer, 100-bit SDR. Fast for testing.
- **`configs/default_text.yaml`** -- Two-layer, 400-bit SDR. Better capacity.
- **`configs/word_rows_text.yaml`** -- Single-layer word mode. One word per HTM step and one input row per letter position with non-overlapping per-letter bit blocks.

You can create your own configs to experiment with different network sizes, numbers of layers, learning rates, and temporal pooling settings. See the existing configs and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details on all parameters.

## Running Tests

```bash
cd build && ctest --output-on-failure
```

## GUI Debugger

The htm_gui Qt6 debugger lets you visualize the HTM network as it processes text -- you can watch column activations, cell predictions, and synapses update in real time.

**You do not need Qt6 installed.** The GUI runs inside a container (podman). The image is built automatically on first use:

```bash
# Launch the GUI (image builds automatically the first time)
./run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml

# Force dark theme from CLI
./run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml --theme dark

# With logging
./run_gui.sh --input tests/test_data/alphabet.txt --config configs/default_text.yaml --log

# Word-row mode in GUI
./run_gui.sh --input tests/test_data/simple_sentences.txt --config configs/word_rows_text.yaml --epochs 20 --log

# Your own text file
./run_gui.sh --input mydata.txt --config configs/default_text.yaml --epochs 10
```

If you have Qt6 installed locally and prefer a native build:
```bash
./build.sh Release GUI
./build/chat_htm --input tests/test_data/hello_world.txt --config configs/default_text.yaml --gui
```

Theme can also be set in YAML and will be used as the default when `--theme` is not provided:

```yaml
gui:
  theme: dark
```

## Project Structure

```
chat_htm/
├── htm_flow/          HTM library (git submodule)
├── src/               Source code
│   ├── main.cpp       CLI entry point
│   ├── encoders/      Character-to-SDR encoding
│   ├── text/          Text file reading and chunking
│   └── runtime/       HTM runtime with text input (GUI-compatible)
├── gui/               Container-based GUI scripts
│   ├── build_image.sh Build the Qt6 container image
│   └── run_gui.sh     Run chat_htm with GUI in the container
├── configs/           YAML configuration files
├── tests/             Unit and integration tests
├── docs/              Architecture documentation
└── CMakeLists.txt     Build configuration
```
