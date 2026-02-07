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

2. Install htm_flow dependencies:
```bash
cd htm_flow && ./setup.sh && cd ..
```

3. Build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
# Feed a text file to the HTM network
./chat_htm --input path/to/text.txt --config ../configs/small_text.yaml

# With logging to see progress
./chat_htm --input path/to/text.txt --config ../configs/default_text.yaml --epochs 10 --log

# With GUI visualization (requires building with -DHTM_FLOW_WITH_GUI=ON)
./chat_htm --input path/to/text.txt --config ../configs/default_text.yaml --gui

# Process a specific number of steps
./chat_htm --input path/to/text.txt --config ../configs/small_text.yaml --steps 500

# List available configs
./chat_htm --list-configs
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--input FILE` | Path to a text file (required) |
| `--config FILE` | Path to a YAML config file (required) |
| `--steps N` | Number of character steps (default: entire file) |
| `--epochs N` | Number of passes through the file (default: 1) |
| `--gui` | Launch the htm_gui visual debugger |
| `--log` | Print per-step progress and accuracy |
| `--list-configs` | List YAML configs in `configs/` |

## Configuration

YAML config files in `configs/` control both the text encoding and HTM network parameters:

- **`configs/small_text.yaml`** -- Single-layer, 100-bit SDR. Fast for testing.
- **`configs/default_text.yaml`** -- Two-layer, 400-bit SDR. Better capacity.

You can create your own configs to experiment with different network sizes, numbers of layers, learning rates, and temporal pooling settings. See the existing configs and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details on all parameters.

## Running Tests

```bash
cd build
ctest --output-on-failure
```

## Building with GUI Support

The htm_gui Qt6 debugger lets you visualize the HTM network as it processes text -- you can see column activations, cell predictions, and synapses update in real time.

### Option 1: Container-based GUI (recommended, no local Qt6 needed)

Build the container image once, then use `run_gui.sh` to launch:

```bash
# Build the image (uses podman, only needed once)
./gui/build_image.sh

# Run chat_htm with GUI inside the container
./gui/run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml
./gui/run_gui.sh --input tests/test_data/alphabet.txt --config configs/default_text.yaml --log
./gui/run_gui.sh --input mydata.txt --config configs/default_text.yaml --epochs 10
```

If you already built htm_flow's GUI image you can reuse it:
```bash
IMAGE_NAME=htm_flow_gui:qt6 ./gui/run_gui.sh --input tests/test_data/hello_world.txt --config configs/small_text.yaml
```

### Option 2: Native build (requires Qt6 installed locally)

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHTM_FLOW_WITH_GUI=ON
make -j$(nproc)
./chat_htm --input ../tests/test_data/hello_world.txt --config ../configs/default_text.yaml --gui
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
