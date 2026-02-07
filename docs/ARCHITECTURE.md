# Chat HTM Architecture

This document explains how `chat_htm` works so that future developers (human or
LLM agents) can understand the project, extend it, and debug it quickly.

## Goal

Teach an HTM (Hierarchical Temporal Memory) network to learn and predict
temporal patterns in simple text.  The system reads a plain text file, converts
each character into a Sparse Distributed Representation (SDR), and feeds that
SDR to an `htm_flow` HTM network one timestep at a time.  The HTM learns to
predict the next character in the sequence.

## High-Level Pipeline

```
Text File  -->  TextChunker  -->  ScalarEncoder  -->  HTMRegion  -->  Predictions
                 (char)            (SDR)             (htm_flow)
```

1. **TextChunker** (`src/text/text_chunker.hpp`) reads a text file into memory
   and yields one character at a time.  When it reaches the end of the file it
   wraps around (enabling multi-epoch training).

2. **ScalarEncoder** (`src/encoders/scalar_encoder.hpp`) converts each
   character's ASCII value (0-127) into a binary SDR (a `std::vector<int>` of
   0s and 1s).  The encoder slides a contiguous window of `w` active bits
   across `n` total bits so that nearby ASCII values share overlapping bits.
   This gives the HTM spatial pooler a built-in notion of character similarity.

3. **HTMRegion** (from `htm_flow`) is a stack of HTM layers that performs
   spatial pooling, sequence memory, and (optionally) temporal pooling.  Layer 0
   receives the character SDR.  Higher layers learn increasingly abstract
   patterns over the outputs of the layers below.

4. **TextRuntime** (`src/runtime/text_runtime.hpp`) orchestrates the above
   components and implements the `IHtmRuntime` interface so the `htm_gui` Qt
   debugger can visualize the network in real time.

## How the Scalar Encoder Works

The encoder maps a value `v` in `[min_val, max_val]` to a binary vector of
length `n`, where exactly `w` contiguous bits are set to 1.

```
n = 20, w = 5, range = [0, 9]

encode(0): ██████░░░░░░░░░░░░░░   (bits 0-4)
encode(3): ░░░░██████░░░░░░░░░░   (bits 5-9)
encode(9): ░░░░░░░░░░░░░░░██████  (bits 15-19)
```

Key properties:
- **Sparsity**: Only `w/n` fraction of bits are active (~5%).
- **Semantic overlap**: Nearby values share active bits, distant values do not.
- **Deterministic**: Same input always produces the same SDR.

For text, `min_val=0, max_val=127` covers printable ASCII.  With `n=400`
(matching a 20x20 input grid) and `w=21`, adjacent characters like 'a' and 'b'
share ~18 of 21 active bits.

## Configuration

Configuration is via YAML files in `configs/`.  Each config file contains:

1. **Text/Encoder section** (chat_htm-specific, ignored by htm_flow):

```yaml
text:
  mode: character         # Only mode for now
encoder:
  active_bits: 21         # w: active bits per SDR
  min_value: 0            # ASCII range start
  max_value: 127          # ASCII range end
```

2. **HTM Region section** (standard htm_flow format):

```yaml
enable_feedback: false
layers:
  - name: Layer0_Text
    input:
      rows: 20
      cols: 20            # n = rows * cols = 400
    columns:
      rows: 20
      cols: 40
    overlap: { ... }
    inhibition: { ... }
    spatial_learning: { ... }
    sequence_memory: { ... }
    temporal_pooling: { ... }
```

The `input.rows * input.cols` of Layer 0 determines `n` (total encoder bits).
`encoder.active_bits` determines `w`.  All other layer parameters control the
HTM algorithm (see htm_flow's config documentation).

### Key parameters to experiment with

| Parameter | Where | Effect |
|-----------|-------|--------|
| `encoder.active_bits` | encoder | More bits = more overlap between characters |
| `input.rows/cols` | layer 0 | Larger = more SDR resolution, slower |
| `cells_per_column` | sequence_memory | More cells = more sequence capacity |
| `max_segments_per_cell` | sequence_memory | More segments = more contexts |
| `activation_threshold` | sequence_memory | Lower = more sensitive predictions |
| `temporal_pooling.enabled` | temporal_pooling | Stabilize representations across time |
| Number of layers | layers array | More layers = more abstraction |

## Integration with htm_flow

`chat_htm` uses `htm_flow` as a git submodule and links against the
`htm_flow_core` static library target.

```
chat_htm/CMakeLists.txt
  └── add_subdirectory(htm_flow)   # provides htm_flow_core target
      └── target_link_libraries(chat_htm PRIVATE htm_flow_core)
```

The `htm_flow_core` library provides:
- `HTMRegion` / `HTMLayer` -- the core HTM algorithm
- `HTMRegionConfig` / `HTMLayerConfig` -- configuration structs
- `load_region_config()` / `save_region_config()` -- YAML config I/O
- `IHtmRuntime` interface -- for GUI visualization
- `HTMRegionRuntime` -- GUI runtime adapter (not used by chat_htm directly,
  but available for reference)

### Adding the GUI

Build with `-DHTM_FLOW_WITH_GUI=ON` (requires Qt6).  Then pass `--gui` to the
CLI.  The `TextRuntime` class implements `IHtmRuntime`, so the htm_gui
debugger can visualize column activations, cell predictions, and synapses
while the network processes text.

## Adding New Encoding Schemes

To add a new encoder (e.g. word-level, n-gram, or hash-based):

1. Create a new header in `src/encoders/` implementing an `encode(T) -> std::vector<int>` method.
2. Update `TextRuntime` (or create a new runtime) to use the new encoder.
3. Add a `text.mode` value in the YAML config to select it.
4. Add unit tests in `tests/unit/`.

## Project Layout

```
src/
  main.cpp                 CLI entry point
  encoders/
    scalar_encoder.hpp     Scalar-to-SDR encoder (header-only)
  text/
    text_chunker.hpp       Text file reader (header-only)
  runtime/
    text_runtime.hpp/cpp   IHtmRuntime for text (links to htm_flow)

configs/
  default_text.yaml        2-layer, 400-bit SDR
  small_text.yaml          1-layer, 100-bit SDR (fast testing)

tests/
  unit/                    Encoder and chunker tests
  integration/             Full pipeline tests
  test_data/               Sample text files

docs/
  ARCHITECTURE.md          This file
```

## Building and Running

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./chat_htm --input ../tests/test_data/hello_world.txt \
           --config ../configs/small_text.yaml \
           --epochs 5 --log

# Run tests
ctest --output-on-failure
```

### Running with GUI (container -- no local Qt6 needed)

```bash
# Build container image once
./gui/build_image.sh

# Launch GUI (all paths relative to project root)
./gui/run_gui.sh --input tests/test_data/hello_world.txt \
                 --config configs/default_text.yaml --log
```

### Running with GUI (native -- requires Qt6 installed)

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHTM_FLOW_WITH_GUI=ON
make -j$(nproc)
./chat_htm --input ../tests/test_data/hello_world.txt \
           --config ../configs/default_text.yaml --gui
```
