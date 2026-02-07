#include <gtest/gtest.h>

#include <memory>

#include <htm_flow/config.hpp>
#include <htm_flow/config_loader.hpp>

#include "encoders/scalar_encoder.hpp"
#include "runtime/text_runtime.hpp"
#include "text/text_chunker.hpp"

using chat_htm::ScalarEncoder;
using chat_htm::TextChunker;
using chat_htm::TextRuntime;

namespace {

/// Create a small single-layer config suitable for fast integration tests.
htm_flow::HTMRegionConfig make_test_config(int input_rows, int input_cols) {
  htm_flow::HTMLayerConfig layer;
  layer.num_input_rows = input_rows;
  layer.num_input_cols = input_cols;
  layer.num_column_rows = 10;
  layer.num_column_cols = 20;
  layer.pot_width = 10;
  layer.pot_height = 1;
  layer.center_pot_synapses = true;
  layer.connected_perm = 0.3f;
  layer.min_overlap = 2;
  layer.wrap_input = true;
  layer.inhibition_width = 20;
  layer.inhibition_height = 1;
  layer.desired_local_activity = 1;
  layer.spatial_permanence_inc = 0.1f;
  layer.spatial_permanence_dec = 0.05f;
  layer.cells_per_column = 4;
  layer.max_segments_per_cell = 3;
  layer.max_synapses_per_segment = 15;
  layer.activation_threshold = 4;
  layer.sequence_permanence_inc = 0.1f;
  layer.sequence_permanence_dec = 0.05f;
  layer.temp_enabled = false;
  layer.log_timings = false;

  htm_flow::HTMRegionConfig cfg;
  cfg.layers.push_back(layer);
  return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// Smoke test: can we create a TextRuntime and step it?
// ---------------------------------------------------------------------------

TEST(TextHTMIntegration, SmokeTestSteps) {
  int rows = 10, cols = 10;
  auto cfg = make_test_config(rows, cols);
  ScalarEncoder::Params ep{.n = rows * cols, .w = 9, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(ep);

  auto chunker = std::make_unique<TextChunker>(
      TextChunker::from_string("abcabcabc"));
  TextRuntime rt(cfg, std::move(chunker), enc, "smoke");

  // Should not throw
  rt.step(20);
  EXPECT_GT(rt.chunker().total_steps(), 0u);
}

// ---------------------------------------------------------------------------
// Verify that the encoder output matches the HTM input dimensions
// ---------------------------------------------------------------------------

TEST(TextHTMIntegration, EncoderDimensionsMatchLayer0) {
  int rows = 10, cols = 10;
  auto cfg = make_test_config(rows, cols);
  ScalarEncoder::Params ep{.n = rows * cols, .w = 9, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(ep);

  auto sdr = enc.encode('a');
  EXPECT_EQ(static_cast<int>(sdr.size()),
            cfg.layers[0].num_input_rows * cfg.layers[0].num_input_cols);
}

// ---------------------------------------------------------------------------
// Feed a simple repeating pattern and verify the network runs
// ---------------------------------------------------------------------------

TEST(TextHTMIntegration, RepeatingPatternRuns) {
  int rows = 10, cols = 10;
  auto cfg = make_test_config(rows, cols);
  ScalarEncoder::Params ep{.n = rows * cols, .w = 9, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(ep);

  // "ababab..." is a simple repeating sequence the HTM should eventually learn.
  auto chunker = std::make_unique<TextChunker>(
      TextChunker::from_string("ababababababababababab"));
  TextRuntime rt(cfg, std::move(chunker), enc, "pattern");

  // Run for multiple passes through the text.
  rt.step(200);

  // At minimum, the network should have processed characters.
  EXPECT_EQ(rt.chunker().total_steps(), 200u);
}

// ---------------------------------------------------------------------------
// Multi-layer region with text input
// ---------------------------------------------------------------------------

TEST(TextHTMIntegration, MultiLayerRuns) {
  int rows = 10, cols = 10;
  auto cfg = make_test_config(rows, cols);

  // Add a second layer
  htm_flow::HTMLayerConfig layer2;
  layer2.num_column_rows = 10;
  layer2.num_column_cols = 20;
  layer2.pot_width = 10;
  layer2.pot_height = 1;
  layer2.center_pot_synapses = true;
  layer2.min_overlap = 2;
  layer2.inhibition_width = 20;
  layer2.desired_local_activity = 1;
  layer2.cells_per_column = 3;
  layer2.max_segments_per_cell = 2;
  layer2.activation_threshold = 3;
  layer2.temp_enabled = false;
  layer2.log_timings = false;
  cfg.layers.push_back(layer2);

  ScalarEncoder::Params ep{.n = rows * cols, .w = 9, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(ep);

  auto chunker = std::make_unique<TextChunker>(
      TextChunker::from_string("hello world "));
  TextRuntime rt(cfg, std::move(chunker), enc, "multilayer");

  // Run several epochs
  rt.step(100);
  EXPECT_EQ(rt.num_layers(), 2);
  EXPECT_EQ(rt.chunker().total_steps(), 100u);
}

// ---------------------------------------------------------------------------
// Load config from YAML file
// ---------------------------------------------------------------------------

TEST(TextHTMIntegration, LoadFromYAML) {
  // Use the small_text.yaml config that ships with the project.
  std::string config_path = CHAT_HTM_TEST_DATA_DIR "/../../configs/small_text.yaml";

  htm_flow::HTMRegionConfig cfg;
  ASSERT_NO_THROW(cfg = htm_flow::load_region_config(config_path));
  EXPECT_GE(cfg.layers.size(), 1u);

  int input_bits = cfg.layers[0].num_input_rows * cfg.layers[0].num_input_cols;
  ScalarEncoder::Params ep{.n = input_bits, .w = 9, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(ep);

  auto chunker = std::make_unique<TextChunker>(
      TextChunker::from_string("test"));
  TextRuntime rt(cfg, std::move(chunker), enc, "yaml_test");
  rt.step(10);
}
