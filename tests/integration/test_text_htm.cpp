#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>

#include <yaml-cpp/yaml.h>

#include <htm_flow/config.hpp>
#include <htm_flow/config_loader.hpp>

#include "encoders/scalar_encoder.hpp"
#include "encoders/word_row_encoder.hpp"
#include "runtime/text_runtime.hpp"
#include "text/text_chunker.hpp"
#include "text/word_chunker.hpp"

using chat_htm::ScalarEncoder;
using chat_htm::TextChunker;
using chat_htm::TextRuntime;
using chat_htm::WordChunker;
using chat_htm::WordRowEncoder;

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

std::filesystem::path write_temp_yaml(const std::string& stem, const std::string& contents) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    (stem + "_" + std::to_string(stamp) + ".yaml");
  std::ofstream out(path);
  out << contents;
  out.close();
  return path;
}

std::string test_data_dir() {
#ifdef CHAT_HTM_TEST_DATA_DIR
  return CHAT_HTM_TEST_DATA_DIR;
#else
  return ".";
#endif
}

WordRowEncoder::Params parse_word_row_encoder_params(const std::string& config_path,
                                                     int input_rows,
                                                     int input_cols) {
  WordRowEncoder::Params p;
  p.rows = input_rows;
  p.cols = input_cols;

  YAML::Node root = YAML::LoadFile(config_path);
  if (root["encoder"]) {
    const auto& enc = root["encoder"];
    if (enc["letter_bits"]) {
      p.letter_bits = enc["letter_bits"].as<int>();
    }
    if (enc["alphabet"]) {
      p.alphabet = enc["alphabet"].as<std::string>();
    }
  }

  return p;
}

struct LayerWindowMetrics {
  double mean_burst_fraction = 0.0;
  double mean_predictive_fraction = 0.0;
  double mean_learning_fraction = 0.0;
};

int popcount64(std::uint64_t value) {
  return __builtin_popcountll(value);
}

LayerWindowMetrics sample_layer_window(TextRuntime& rt, int layer_idx, int steps) {
  LayerWindowMetrics out;
  for (int i = 0; i < steps; ++i) {
    rt.step(1);
    const auto snap = rt.region().layer(layer_idx).snapshot();
    const int active_cols = static_cast<int>(snap.active_column_indices.size());
    if (active_cols <= 0) {
      continue;
    }

    int bursting_cols = 0;
    int predictive_cols = 0;
    int learning_cols = 0;
    for (int idx : snap.active_column_indices) {
      if (idx < 0 || idx >= static_cast<int>(snap.column_cell_masks.size())) {
        continue;
      }
      const auto& masks = snap.column_cell_masks[static_cast<std::size_t>(idx)];
      if (popcount64(masks.active) > 1) {
        ++bursting_cols;
      }
      if (masks.predictive != 0) {
        ++predictive_cols;
      }
      if (masks.learning != 0) {
        ++learning_cols;
      }
    }

    out.mean_burst_fraction += static_cast<double>(bursting_cols) / static_cast<double>(active_cols);
    out.mean_predictive_fraction += static_cast<double>(predictive_cols) / static_cast<double>(active_cols);
    out.mean_learning_fraction += static_cast<double>(learning_cols) / static_cast<double>(active_cols);
  }

  if (steps > 0) {
    out.mean_burst_fraction /= static_cast<double>(steps);
    out.mean_predictive_fraction /= static_cast<double>(steps);
    out.mean_learning_fraction /= static_cast<double>(steps);
  }
  return out;
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
  std::string config_path = test_data_dir() + "/../../configs/small_text.yaml";

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

TEST(TextHTMIntegration, RuntimePatchFileUpdatesLiveParameters) {
  int rows = 10, cols = 10;
  auto cfg = make_test_config(rows, cols);
  ScalarEncoder::Params ep{.n = rows * cols, .w = 9, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(ep);

  auto chunker = std::make_unique<TextChunker>(TextChunker::from_string("abcabcabc"));
  TextRuntime rt(cfg, std::move(chunker), enc, "runtime_patch");
  rt.step(3);
  ASSERT_EQ(rt.timestep(), 3);

  const auto patch_path = write_temp_yaml(
      "text_runtime_patch",
      R"(layers:
  - spatial_learning:
      permanence_inc: 0.23
    sequence_memory:
      activation_threshold: 8
    temporal_pooling:
      spatial_permanence_inc: 0.07
      active_predict_proximal_scale: 0.8
      post_active_proximal_scale: 0.1
)");

  const auto result = rt.apply_runtime_patch_file(patch_path.string());
  ASSERT_TRUE(result.ok) << result.message;
  EXPECT_EQ(rt.timestep(), 3);
  EXPECT_FLOAT_EQ(rt.region().layer(0).config().spatial_permanence_inc, 0.23f);
  EXPECT_EQ(rt.region().layer(0).config().activation_threshold, 8);
  EXPECT_FLOAT_EQ(rt.region().layer(0).config().temp_spatial_permanence_inc, 0.07f);
  EXPECT_FLOAT_EQ(rt.region().layer(0).config().temp_active_predict_proximal_scale, 0.8f);
  EXPECT_FLOAT_EQ(rt.region().layer(0).config().temp_post_active_proximal_scale, 0.1f);
}

TEST(TextHTMIntegration, ConfigYAMLLoadsTemporalPoolingProximalReinforcement) {
  const auto config_path = write_temp_yaml(
      "tp_proximal_reinforcement_config",
      R"(layers:
  - input:
      rows: 10
      cols: 10
    columns:
      rows: 10
      cols: 20
    temporal_pooling:
      enabled: true
      spatial_permanence_inc: 0.075
      active_predict_proximal_scale: 0.9
      post_active_proximal_scale: 0.1
)");

  htm_flow::HTMRegionConfig cfg;
  ASSERT_NO_THROW(cfg = htm_flow::load_region_config(config_path.string()));
  ASSERT_EQ(cfg.layers.size(), 1u);
  EXPECT_FLOAT_EQ(cfg.layers[0].temp_spatial_permanence_inc, 0.075f);
  EXPECT_FLOAT_EQ(cfg.layers[0].temp_active_predict_proximal_scale, 0.9f);
  EXPECT_FLOAT_EQ(cfg.layers[0].temp_post_active_proximal_scale, 0.1f);
}

TEST(TextHTMIntegration, WordRowsModeLearnsSimpleSentenceSequence) {
  // For the word-row encoder:
  // rows = max word length, cols = letter_bits * (alphabet_size + unknown_bucket)
  int rows = 5;
  int cols = 108;  // 4 * (26 + 1)
  auto cfg = make_test_config(rows, cols);

  WordRowEncoder::Params ep;
  ep.rows = rows;
  ep.cols = cols;
  ep.letter_bits = 4;
  ep.alphabet = "abcdefghijklmnopqrstuvwxyz";
  WordRowEncoder enc(ep);

  auto chunker = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));
  TextRuntime rt(cfg, std::move(chunker), enc, "word_rows");

  rt.step(600);
  auto snap = rt.region().layer(0).snapshot();
  int predictive_cells = 0;
  for (const auto& masks : snap.column_cell_masks) {
    predictive_cells += (masks.predictive != 0) ? 1 : 0;
  }

  EXPECT_EQ(rt.num_layers(), 1);
  EXPECT_EQ(rt.word_chunker().total_steps(), 600u);
  EXPECT_GT(snap.active_column_indices.size(), 0u);
  EXPECT_GT(predictive_cells, 0);
}

TEST(TextHTMIntegration, RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting) {
  const std::string config_path =
      test_data_dir() + "/../../configs/word_rows_2layer_text.yaml";
  htm_flow::HTMRegionConfig cfg;
  ASSERT_NO_THROW(cfg = htm_flow::load_region_config(config_path));
  ASSERT_EQ(cfg.layers.size(), 2u);
  cfg.layers[1].temp_enabled = false;
  cfg.layers[1].temp_enable_persistence = false;
  cfg.layers[1].temp_delay_length = 6;
  cfg.layers[1].temp_spatial_permanence_inc = 0.0f;
  cfg.layers[1].temp_sequence_permanence_inc = 0.08f;
  cfg.layers[1].temp_sequence_permanence_dec = 0.01f;

  const auto encoder_params = parse_word_row_encoder_params(
      config_path, cfg.layers[0].num_input_rows, cfg.layers[0].num_input_cols);
  WordRowEncoder enc(encoder_params);
  auto chunker = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));
  TextRuntime rt(cfg, std::move(chunker), enc, "word_rows_schedule");

  const int patch_timestep = 1000;
  const int compare_window = 80;
  ASSERT_GT(patch_timestep, compare_window);
  ASSERT_EQ(rt.num_layers(), 2);

  rt.step(patch_timestep - compare_window);
  ASSERT_EQ(rt.timestep(), patch_timestep - compare_window);

  const LayerWindowMetrics before_enable = sample_layer_window(rt, /*layer_idx=*/1, compare_window);
  ASSERT_EQ(rt.timestep(), patch_timestep);

  const auto patch_path = write_temp_yaml(
      "layer1_enable_temporal_pooling",
      R"(layers:
  - {}
  - temporal_pooling:
      enabled: true
      enable_persistence: false
      delay_length: 6
      spatial_permanence_inc: 0.01
      sequence_permanence_inc: 0.08
      sequence_permanence_dec: 0.01
)");
  const auto result = rt.apply_runtime_patch_file(patch_path.string());
  ASSERT_TRUE(result.ok) << result.message;
  EXPECT_FLOAT_EQ(rt.region().layer(1).config().temp_spatial_permanence_inc, 0.01f);

  const int post_window = compare_window;
  ASSERT_GT(post_window, 0);

  const LayerWindowMetrics after_enable = sample_layer_window(rt, /*layer_idx=*/1, post_window);

  EXPECT_LE(after_enable.mean_burst_fraction, before_enable.mean_burst_fraction + 0.10)
      << "Layer 1 burst fraction spiked after enabling temporal pooling. before="
      << before_enable.mean_burst_fraction << " after=" << after_enable.mean_burst_fraction;
  EXPECT_GE(after_enable.mean_predictive_fraction, before_enable.mean_predictive_fraction - 0.10)
      << "Layer 1 predictive coverage collapsed after enabling temporal pooling. before="
      << before_enable.mean_predictive_fraction << " after=" << after_enable.mean_predictive_fraction;
  EXPECT_GE(after_enable.mean_learning_fraction, before_enable.mean_learning_fraction - 0.10)
      << "Layer 1 learning coverage collapsed after enabling temporal pooling. before="
      << before_enable.mean_learning_fraction << " after=" << after_enable.mean_learning_fraction;
}

TEST(TextHTMIntegration, AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP) {
  const std::string config_path =
      test_data_dir() + "/../../configs/word_rows_2layer_text.yaml";

  htm_flow::HTMRegionConfig cfg_off;
  ASSERT_NO_THROW(cfg_off = htm_flow::load_region_config(config_path));
  ASSERT_EQ(cfg_off.layers.size(), 2u);
  cfg_off.layers[1].temp_enabled = false;
  cfg_off.layers[1].temp_enable_persistence = false;

  htm_flow::HTMRegionConfig cfg_on = cfg_off;
  cfg_on.layers[1].temp_enabled = true;
  cfg_on.layers[1].temp_enable_persistence = false;
  cfg_on.layers[1].temp_delay_length = 6;
  cfg_on.layers[1].temp_spatial_permanence_inc = 0.01f;
  cfg_on.layers[1].temp_sequence_permanence_inc = 0.08f;
  cfg_on.layers[1].temp_sequence_permanence_dec = 0.01f;

  const auto encoder_params = parse_word_row_encoder_params(
      config_path, cfg_on.layers[0].num_input_rows, cfg_on.layers[0].num_input_cols);
  WordRowEncoder enc(encoder_params);

  auto chunker_off = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));
  auto chunker_on = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));

  TextRuntime rt_off(cfg_off, std::move(chunker_off), enc, "word_rows_tp_off");
  TextRuntime rt_on(cfg_on, std::move(chunker_on), enc, "word_rows_tp_on");

  const int warmup_steps = 900;
  const int measure_steps = 120;
  rt_off.step(warmup_steps);
  rt_on.step(warmup_steps);

  const LayerWindowMetrics off_metrics = sample_layer_window(rt_off, /*layer_idx=*/1, measure_steps);
  const LayerWindowMetrics on_metrics = sample_layer_window(rt_on, /*layer_idx=*/1, measure_steps);

  EXPECT_LE(on_metrics.mean_burst_fraction, off_metrics.mean_burst_fraction + 0.10)
      << "Always-on temporal pooling increased Layer 1 bursting too much. off="
      << off_metrics.mean_burst_fraction << " on=" << on_metrics.mean_burst_fraction;
  EXPECT_GE(on_metrics.mean_predictive_fraction, off_metrics.mean_predictive_fraction - 0.10)
      << "Always-on temporal pooling reduced Layer 1 predictive coverage too much. off="
      << off_metrics.mean_predictive_fraction << " on=" << on_metrics.mean_predictive_fraction;
  EXPECT_GE(on_metrics.mean_learning_fraction, off_metrics.mean_learning_fraction - 0.10)
      << "Always-on temporal pooling reduced Layer 1 learning coverage too much. off="
      << off_metrics.mean_learning_fraction << " on=" << on_metrics.mean_learning_fraction;
}

TEST(TextHTMIntegration, AlwaysOnTemporalPoolingWithPersistenceDoesNotRaiseLayer1BurstingVsNoTP) {
  const std::string config_path =
      test_data_dir() + "/../../configs/word_rows_2layer_text.yaml";

  htm_flow::HTMRegionConfig cfg_off;
  ASSERT_NO_THROW(cfg_off = htm_flow::load_region_config(config_path));
  ASSERT_EQ(cfg_off.layers.size(), 2u);
  cfg_off.layers[1].temp_enabled = false;
  cfg_off.layers[1].temp_enable_persistence = false;

  htm_flow::HTMRegionConfig cfg_on = cfg_off;
  cfg_on.layers[1].temp_enabled = true;
  cfg_on.layers[1].temp_enable_persistence = true;
  cfg_on.layers[1].temp_delay_length = 6;
  cfg_on.layers[1].temp_spatial_permanence_inc = 0.01f;
  cfg_on.layers[1].temp_sequence_permanence_inc = 0.08f;
  cfg_on.layers[1].temp_sequence_permanence_dec = 0.01f;

  const auto encoder_params = parse_word_row_encoder_params(
      config_path, cfg_on.layers[0].num_input_rows, cfg_on.layers[0].num_input_cols);
  WordRowEncoder enc(encoder_params);

  auto chunker_off = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));
  auto chunker_on = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));

  TextRuntime rt_off(cfg_off, std::move(chunker_off), enc, "word_rows_tp_off");
  TextRuntime rt_on(cfg_on, std::move(chunker_on), enc, "word_rows_tp_persistent");

  const int warmup_steps = 900;
  const int measure_steps = 120;
  rt_off.step(warmup_steps);
  rt_on.step(warmup_steps);

  const LayerWindowMetrics off_metrics = sample_layer_window(rt_off, /*layer_idx=*/1, measure_steps);
  const LayerWindowMetrics on_metrics = sample_layer_window(rt_on, /*layer_idx=*/1, measure_steps);

  EXPECT_LE(on_metrics.mean_burst_fraction, off_metrics.mean_burst_fraction + 0.10)
      << "Persistence-enabled temporal pooling increased Layer 1 bursting too much. off="
      << off_metrics.mean_burst_fraction << " on=" << on_metrics.mean_burst_fraction;
  EXPECT_GE(on_metrics.mean_predictive_fraction, off_metrics.mean_predictive_fraction - 0.10)
      << "Persistence-enabled temporal pooling reduced Layer 1 predictive coverage too much. off="
      << off_metrics.mean_predictive_fraction << " on=" << on_metrics.mean_predictive_fraction;
  EXPECT_GE(on_metrics.mean_learning_fraction, off_metrics.mean_learning_fraction - 0.10)
      << "Persistence-enabled temporal pooling reduced Layer 1 learning coverage too much. off="
      << off_metrics.mean_learning_fraction << " on=" << on_metrics.mean_learning_fraction;
}

TEST(TextHTMIntegration, StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP) {
  const std::string config_path =
      test_data_dir() + "/../../configs/word_rows_2layer_text.yaml";

  htm_flow::HTMRegionConfig cfg_off;
  ASSERT_NO_THROW(cfg_off = htm_flow::load_region_config(config_path));
  ASSERT_EQ(cfg_off.layers.size(), 2u);
  cfg_off.layers[1].temp_enabled = false;
  cfg_off.layers[1].temp_enable_persistence = false;

  htm_flow::HTMRegionConfig cfg_on = cfg_off;
  cfg_on.layers[1].temp_enabled = true;
  cfg_on.layers[1].temp_enable_persistence = true;
  cfg_on.layers[1].temp_delay_length = 8;
  cfg_on.layers[1].temp_spatial_permanence_inc = 0.04f;
  cfg_on.layers[1].temp_sequence_permanence_inc = 0.18f;
  cfg_on.layers[1].temp_sequence_permanence_dec = 0.004f;

  const auto encoder_params = parse_word_row_encoder_params(
      config_path, cfg_on.layers[0].num_input_rows, cfg_on.layers[0].num_input_cols);
  WordRowEncoder enc(encoder_params);

  auto chunker_off = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));
  auto chunker_on = std::make_unique<WordChunker>(
      WordChunker::from_string("small cat likes warm milk small dog likes warm soup "));

  TextRuntime rt_off(cfg_off, std::move(chunker_off), enc, "word_rows_tp_off");
  TextRuntime rt_on(cfg_on, std::move(chunker_on), enc, "word_rows_tp_strong");

  const int warmup_steps = 900;
  const int measure_steps = 120;
  rt_off.step(warmup_steps);
  rt_on.step(warmup_steps);

  const LayerWindowMetrics off_metrics = sample_layer_window(rt_off, /*layer_idx=*/1, measure_steps);
  const LayerWindowMetrics on_metrics = sample_layer_window(rt_on, /*layer_idx=*/1, measure_steps);

  EXPECT_LE(on_metrics.mean_burst_fraction, off_metrics.mean_burst_fraction + 0.10)
      << "Strong temporal pooling increased Layer 1 bursting too much. off="
      << off_metrics.mean_burst_fraction << " on=" << on_metrics.mean_burst_fraction;
  EXPECT_GE(on_metrics.mean_predictive_fraction, off_metrics.mean_predictive_fraction - 0.10)
      << "Strong temporal pooling reduced Layer 1 predictive coverage too much. off="
      << off_metrics.mean_predictive_fraction << " on=" << on_metrics.mean_predictive_fraction;
  EXPECT_GE(on_metrics.mean_learning_fraction, off_metrics.mean_learning_fraction - 0.10)
      << "Strong temporal pooling reduced Layer 1 learning coverage too much. off="
      << off_metrics.mean_learning_fraction << " on=" << on_metrics.mean_learning_fraction;
}
