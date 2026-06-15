#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <htm_flow/config_loader.hpp>

#include "encoders/scalar_encoder.hpp"
#include "encoders/word_row_encoder.hpp"
#include "runtime/text_runtime.hpp"
#include "text/text_chunker.hpp"
#include "text/word_chunker.hpp"

#ifdef HTM_FLOW_WITH_GUI
#include <htm_gui/debugger.hpp>
#endif

namespace {

enum class TextMode {
  Character,
  WordRows
};

void usage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " --input FILE --config FILE [options]\n\n"
      << "Required:\n"
      << "  --input  FILE   Path to a text file to feed to the HTM network\n"
      << "  --config FILE   Path to a YAML config file (see configs/)\n\n"
      << "Options:\n"
      << "  --steps  N      Number of input steps to process (default: whole file/word list)\n"
      << "  --epochs N      Number of passes through the text file (default: 1)\n"
      << "  --gui           Launch the htm_gui debugger for visualization\n"
      << "  --theme MODE    GUI theme: light|dark (CLI overrides YAML gui.theme)\n"
      << "  --log           Print per-step logging (character, epoch, accuracy)\n"
      << "  --list-configs  List available YAML configs in configs/\n"
      << "  -h, --help      Show this help message\n\n"
      << "Examples:\n"
      << "  " << prog << " --input data/hello.txt --config configs/small_text.yaml\n"
      << "  " << prog << " --input data/hello.txt --config configs/default_text.yaml --gui\n"
      << "  " << prog << " --input data/hello.txt --config configs/small_text.yaml --epochs 10 --log\n";
}

/// Parse the text/encoder sections from the YAML config.
/// These are chat_htm-specific keys that htm_flow's loader silently ignores.
TextMode parse_text_mode(const std::string& config_path) {
  try {
    YAML::Node root = YAML::LoadFile(config_path);
    if (root["text"] && root["text"]["mode"]) {
      const std::string mode = root["text"]["mode"].as<std::string>();
      if (mode == "word_rows") return TextMode::WordRows;
    }
  } catch (const YAML::Exception& e) {
    std::cerr << "Warning: could not parse text mode: " << e.what() << "\n";
  }
  return TextMode::Character;
}

std::string parse_gui_theme(const std::string& config_path) {
  try {
    YAML::Node root = YAML::LoadFile(config_path);
    if (root["gui"] && root["gui"]["theme"]) {
      return root["gui"]["theme"].as<std::string>();
    }
  } catch (const YAML::Exception& e) {
    std::cerr << "Warning: could not parse gui theme: " << e.what() << "\n";
  }
  return {};
}

int popcount64(std::uint64_t value) {
#if defined(__GNUG__) || defined(__clang__)
  return __builtin_popcountll(value);
#else
  int count = 0;
  while (value != 0) {
    value &= (value - 1);
    ++count;
  }
  return count;
#endif
}

bool col_time_is_set(const std::vector<int>& col_time_tensor2, int col, int time_step) {
  const int idx0 = col * 2;
  if (idx0 < 0 || idx0 + 1 >= static_cast<int>(col_time_tensor2.size())) {
    return false;
  }
  return col_time_tensor2[static_cast<std::size_t>(idx0)] == time_step ||
         col_time_tensor2[static_cast<std::size_t>(idx0 + 1)] == time_step;
}

bool cell_time_is_set(const std::vector<int>& cell_time_tensor2,
                      int cells_per_column,
                      int col,
                      int cell,
                      int time_step) {
  const int idx0 = (col * cells_per_column + cell) * 2;
  if (idx0 < 0 || idx0 + 1 >= static_cast<int>(cell_time_tensor2.size())) {
    return false;
  }
  return cell_time_tensor2[static_cast<std::size_t>(idx0)] == time_step ||
         cell_time_tensor2[static_cast<std::size_t>(idx0 + 1)] == time_step;
}

bool cell_has_active_segment(const std::vector<int>& active_segments_time,
                             int cells_per_column,
                             int max_segments_per_cell,
                             int col,
                             int cell,
                             int time_step) {
  const int base = (col * cells_per_column + cell) * max_segments_per_cell;
  if (base < 0 || base + max_segments_per_cell > static_cast<int>(active_segments_time.size())) {
    return false;
  }
  for (int seg = 0; seg < max_segments_per_cell; ++seg) {
    if (active_segments_time[static_cast<std::size_t>(base + seg)] == time_step) {
      return true;
    }
  }
  return false;
}

struct LayerMetrics {
  int samples = 0;
  double active_columns = 0.0;
  double multi_active_fraction = 0.0;
  double true_burst_fraction = 0.0;
  double new_true_burst_fraction = 0.0;
  double repeated_true_burst_fraction = 0.0;
  double predicted_multi_active_fraction = 0.0;
  double predictive_fraction = 0.0;
  double learning_fraction = 0.0;
  int new_true_bursts = 0;
  int new_burst_no_prev_prediction = 0;
  int new_burst_prev_prediction_without_segment = 0;
  int new_burst_prev_segment_without_prediction = 0;
  int new_burst_had_prev_prediction_and_segment = 0;
  std::vector<uint8_t> prev_true_bursting;
  std::vector<int> true_burst_counts;
  std::vector<int> new_true_burst_counts;
  std::vector<int> position_samples;
  std::vector<int> position_new_true_burst_counts;

  void add(const htm_gui::Snapshot& snap,
           const std::vector<int>& burst_cols_time,
           const std::vector<int>& predict_cells_time,
           const std::vector<int>& active_segments_time,
           int cells_per_column,
           int max_segments_per_cell,
           int input_position,
           int input_size) {
    const int active = static_cast<int>(snap.active_column_indices.size());
    if (active <= 0) {
      prev_true_bursting.assign(snap.column_cell_masks.size(), 0);
      return;
    }
    if (input_size > 0 && static_cast<int>(position_samples.size()) != input_size) {
      position_samples.assign(static_cast<std::size_t>(input_size), 0);
      position_new_true_burst_counts.assign(static_cast<std::size_t>(input_size), 0);
    }
    if (input_position >= 0 && input_position < static_cast<int>(position_samples.size())) {
      ++position_samples[static_cast<std::size_t>(input_position)];
    }
    if (prev_true_bursting.size() != snap.column_cell_masks.size()) {
      prev_true_bursting.assign(snap.column_cell_masks.size(), 0);
    }
    if (true_burst_counts.size() != snap.column_cell_masks.size()) {
      true_burst_counts.assign(snap.column_cell_masks.size(), 0);
    }
    if (new_true_burst_counts.size() != snap.column_cell_masks.size()) {
      new_true_burst_counts.assign(snap.column_cell_masks.size(), 0);
    }

    int multi_active = 0;
    int true_bursting = 0;
    int new_true_bursting = 0;
    int repeated_true_bursting = 0;
    int predicted_multi_active = 0;
    int predictive = 0;
    int learning = 0;
    std::vector<uint8_t> current_true_bursting(snap.column_cell_masks.size(), 0);
    for (int idx : snap.active_column_indices) {
      if (idx < 0 || idx >= static_cast<int>(snap.column_cell_masks.size())) {
        continue;
      }
      const auto& masks = snap.column_cell_masks[static_cast<std::size_t>(idx)];
      const bool has_prediction = masks.predictive != 0;
      const bool is_multi_active = popcount64(masks.active) > 1;
      const bool is_true_burst = col_time_is_set(burst_cols_time, idx, snap.timestep);
      if (is_multi_active) {
        ++multi_active;
        if (!is_true_burst && has_prediction) {
          ++predicted_multi_active;
        }
      }
      if (is_true_burst) {
        ++true_bursting;
        current_true_bursting[static_cast<std::size_t>(idx)] = 1;
        ++true_burst_counts[static_cast<std::size_t>(idx)];
        if (prev_true_bursting[static_cast<std::size_t>(idx)] == 1) {
          ++repeated_true_bursting;
        } else {
          ++new_true_bursting;
          ++new_true_bursts;
          ++new_true_burst_counts[static_cast<std::size_t>(idx)];
          if (input_position >= 0 && input_position < static_cast<int>(position_new_true_burst_counts.size())) {
            ++position_new_true_burst_counts[static_cast<std::size_t>(input_position)];
          }

          bool had_prev_prediction = false;
          bool had_prev_segment = false;
          bool had_prev_prediction_and_segment = false;
          for (int cell = 0; cell < cells_per_column; ++cell) {
            const bool pred_prev = cell_time_is_set(predict_cells_time,
                                                    cells_per_column,
                                                    idx,
                                                    cell,
                                                    snap.timestep - 1);
            const bool seg_prev = cell_has_active_segment(active_segments_time,
                                                          cells_per_column,
                                                          max_segments_per_cell,
                                                          idx,
                                                          cell,
                                                          snap.timestep - 1);
            had_prev_prediction = had_prev_prediction || pred_prev;
            had_prev_segment = had_prev_segment || seg_prev;
            had_prev_prediction_and_segment = had_prev_prediction_and_segment || (pred_prev && seg_prev);
          }

          if (had_prev_prediction_and_segment) {
            ++new_burst_had_prev_prediction_and_segment;
          } else if (had_prev_prediction) {
            ++new_burst_prev_prediction_without_segment;
          } else if (had_prev_segment) {
            ++new_burst_prev_segment_without_prediction;
          } else {
            ++new_burst_no_prev_prediction;
          }
        }
      }
      if (has_prediction) {
        ++predictive;
      }
      if (masks.learning != 0) {
        ++learning;
      }
    }

    ++samples;
    active_columns += static_cast<double>(active);
    multi_active_fraction += static_cast<double>(multi_active) / static_cast<double>(active);
    true_burst_fraction += static_cast<double>(true_bursting) / static_cast<double>(active);
    new_true_burst_fraction += static_cast<double>(new_true_bursting) / static_cast<double>(active);
    repeated_true_burst_fraction += static_cast<double>(repeated_true_bursting) / static_cast<double>(active);
    predicted_multi_active_fraction += static_cast<double>(predicted_multi_active) / static_cast<double>(active);
    predictive_fraction += static_cast<double>(predictive) / static_cast<double>(active);
    learning_fraction += static_cast<double>(learning) / static_cast<double>(active);
    prev_true_bursting = std::move(current_true_bursting);
  }

  void print(const std::string& label, int layer_idx) const {
    if (samples <= 0) {
      return;
    }
    const double denom = static_cast<double>(samples);
    std::cout << label << " layer=" << layer_idx
              << " samples=" << samples
              << " active_cols=" << (active_columns / denom)
              << " multi_active_fraction=" << (multi_active_fraction / denom)
              << " true_burst_fraction=" << (true_burst_fraction / denom)
              << " new_true_burst_fraction=" << (new_true_burst_fraction / denom)
              << " repeated_true_burst_fraction=" << (repeated_true_burst_fraction / denom)
              << " predicted_multi_active_fraction=" << (predicted_multi_active_fraction / denom)
              << " predictive_fraction=" << (predictive_fraction / denom)
              << " learning_fraction=" << (learning_fraction / denom)
              << "\n";
    if (new_true_bursts > 0) {
      std::cout << label
                << " new_true_burst_causes"
                << " total=" << new_true_bursts
                << " no_prev_prediction=" << new_burst_no_prev_prediction
                << " prev_prediction_without_segment=" << new_burst_prev_prediction_without_segment
                << " prev_segment_without_prediction=" << new_burst_prev_segment_without_prediction
                << " had_prev_prediction_and_segment=" << new_burst_had_prev_prediction_and_segment
                << "\n";
    }

    std::vector<std::pair<int, int>> counts;
    counts.reserve(true_burst_counts.size());
    for (std::size_t col = 0; col < true_burst_counts.size(); ++col) {
      if (true_burst_counts[col] > 0) {
        counts.push_back({true_burst_counts[col], static_cast<int>(col)});
      }
    }
    std::sort(counts.begin(), counts.end(), std::greater<>());
    const std::size_t limit = std::min<std::size_t>(counts.size(), 8);
    if (limit > 0) {
      std::cout << label << " top_true_burst_columns=";
      for (std::size_t i = 0; i < limit; ++i) {
        if (i > 0) {
          std::cout << ",";
        }
        std::cout << counts[i].second << ":" << counts[i].first;
      }
      std::cout << "\n";
    }

    std::vector<std::pair<int, int>> new_counts;
    new_counts.reserve(new_true_burst_counts.size());
    for (std::size_t col = 0; col < new_true_burst_counts.size(); ++col) {
      if (new_true_burst_counts[col] > 0) {
        new_counts.push_back({new_true_burst_counts[col], static_cast<int>(col)});
      }
    }
    std::sort(new_counts.begin(), new_counts.end(), std::greater<>());
    const std::size_t new_limit = std::min<std::size_t>(new_counts.size(), 8);
    if (new_limit > 0) {
      std::cout << label << " top_new_true_burst_columns=";
      for (std::size_t i = 0; i < new_limit; ++i) {
        if (i > 0) {
          std::cout << ",";
        }
        std::cout << new_counts[i].second << ":" << new_counts[i].first;
      }
      std::cout << "\n";
    }

    std::vector<std::pair<int, int>> pos_counts;
    pos_counts.reserve(position_new_true_burst_counts.size());
    for (std::size_t pos = 0; pos < position_new_true_burst_counts.size(); ++pos) {
      if (position_new_true_burst_counts[pos] > 0) {
        pos_counts.push_back({position_new_true_burst_counts[pos], static_cast<int>(pos)});
      }
    }
    std::sort(pos_counts.begin(), pos_counts.end(), std::greater<>());
    const std::size_t pos_limit = std::min<std::size_t>(pos_counts.size(), 10);
    if (pos_limit > 0) {
      std::cout << label << " top_new_true_burst_positions=";
      for (std::size_t i = 0; i < pos_limit; ++i) {
        const int pos = pos_counts[i].second;
        if (i > 0) {
          std::cout << ",";
        }
        std::cout << pos << ":" << pos_counts[i].first;
        if (pos >= 0 && pos < static_cast<int>(position_samples.size())) {
          std::cout << "/" << position_samples[static_cast<std::size_t>(pos)];
        }
      }
      std::cout << "\n";
    }
  }
};

chat_htm::ScalarEncoder::Params parse_scalar_encoder_params(const std::string& config_path,
                                                            int layer0_input_bits) {
  chat_htm::ScalarEncoder::Params p;
  p.n = layer0_input_bits;  // must match Layer 0 input size

  try {
    YAML::Node root = YAML::LoadFile(config_path);
    if (root["encoder"]) {
      const auto& enc = root["encoder"];
      if (enc["active_bits"]) p.w = enc["active_bits"].as<int>();
      if (enc["min_value"]) p.min_val = enc["min_value"].as<int>();
      if (enc["max_value"]) p.max_val = enc["max_value"].as<int>();
    }
  } catch (const YAML::Exception& e) {
    std::cerr << "Warning: could not parse encoder section: " << e.what() << "\n";
  }

  return p;
}

chat_htm::WordRowEncoder::Params parse_word_row_encoder_params(const std::string& config_path,
                                                               int input_rows, int input_cols) {
  chat_htm::WordRowEncoder::Params p;
  p.rows = input_rows;
  p.cols = input_cols;
  try {
    YAML::Node root = YAML::LoadFile(config_path);
    if (root["encoder"]) {
      const auto& enc = root["encoder"];
      if (enc["letter_bits"]) p.letter_bits = enc["letter_bits"].as<int>();
      if (enc["alphabet"]) p.alphabet = enc["alphabet"].as<std::string>();
    }
  } catch (const YAML::Exception& e) {
    std::cerr << "Warning: could not parse word-row encoder section: " << e.what() << "\n";
  }
  return p;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string input_file;
  std::string config_file;
  int steps = -1;    // -1 means "whole file"
  int epochs = 1;
  bool use_gui = false;
  bool log = false;
  std::string cli_theme;

  // --- Parse arguments ---
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (arg == "--list-configs") {
      std::cout << "Available YAML configs in configs/:\n";
      auto files = htm_flow::list_config_files("configs");
      if (files.empty()) {
        std::cout << "  (none found)\n";
      } else {
        for (const auto& f : files) {
          std::cout << "  " << f << "\n";
        }
      }
      return 0;
    }
    if (arg == "--input") {
      if (i + 1 >= argc) { std::cerr << "--input requires a file path\n"; return 2; }
      input_file = argv[++i];
      continue;
    }
    if (arg == "--config") {
      if (i + 1 >= argc) { std::cerr << "--config requires a file path\n"; return 2; }
      config_file = argv[++i];
      continue;
    }
    if (arg == "--steps") {
      if (i + 1 >= argc) { std::cerr << "--steps requires a number\n"; return 2; }
      steps = std::atoi(argv[++i]);
      continue;
    }
    if (arg == "--epochs") {
      if (i + 1 >= argc) { std::cerr << "--epochs requires a number\n"; return 2; }
      epochs = std::atoi(argv[++i]);
      continue;
    }
    if (arg == "--theme") {
      if (i + 1 >= argc) { std::cerr << "--theme requires a value: light|dark\n"; return 2; }
      cli_theme = argv[++i];
      continue;
    }
    if (arg == "--gui") { use_gui = true; continue; }
    if (arg == "--log") { log = true; continue; }

    std::cerr << "Unknown argument: " << arg << "\n";
    usage(argv[0]);
    return 2;
  }

  if (input_file.empty() || config_file.empty()) {
    std::cerr << "Error: --input and --config are required.\n\n";
    usage(argv[0]);
    return 2;
  }

  // --- Load configuration ---
  htm_flow::HTMRegionConfig region_cfg;
  std::vector<htm_flow::RuntimeParameterScheduleEntry> runtime_schedule;
  try {
    region_cfg = htm_flow::load_region_config(config_file);
    runtime_schedule = htm_flow::load_runtime_parameter_schedule(config_file);
  } catch (const std::exception& e) {
    std::cerr << "Error loading config: " << e.what() << "\n";
    return 1;
  }

  // Apply logging setting
  for (auto& layer_cfg : region_cfg.layers) {
    layer_cfg.log_timings = log;
  }

  // Compute Layer 0 input size
  const int input_rows = region_cfg.layers[0].num_input_rows;
  const int input_cols = region_cfg.layers[0].num_input_cols;
  const int input_bits = input_rows * input_cols;
  const TextMode text_mode = parse_text_mode(config_file);
  const std::string yaml_theme = parse_gui_theme(config_file);
  const std::string effective_theme = cli_theme.empty() ? yaml_theme : cli_theme;

  std::cout << "Config:  " << config_file << " (" << region_cfg.layers.size() << " layer"
            << (region_cfg.layers.size() > 1 ? "s" : "") << ")\n";
  std::cout << "Input:   " << input_file << "\n";
  std::unique_ptr<chat_htm::TextRuntime> runtime;
  std::string name = std::filesystem::path(config_file).stem().string();
  try {
    if (text_mode == TextMode::WordRows) {
      auto enc_params = parse_word_row_encoder_params(config_file, input_rows, input_cols);
      chat_htm::WordRowEncoder encoder(enc_params);
      auto chunker = std::make_unique<chat_htm::WordChunker>(input_file);
      runtime = std::make_unique<chat_htm::TextRuntime>(
          region_cfg, std::move(chunker), encoder, name);
      runtime->set_runtime_schedule(runtime_schedule);
      std::cout << "Mode:    word_rows\n";
      std::cout << "Encoder: rows=" << enc_params.rows
                << " cols=" << enc_params.cols
                << " letter_bits=" << enc_params.letter_bits
                << " alphabet_size=" << enc_params.alphabet.size() << "\n";
      std::cout << "Text:    " << runtime->input_size() << " words\n\n";
    } else {
      auto enc_params = parse_scalar_encoder_params(config_file, input_bits);
      chat_htm::ScalarEncoder encoder(enc_params);
      auto chunker = std::make_unique<chat_htm::TextChunker>(input_file);
      runtime = std::make_unique<chat_htm::TextRuntime>(
          region_cfg, std::move(chunker), encoder, name);
      runtime->set_runtime_schedule(runtime_schedule);
      std::cout << "Mode:    character\n";
      std::cout << "Encoder: n=" << enc_params.n << " w=" << enc_params.w
                << " range=[" << enc_params.min_val << "," << enc_params.max_val << "]\n";
      std::cout << "Text:    " << runtime->input_size() << " characters\n\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error creating runtime: " << e.what() << "\n";
    return 1;
  }

  // Compute total steps
  int total_steps = steps;
  if (total_steps < 0) {
    total_steps = static_cast<int>(runtime->input_size()) * epochs;
  }

  // Enable per-step text logging (works in both GUI and headless modes)
  if (log) {
    runtime->set_log_text(true);
  }

  // --- GUI mode ---
  if (use_gui) {
#ifdef HTM_FLOW_WITH_GUI
    std::cout << "Launching GUI debugger...\n";
    htm_gui::DebuggerOptions opts;
    opts.window_title = name;
    opts.theme = effective_theme;
    return htm_gui::run_debugger(argc, argv, *runtime, opts);
#else
    std::cerr << "This binary was built without GUI support.\n"
              << "Use the container-based GUI instead (no local Qt6 needed):\n"
              << "  ./run_gui.sh --input " << input_file
              << " --config " << config_file << "\n"
              << "\n"
              << "Or rebuild natively with Qt6 installed:\n"
              << "  ./build.sh Release GUI\n";
    return 2;
#endif
  }

  // --- Headless mode ---
  int log_interval = std::max(1, total_steps / 20);  // Log ~20 times
  const int measured_layer = (runtime->num_layers() > 1) ? 1 : 0;
  const int first_override_timestep =
      runtime_schedule.empty() ? -1 : runtime_schedule.front().at_timestep;
  LayerMetrics pre_override_metrics;
  LayerMetrics post_override_metrics;
  for (int i = 0; i < total_steps; ++i) {
    try {
      runtime->step(1);
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return 1;
    }

    const auto& measured = runtime->region().layer(measured_layer);
    const auto layer_snapshot = measured.snapshot();
    const auto& measured_cfg = measured.config();
    const int input_size = static_cast<int>(runtime->input_size());
    const int input_position =
        (input_size > 0 && runtime->input_total_steps() > 0)
            ? static_cast<int>((runtime->input_total_steps() - 1) % static_cast<std::size_t>(input_size))
            : -1;
    if (first_override_timestep >= 0 && runtime->timestep() > first_override_timestep) {
      post_override_metrics.add(layer_snapshot,
                                measured.burst_columns_time(),
                                measured.predict_cells_time(),
                                measured.active_segments_time(),
                                measured_cfg.cells_per_column,
                                measured_cfg.max_segments_per_cell,
                                input_position,
                                input_size);
    } else {
      pre_override_metrics.add(layer_snapshot,
                               measured.burst_columns_time(),
                               measured.predict_cells_time(),
                               measured.active_segments_time(),
                               measured_cfg.cells_per_column,
                               measured_cfg.max_segments_per_cell,
                               input_position,
                               input_size);
    }

    if (log && (i % log_interval == 0 || i == total_steps - 1)) {
      std::cout << "Step " << (i + 1) << "/" << total_steps
                << "  epoch=" << runtime->input_epoch()
                << "  accuracy=" << (runtime->prediction_accuracy() * 100.0) << "%"
                << "  | " << runtime->input_context()
                << "\n";
    }
  }

  std::cout << "\nDone. " << total_steps << " steps processed.\n";
  std::cout << "Final prediction accuracy: "
            << (runtime->prediction_accuracy() * 100.0) << "%\n";
  pre_override_metrics.print("Pre-override metrics", measured_layer);
  post_override_metrics.print("Post-override metrics", measured_layer);

  return 0;
}
