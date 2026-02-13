#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

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
  try {
    region_cfg = htm_flow::load_region_config(config_file);
  } catch (const std::exception& e) {
    std::cerr << "Error loading config: " << e.what() << "\n";
    return 1;
  }

  // Apply logging setting
  for (auto& layer_cfg : region_cfg.layers) {
    layer_cfg.log_timings = (!use_gui) || log;
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
  for (int i = 0; i < total_steps; ++i) {
    runtime->step(1);

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

  return 0;
}
