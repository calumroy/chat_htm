#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <htm_gui/runtime.hpp>

#include <htm_flow/config.hpp>
#include <htm_flow/htm_region.hpp>

#include "encoders/scalar_encoder.hpp"
#include "encoders/word_row_encoder.hpp"
#include "text/text_chunker.hpp"
#include "text/word_chunker.hpp"

namespace chat_htm {

/// IHtmRuntime implementation that feeds text characters to an HTMRegion.
///
/// Each call to step() reads the next character from a TextChunker, encodes
/// it via ScalarEncoder, and passes the resulting SDR to the HTMRegion.
///
/// Implements the full IHtmRuntime interface so the htm_gui debugger can be
/// used for visualizing how the network processes text.
class TextRuntime : public htm_gui::IHtmRuntime {
public:
  enum class InputMode {
    Character,
    WordRows
  };

  /// Construct a TextRuntime with an existing TextChunker and encoder.
  /// @param cfg     HTM region configuration (loaded from YAML)
  /// @param chunker Text source (takes ownership)
  /// @param encoder Scalar encoder for character values
  /// @param name    Friendly name for display
  TextRuntime(const htm_flow::HTMRegionConfig& cfg,
              std::unique_ptr<TextChunker> chunker,
              const ScalarEncoder& encoder,
              const std::string& name = "chat_htm");
  TextRuntime(const htm_flow::HTMRegionConfig& cfg,
              std::unique_ptr<WordChunker> chunker,
              const WordRowEncoder& encoder,
              const std::string& name = "chat_htm");

  // --- IHtmRuntime interface (delegates to the active layer) ---
  htm_gui::Snapshot snapshot() const override;
  void step(int n = 1) override;
  htm_gui::ProximalSynapseQuery query_proximal(int column_x, int column_y) const override;
  int num_segments(int column_x, int column_y, int cell) const override;
  htm_gui::DistalSynapseQuery query_distal(int column_x, int column_y, int cell, int segment) const override;
  std::vector<htm_gui::InputSequence> input_sequences() const override;
  int input_sequence() const override { return 0; }
  void set_input_sequence(int /*id*/) override {}
  int activation_threshold() const override;
  std::string name() const override;

  // --- Layer selection for GUI ---
  std::vector<htm_gui::InputSequence> layer_options() const override;
  int num_layers() const override;
  int active_layer() const override { return active_layer_idx_; }
  void set_active_layer(int idx) override;

  // --- Text-specific accessors ---
  const TextChunker& chunker() const { return *chunker_; }
  const WordChunker& word_chunker() const { return *word_chunker_; }
  const ScalarEncoder& encoder() const { return encoder_; }
  const WordRowEncoder& word_encoder() const { return word_encoder_; }
  htm_flow::HTMRegion& region() { return *region_; }
  const htm_flow::HTMRegion& region() const { return *region_; }
  InputMode input_mode() const { return input_mode_; }
  std::size_t input_size() const;
  int input_epoch() const;
  std::size_t input_total_steps() const;
  std::string input_context() const;

  /// The character that was most recently fed to the network.
  char last_char() const { return last_char_; }
  const std::string& last_word() const { return last_word_; }

  /// Enable/disable per-step text input logging.
  /// When enabled, each step() prints the current text context to stdout.
  void set_log_text(bool enabled) { log_text_ = enabled; }
  bool log_text() const { return log_text_; }

  /// Cumulative prediction accuracy (fraction of steps where the HTM
  /// predicted the correct next column activation pattern).
  double prediction_accuracy() const;

private:
  /// Print a readable character (replace control chars with spaces/dots).
  static char printable(char c);
  /// Build a context string showing surrounding text with current char highlighted.
  std::string text_context() const;
  /// Build a context string showing surrounding words with current word highlighted.
  std::string word_context() const;

  std::unique_ptr<htm_flow::HTMRegion> region_;
  std::unique_ptr<TextChunker> chunker_;
  std::unique_ptr<WordChunker> word_chunker_;
  ScalarEncoder encoder_;
  WordRowEncoder word_encoder_;
  InputMode input_mode_{InputMode::Character};
  std::string name_;
  int active_layer_idx_{0};
  bool log_text_{false};

  char last_char_{'\0'};
  std::string last_word_;
  int correct_predictions_{0};
  int total_predictions_{0};
};

}  // namespace chat_htm
