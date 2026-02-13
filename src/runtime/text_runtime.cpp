#include "runtime/text_runtime.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace chat_htm {

TextRuntime::TextRuntime(const htm_flow::HTMRegionConfig& cfg,
                         std::unique_ptr<TextChunker> chunker,
                         const ScalarEncoder& encoder,
                         const std::string& name)
    : region_(std::make_unique<htm_flow::HTMRegion>(cfg, name)),
      chunker_(std::move(chunker)),
      encoder_(encoder),
      word_encoder_(WordRowEncoder::Params{}),
      input_mode_(InputMode::Character),
      name_(name) {
  if (!chunker_) {
    throw std::invalid_argument("TextRuntime: chunker must not be null");
  }
}

TextRuntime::TextRuntime(const htm_flow::HTMRegionConfig& cfg,
                         std::unique_ptr<WordChunker> chunker,
                         const WordRowEncoder& encoder,
                         const std::string& name)
    : region_(std::make_unique<htm_flow::HTMRegion>(cfg, name)),
      word_chunker_(std::move(chunker)),
      encoder_(ScalarEncoder::Params{}),
      word_encoder_(encoder),
      input_mode_(InputMode::WordRows),
      name_(name) {
  if (!word_chunker_) {
    throw std::invalid_argument("TextRuntime: word chunker must not be null");
  }
}

htm_gui::Snapshot TextRuntime::snapshot() const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return {};
  }
  return region_->layer(active_layer_idx_).snapshot();
}

void TextRuntime::step(int n) {
  if (!region_ || n <= 0) return;
  if (input_mode_ == InputMode::Character && !chunker_) return;
  if (input_mode_ == InputMode::WordRows && !word_chunker_) return;

  for (int i = 0; i < n; ++i) {
    if (total_predictions_ > 0 || region_->timestep() > 0) {
      const auto& layer0 = region_->layer(0);
      auto snap = layer0.snapshot();
      if (!snap.column_cell_masks.empty()) {
        int predicted_and_active = 0;
        int total_active = 0;
        for (int idx : snap.active_column_indices) {
          if (idx >= 0 && idx < static_cast<int>(snap.column_cell_masks.size())) {
            ++total_active;
            if (snap.column_cell_masks[static_cast<std::size_t>(idx)].predictive != 0) {
              ++predicted_and_active;
            }
          }
        }
        if (total_active > 0 && predicted_and_active > total_active / 2) {
          ++correct_predictions_;
        }
        ++total_predictions_;
      }
    }

    std::vector<int> sdr;
    if (input_mode_ == InputMode::Character) {
      if (!chunker_) return;
      int char_val = chunker_->next();
      last_char_ = static_cast<char>(char_val);
      sdr = encoder_.encode(char_val);
    } else {
      if (!word_chunker_) return;
      last_word_ = word_chunker_->next();
      sdr = word_encoder_.encode(last_word_);
    }
    region_->set_input(sdr);
    region_->step(1);

    // Log text context after each step if enabled.
    if (log_text_) {
      std::cout << "[text] step=" << region_->timestep()
                << "  epoch=" << input_epoch()
                << "  accuracy=" << std::fixed << std::setprecision(1)
                << (prediction_accuracy() * 100.0) << "%"
                << "  | " << input_context()
                << std::endl;
    }
  }
}

htm_gui::ProximalSynapseQuery TextRuntime::query_proximal(int column_x, int column_y) const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return {};
  }
  return region_->layer(active_layer_idx_).query_proximal(column_x, column_y);
}

int TextRuntime::num_segments(int column_x, int column_y, int cell) const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return 0;
  }
  return region_->layer(active_layer_idx_).num_segments(column_x, column_y, cell);
}

htm_gui::DistalSynapseQuery TextRuntime::query_distal(int column_x, int column_y,
                                                       int cell, int segment) const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return {};
  }
  return region_->layer(active_layer_idx_).query_distal(column_x, column_y, cell, segment);
}

std::vector<htm_gui::InputSequence> TextRuntime::input_sequences() const {
  if (input_mode_ == InputMode::Character && chunker_) {
    return {{0, "Text: " + chunker_->path()}};
  }
  if (word_chunker_) {
    return {{0, "Text: " + word_chunker_->path()}};
  }
  return {{0, "Text: <unknown>"}};
}

int TextRuntime::activation_threshold() const {
  if (!region_ || active_layer_idx_ < 0 || active_layer_idx_ >= num_layers()) {
    return 0;
  }
  return region_->layer(active_layer_idx_).activation_threshold();
}

std::string TextRuntime::name() const {
  return name_ + " (Layer " + std::to_string(active_layer_idx_) + "/"
         + std::to_string(num_layers()) + ")";
}

std::vector<htm_gui::InputSequence> TextRuntime::layer_options() const {
  std::vector<htm_gui::InputSequence> options;
  const int n = num_layers();
  options.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    options.push_back({i, "Layer " + std::to_string(i)});
  }
  return options;
}

int TextRuntime::num_layers() const {
  return region_ ? region_->num_layers() : 0;
}

void TextRuntime::set_active_layer(int idx) {
  if (idx >= 0 && idx < num_layers()) {
    active_layer_idx_ = idx;
  }
}

double TextRuntime::prediction_accuracy() const {
  if (total_predictions_ == 0) return 0.0;
  return static_cast<double>(correct_predictions_) / total_predictions_;
}

std::size_t TextRuntime::input_size() const {
  if (input_mode_ == InputMode::Character && chunker_) return chunker_->size();
  if (word_chunker_) return word_chunker_->size();
  return 0;
}

int TextRuntime::input_epoch() const {
  if (input_mode_ == InputMode::Character && chunker_) return chunker_->epoch();
  if (word_chunker_) return word_chunker_->epoch();
  return 0;
}

std::size_t TextRuntime::input_total_steps() const {
  if (input_mode_ == InputMode::Character && chunker_) return chunker_->total_steps();
  if (word_chunker_) return word_chunker_->total_steps();
  return 0;
}

std::string TextRuntime::input_context() const {
  if (input_mode_ == InputMode::Character && chunker_) return text_context();
  if (word_chunker_) return word_context();
  return {};
}

char TextRuntime::printable(char c) {
  if (c == '\n' || c == '\r' || c == '\t') return ' ';
  if (c < 32 || c > 126) return '.';
  return c;
}

std::string TextRuntime::text_context() const {
  if (!chunker_ || chunker_->text().empty()) return {};
  const auto& text = chunker_->text();
  auto pos = chunker_->position();
  // position() is already advanced past the char we just read,
  // so the char we just fed is at pos-1 (wrapping).
  std::size_t cur = (pos == 0) ? text.size() - 1 : pos - 1;
  const int ctx = 10;  // chars of context each side
  std::string result;
  for (int j = -ctx; j <= ctx; ++j) {
    std::size_t idx = (cur + text.size() + static_cast<std::size_t>(j)) % text.size();
    char c = printable(text[idx]);
    if (j == 0) {
      result += '[';
      result += c;
      result += ']';
    } else {
      result += c;
    }
  }
  return result;
}

std::string TextRuntime::word_context() const {
  if (!word_chunker_ || word_chunker_->words().empty()) return {};
  const auto& words = word_chunker_->words();
  auto pos = word_chunker_->position();
  std::size_t cur = (pos == 0) ? words.size() - 1 : pos - 1;
  const int ctx = 4;
  std::string result;
  for (int j = -ctx; j <= ctx; ++j) {
    std::size_t idx = (cur + words.size() + static_cast<std::size_t>(j)) % words.size();
    const std::string& w = words[idx];
    if (j == 0) {
      result += "[";
      result += w;
      result += "]";
    } else {
      result += w;
    }
    if (j < ctx) result += " ";
  }
  return result;
}

}  // namespace chat_htm
