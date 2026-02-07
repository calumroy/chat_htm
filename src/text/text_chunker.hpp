#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace chat_htm {

/// Reads a text file and yields one character at a time.
///
/// The chunker pre-loads the entire file into memory so that repeated
/// iteration (multi-epoch training) is fast.  Call `next()` to get the
/// ASCII value of the current character and advance the cursor.  When the
/// end of file is reached the chunker wraps around and increments the
/// epoch counter.
class TextChunker {
public:
  explicit TextChunker(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error("TextChunker: cannot open file: " + path);
    }
    text_ = std::string((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    if (text_.empty()) {
      throw std::runtime_error("TextChunker: file is empty: " + path);
    }
    path_ = path;
  }

  /// Construct from an in-memory string (useful for tests).
  static TextChunker from_string(const std::string& text) {
    TextChunker tc;
    tc.text_ = text;
    tc.path_ = "<memory>";
    if (text.empty()) {
      throw std::invalid_argument("TextChunker::from_string: text must not be empty");
    }
    return tc;
  }

  /// Return the ASCII value of the current character and advance.
  /// Wraps around to the beginning when the end of the text is reached.
  int next() {
    int value = static_cast<unsigned char>(text_[pos_]);
    ++pos_;
    ++total_steps_;
    if (pos_ >= text_.size()) {
      pos_ = 0;
      ++epoch_;
    }
    return value;
  }

  /// Peek at the current character without advancing.
  int peek() const {
    return static_cast<unsigned char>(text_[pos_]);
  }

  /// Peek at the character at an arbitrary offset from the current position.
  /// Wraps around the text boundary.
  int peek_at(int offset) const {
    auto idx = (pos_ + static_cast<std::size_t>(offset)) % text_.size();
    return static_cast<unsigned char>(text_[idx]);
  }

  /// Reset to the beginning.
  void reset() {
    pos_ = 0;
    epoch_ = 0;
    total_steps_ = 0;
  }

  /// Number of characters in the loaded text.
  std::size_t size() const { return text_.size(); }

  /// Current position within the text (0-based).
  std::size_t position() const { return pos_; }

  /// How many complete passes through the text have been made.
  int epoch() const { return epoch_; }

  /// Total number of characters yielded since construction / last reset.
  std::size_t total_steps() const { return total_steps_; }

  /// The file path (or "<memory>" for from_string).
  const std::string& path() const { return path_; }

  /// The full loaded text.
  const std::string& text() const { return text_; }

private:
  TextChunker() = default;

  std::string text_;
  std::string path_;
  std::size_t pos_{0};
  int epoch_{0};
  std::size_t total_steps_{0};
};

}  // namespace chat_htm
