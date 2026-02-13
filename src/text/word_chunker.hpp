#pragma once

#include <cctype>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace chat_htm {

/// Reads a text file and yields one normalized word at a time.
///
/// Words are sequences of [A-Za-z]. Input is lowercased and tokenized
/// once at construction for fast repeated epoch iteration.
class WordChunker {
public:
  explicit WordChunker(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error("WordChunker: cannot open file: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    path_ = path;
    words_ = tokenize(content);
    if (words_.empty()) {
      throw std::runtime_error("WordChunker: no words found in file: " + path);
    }
  }

  static WordChunker from_string(const std::string& text) {
    WordChunker wc;
    wc.path_ = "<memory>";
    wc.words_ = tokenize(text);
    if (wc.words_.empty()) {
      throw std::invalid_argument("WordChunker::from_string: text must contain words");
    }
    return wc;
  }

  const std::string& next() {
    const std::string& value = words_[pos_];
    ++pos_;
    ++total_steps_;
    if (pos_ >= words_.size()) {
      pos_ = 0;
      ++epoch_;
    }
    return value;
  }

  const std::string& peek() const { return words_[pos_]; }

  void reset() {
    pos_ = 0;
    epoch_ = 0;
    total_steps_ = 0;
  }

  std::size_t size() const { return words_.size(); }
  std::size_t position() const { return pos_; }
  int epoch() const { return epoch_; }
  std::size_t total_steps() const { return total_steps_; }
  const std::string& path() const { return path_; }
  const std::vector<std::string>& words() const { return words_; }

private:
  WordChunker() = default;

  static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string current;
    current.reserve(32);
    for (char c : text) {
      unsigned char uc = static_cast<unsigned char>(c);
      if (std::isalpha(uc)) {
        current.push_back(static_cast<char>(std::tolower(uc)));
      } else if (!current.empty()) {
        out.push_back(current);
        current.clear();
      }
    }
    if (!current.empty()) out.push_back(current);
    return out;
  }

  std::vector<std::string> words_;
  std::string path_;
  std::size_t pos_{0};
  int epoch_{0};
  std::size_t total_steps_{0};
};

}  // namespace chat_htm
