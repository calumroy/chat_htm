#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace chat_htm {

/// Encodes a word into a row-wise SDR.
///
/// For each row (character position), one letter-specific non-overlapping
/// bit block is activated in that row. Rows beyond the word length stay zero.
class WordRowEncoder {
public:
  struct Params {
    int rows = 5;
    int cols = 108;
    int letter_bits = 4;
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  };

  explicit WordRowEncoder(const Params& p) : params_(p) { validate(); }

  std::vector<int> encode(const std::string& word) const {
    std::vector<int> sdr(static_cast<std::size_t>(params_.rows * params_.cols), 0);
    for (int r = 0; r < params_.rows; ++r) {
      if (r >= static_cast<int>(word.size())) break;
      const int bucket = bucket_for_char(word[static_cast<std::size_t>(r)]);
      const int start_col = bucket * params_.letter_bits;
      const int row_offset = r * params_.cols;
      for (int i = 0; i < params_.letter_bits; ++i) {
        sdr[static_cast<std::size_t>(row_offset + start_col + i)] = 1;
      }
    }
    return sdr;
  }

  const Params& params() const { return params_; }
  int total_bits() const { return params_.rows * params_.cols; }

private:
  void validate() const {
    if (params_.rows <= 0) throw std::invalid_argument("WordRowEncoder: rows must be > 0");
    if (params_.cols <= 0) throw std::invalid_argument("WordRowEncoder: cols must be > 0");
    if (params_.letter_bits <= 0)
      throw std::invalid_argument("WordRowEncoder: letter_bits must be > 0");
    if (params_.alphabet.empty())
      throw std::invalid_argument("WordRowEncoder: alphabet must not be empty");
    const int required_cols = params_.letter_bits * (static_cast<int>(params_.alphabet.size()) + 1);
    if (params_.cols != required_cols) {
      throw std::invalid_argument("WordRowEncoder: cols must equal letter_bits * (alphabet_size + 1)");
    }
  }

  int bucket_for_char(char c) const {
    unsigned char uc = static_cast<unsigned char>(c);
    char lc = static_cast<char>(std::tolower(uc));
    auto idx = params_.alphabet.find(lc);
    if (idx == std::string::npos) return static_cast<int>(params_.alphabet.size());
    return static_cast<int>(idx);
  }

  Params params_;
};

}  // namespace chat_htm
