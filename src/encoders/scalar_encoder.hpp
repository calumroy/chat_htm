#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace chat_htm {

/// Encodes a scalar value into a Sparse Distributed Representation (SDR).
///
/// The encoder maps a numeric value within [min_val, max_val] to a binary
/// vector of length `n`, where exactly `w` contiguous bits are set to 1.
/// As the input value increases, the active-bit window slides from left to
/// right.  Nearby values share overlapping active bits, giving the HTM
/// spatial pooler a notion of semantic similarity.
///
/// Example (n=20, w=5, range 0–9):
///   encode(0) -> 11111 00000 00000 00000
///   encode(1) -> 01111 10000 00000 00000
///   encode(9) -> 00000 00000 00000 11111
class ScalarEncoder {
public:
  struct Params {
    int n = 400;         ///< Total number of bits in the output SDR.
    int w = 21;          ///< Number of active (1) bits per encoding.
    int min_val = 0;     ///< Minimum input value (inclusive).
    int max_val = 127;   ///< Maximum input value (inclusive).
  };

  ScalarEncoder() : ScalarEncoder(Params{}) {}

  explicit ScalarEncoder(const Params& p) : params_(p) {
    validate();
    // Number of discrete bucket positions.  The active window can start at
    // positions 0 .. (n - w).
    num_buckets_ = params_.n - params_.w;
    range_ = params_.max_val - params_.min_val;
  }

  /// Encode a scalar value into a binary SDR of length `n`.
  /// Values outside [min_val, max_val] are clamped.
  std::vector<int> encode(int value) const {
    // Clamp to valid range.
    if (value < params_.min_val) value = params_.min_val;
    if (value > params_.max_val) value = params_.max_val;

    // Compute the starting position of the active window.
    int start = 0;
    if (range_ > 0) {
      start = static_cast<int>(
          static_cast<double>(value - params_.min_val) / range_ * num_buckets_ + 0.5);
    }
    if (start > num_buckets_) start = num_buckets_;

    std::vector<int> sdr(static_cast<std::size_t>(params_.n), 0);
    for (int i = start; i < start + params_.w; ++i) {
      sdr[static_cast<std::size_t>(i)] = 1;
    }
    return sdr;
  }

  /// Return the number of active bits that two encoded values share.
  /// Useful for verifying semantic overlap.
  int overlap(int val_a, int val_b) const {
    auto a = encode(val_a);
    auto b = encode(val_b);
    int count = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a[i] && b[i]) ++count;
    }
    return count;
  }

  const Params& params() const { return params_; }
  int total_bits() const { return params_.n; }
  int active_bits() const { return params_.w; }

private:
  void validate() const {
    if (params_.n <= 0)
      throw std::invalid_argument("ScalarEncoder: n must be > 0");
    if (params_.w <= 0)
      throw std::invalid_argument("ScalarEncoder: w must be > 0");
    if (params_.w > params_.n)
      throw std::invalid_argument("ScalarEncoder: w must be <= n");
    if (params_.max_val < params_.min_val)
      throw std::invalid_argument("ScalarEncoder: max_val must be >= min_val");
  }

  Params params_;
  int num_buckets_{0};
  double range_{0.0};
};

}  // namespace chat_htm
