#include <gtest/gtest.h>

#include "encoders/scalar_encoder.hpp"

#include <algorithm>
#include <numeric>

using chat_htm::ScalarEncoder;

// ---------------------------------------------------------------------------
// Basic properties
// ---------------------------------------------------------------------------

TEST(ScalarEncoder, OutputLengthMatchesN) {
  ScalarEncoder::Params p{.n = 200, .w = 11, .min_val = 0, .max_val = 99};
  ScalarEncoder enc(p);
  auto sdr = enc.encode(50);
  EXPECT_EQ(static_cast<int>(sdr.size()), p.n);
}

TEST(ScalarEncoder, ActiveBitsCountMatchesW) {
  ScalarEncoder::Params p{.n = 400, .w = 21, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(p);

  for (int v = 0; v <= 127; ++v) {
    auto sdr = enc.encode(v);
    int active = std::accumulate(sdr.begin(), sdr.end(), 0);
    EXPECT_EQ(active, p.w) << "value=" << v;
  }
}

TEST(ScalarEncoder, BitsAreBinaryOnly) {
  ScalarEncoder enc;
  auto sdr = enc.encode(65);
  for (int bit : sdr) {
    EXPECT_TRUE(bit == 0 || bit == 1);
  }
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

TEST(ScalarEncoder, MinValueActivatesLeftmostBits) {
  ScalarEncoder::Params p{.n = 100, .w = 5, .min_val = 0, .max_val = 50};
  ScalarEncoder enc(p);
  auto sdr = enc.encode(0);

  // First w bits should be active.
  for (int i = 0; i < p.w; ++i) {
    EXPECT_EQ(sdr[static_cast<std::size_t>(i)], 1) << "bit " << i;
  }
}

TEST(ScalarEncoder, MaxValueActivatesRightmostBits) {
  ScalarEncoder::Params p{.n = 100, .w = 5, .min_val = 0, .max_val = 50};
  ScalarEncoder enc(p);
  auto sdr = enc.encode(50);

  // Last w bits should be active.
  for (int i = p.n - p.w; i < p.n; ++i) {
    EXPECT_EQ(sdr[static_cast<std::size_t>(i)], 1) << "bit " << i;
  }
}

TEST(ScalarEncoder, ClampsBelowMinValue) {
  ScalarEncoder::Params p{.n = 100, .w = 5, .min_val = 10, .max_val = 50};
  ScalarEncoder enc(p);
  auto sdr_clamped = enc.encode(-5);
  auto sdr_min = enc.encode(10);
  EXPECT_EQ(sdr_clamped, sdr_min);
}

TEST(ScalarEncoder, ClampsAboveMaxValue) {
  ScalarEncoder::Params p{.n = 100, .w = 5, .min_val = 10, .max_val = 50};
  ScalarEncoder enc(p);
  auto sdr_clamped = enc.encode(999);
  auto sdr_max = enc.encode(50);
  EXPECT_EQ(sdr_clamped, sdr_max);
}

// ---------------------------------------------------------------------------
// Semantic overlap
// ---------------------------------------------------------------------------

TEST(ScalarEncoder, AdjacentValuesShareBits) {
  ScalarEncoder::Params p{.n = 400, .w = 21, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(p);

  // Adjacent values should have significant overlap.
  int adj_overlap = enc.overlap(65, 66);
  EXPECT_GT(adj_overlap, 0);
  // They should share most bits (w - step), where step is small.
  EXPECT_GT(adj_overlap, p.w / 2);
}

TEST(ScalarEncoder, DistantValuesShareFewerBits) {
  ScalarEncoder::Params p{.n = 400, .w = 21, .min_val = 0, .max_val = 127};
  ScalarEncoder enc(p);

  int adj_overlap = enc.overlap(50, 51);
  int far_overlap = enc.overlap(0, 127);

  EXPECT_GT(adj_overlap, far_overlap);
}

TEST(ScalarEncoder, IdenticalValuesHaveFullOverlap) {
  ScalarEncoder enc;
  EXPECT_EQ(enc.overlap(42, 42), enc.active_bits());
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

TEST(ScalarEncoder, ThrowsOnInvalidParams) {
  EXPECT_THROW(ScalarEncoder(ScalarEncoder::Params{.n = 0}), std::invalid_argument);
  EXPECT_THROW(ScalarEncoder(ScalarEncoder::Params{.n = 10, .w = 0}), std::invalid_argument);
  EXPECT_THROW(ScalarEncoder(ScalarEncoder::Params{.n = 10, .w = 20}), std::invalid_argument);
  EXPECT_THROW(ScalarEncoder(ScalarEncoder::Params{.n = 10, .w = 5, .min_val = 100, .max_val = 50}),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Single-value range (min == max)
// ---------------------------------------------------------------------------

TEST(ScalarEncoder, SingleValueRange) {
  ScalarEncoder::Params p{.n = 50, .w = 5, .min_val = 42, .max_val = 42};
  ScalarEncoder enc(p);
  auto sdr = enc.encode(42);
  int active = std::accumulate(sdr.begin(), sdr.end(), 0);
  EXPECT_EQ(active, 5);
}
