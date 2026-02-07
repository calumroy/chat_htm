#include <gtest/gtest.h>

#include "text/text_chunker.hpp"

using chat_htm::TextChunker;

// ---------------------------------------------------------------------------
// Basic construction
// ---------------------------------------------------------------------------

TEST(TextChunker, FromString) {
  auto tc = TextChunker::from_string("abc");
  EXPECT_EQ(tc.size(), 3u);
  EXPECT_EQ(tc.position(), 0u);
  EXPECT_EQ(tc.epoch(), 0);
}

TEST(TextChunker, FromFile) {
  std::string path = CHAT_HTM_TEST_DATA_DIR "/hello_world.txt";
  TextChunker tc(path);
  EXPECT_GT(tc.size(), 0u);
  EXPECT_EQ(tc.path(), path);
}

TEST(TextChunker, ThrowsOnEmptyString) {
  EXPECT_THROW(TextChunker::from_string(""), std::invalid_argument);
}

TEST(TextChunker, ThrowsOnMissingFile) {
  EXPECT_THROW(TextChunker("/nonexistent/path/file.txt"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Character iteration
// ---------------------------------------------------------------------------

TEST(TextChunker, NextReturnsCorrectCharacters) {
  auto tc = TextChunker::from_string("Hi!");
  EXPECT_EQ(tc.next(), 'H');
  EXPECT_EQ(tc.next(), 'i');
  EXPECT_EQ(tc.next(), '!');
}

TEST(TextChunker, PeekDoesNotAdvance) {
  auto tc = TextChunker::from_string("xy");
  EXPECT_EQ(tc.peek(), 'x');
  EXPECT_EQ(tc.peek(), 'x');  // still at same position
  EXPECT_EQ(tc.position(), 0u);
}

TEST(TextChunker, PositionAdvancesOnNext) {
  auto tc = TextChunker::from_string("abc");
  EXPECT_EQ(tc.position(), 0u);
  tc.next();
  EXPECT_EQ(tc.position(), 1u);
  tc.next();
  EXPECT_EQ(tc.position(), 2u);
}

// ---------------------------------------------------------------------------
// Wrapping / epochs
// ---------------------------------------------------------------------------

TEST(TextChunker, WrapsAroundAtEnd) {
  auto tc = TextChunker::from_string("ab");
  EXPECT_EQ(tc.next(), 'a');
  EXPECT_EQ(tc.next(), 'b');
  // Should wrap to beginning
  EXPECT_EQ(tc.epoch(), 1);
  EXPECT_EQ(tc.position(), 0u);
  EXPECT_EQ(tc.next(), 'a');
}

TEST(TextChunker, EpochIncrementsCorrectly) {
  auto tc = TextChunker::from_string("x");
  tc.next();  // epoch becomes 1
  EXPECT_EQ(tc.epoch(), 1);
  tc.next();  // epoch becomes 2
  EXPECT_EQ(tc.epoch(), 2);
}

TEST(TextChunker, TotalStepsCounts) {
  auto tc = TextChunker::from_string("ab");
  tc.next();
  tc.next();
  tc.next();  // wraps
  EXPECT_EQ(tc.total_steps(), 3u);
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

TEST(TextChunker, ResetGoesBackToStart) {
  auto tc = TextChunker::from_string("hello");
  tc.next();
  tc.next();
  tc.next();
  tc.reset();
  EXPECT_EQ(tc.position(), 0u);
  EXPECT_EQ(tc.epoch(), 0);
  EXPECT_EQ(tc.total_steps(), 0u);
  EXPECT_EQ(tc.next(), 'h');
}

// ---------------------------------------------------------------------------
// Peek at offset
// ---------------------------------------------------------------------------

TEST(TextChunker, PeekAtOffset) {
  auto tc = TextChunker::from_string("abcde");
  EXPECT_EQ(tc.peek_at(0), 'a');
  EXPECT_EQ(tc.peek_at(2), 'c');
  EXPECT_EQ(tc.peek_at(5), 'a');  // wraps around
}
