#include "diagnostics/layer_metrics.hpp"

#include <algorithm>
#include <iostream>
#include <utility>

namespace chat_htm::diagnostics {

namespace {

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

}  // namespace

void LayerMetrics::add(const htm_gui::Snapshot& snap,
                       const std::vector<int>& burst_cols_time,
                       const std::vector<int>& predict_cells_time,
                       const std::vector<int>& active_segments_time,
                       int cells_per_column,
                       int max_segments_per_cell,
                       int input_position,
                       int input_size) {
  const int active = static_cast<int>(snap.active_column_indices.size());
  if (active <= 0) {
    prev_true_bursting_.assign(snap.column_cell_masks.size(), 0);
    return;
  }
  if (input_size > 0 && static_cast<int>(position_samples_.size()) != input_size) {
    position_samples_.assign(static_cast<std::size_t>(input_size), 0);
    position_new_true_burst_counts_.assign(static_cast<std::size_t>(input_size), 0);
  }
  if (input_position >= 0 && input_position < static_cast<int>(position_samples_.size())) {
    ++position_samples_[static_cast<std::size_t>(input_position)];
  }
  if (prev_true_bursting_.size() != snap.column_cell_masks.size()) {
    prev_true_bursting_.assign(snap.column_cell_masks.size(), 0);
  }
  if (true_burst_counts_.size() != snap.column_cell_masks.size()) {
    true_burst_counts_.assign(snap.column_cell_masks.size(), 0);
  }
  if (new_true_burst_counts_.size() != snap.column_cell_masks.size()) {
    new_true_burst_counts_.assign(snap.column_cell_masks.size(), 0);
  }

  int multi_active = 0;
  int true_bursting = 0;
  int new_true_bursting = 0;
  int repeated_true_bursting = 0;
  int predicted_multi_active = 0;
  int predictive = 0;
  int learning = 0;
  std::vector<std::uint8_t> current_true_bursting(snap.column_cell_masks.size(), 0);
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
      ++true_burst_counts_[static_cast<std::size_t>(idx)];
      if (prev_true_bursting_[static_cast<std::size_t>(idx)] == 1) {
        ++repeated_true_bursting;
      } else {
        ++new_true_bursting;
        ++new_true_bursts_;
        ++new_true_burst_counts_[static_cast<std::size_t>(idx)];
        if (input_position >= 0 && input_position < static_cast<int>(position_new_true_burst_counts_.size())) {
          ++position_new_true_burst_counts_[static_cast<std::size_t>(input_position)];
        }

        bool had_prev_prediction = false;
        bool had_prev_segment = false;
        bool had_prev_prediction_and_segment = false;
        for (int cell = 0; cell < cells_per_column; ++cell) {
          const bool pred_prev = cell_time_is_set(predict_cells_time, cells_per_column, idx, cell, snap.timestep - 1);
          const bool seg_prev = cell_has_active_segment(
              active_segments_time, cells_per_column, max_segments_per_cell, idx, cell, snap.timestep - 1);
          had_prev_prediction = had_prev_prediction || pred_prev;
          had_prev_segment = had_prev_segment || seg_prev;
          had_prev_prediction_and_segment = had_prev_prediction_and_segment || (pred_prev && seg_prev);
        }

        if (had_prev_prediction_and_segment) {
          ++new_burst_had_prev_prediction_and_segment_;
        } else if (had_prev_prediction) {
          ++new_burst_prev_prediction_without_segment_;
        } else if (had_prev_segment) {
          ++new_burst_prev_segment_without_prediction_;
        } else {
          ++new_burst_no_prev_prediction_;
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

  ++samples_;
  active_columns_ += static_cast<double>(active);
  multi_active_fraction_ += static_cast<double>(multi_active) / static_cast<double>(active);
  true_burst_fraction_ += static_cast<double>(true_bursting) / static_cast<double>(active);
  new_true_burst_fraction_ += static_cast<double>(new_true_bursting) / static_cast<double>(active);
  repeated_true_burst_fraction_ += static_cast<double>(repeated_true_bursting) / static_cast<double>(active);
  predicted_multi_active_fraction_ += static_cast<double>(predicted_multi_active) / static_cast<double>(active);
  predictive_fraction_ += static_cast<double>(predictive) / static_cast<double>(active);
  learning_fraction_ += static_cast<double>(learning) / static_cast<double>(active);
  prev_true_bursting_ = std::move(current_true_bursting);
}

void LayerMetrics::print(std::ostream& out, const std::string& label, int layer_idx) const {
  if (samples_ <= 0) {
    return;
  }
  const double denom = static_cast<double>(samples_);
  out << label << " layer=" << layer_idx
      << " samples=" << samples_
      << " active_cols=" << (active_columns_ / denom)
      << " multi_active_fraction=" << (multi_active_fraction_ / denom)
      << " true_burst_fraction=" << (true_burst_fraction_ / denom)
      << " new_true_burst_fraction=" << (new_true_burst_fraction_ / denom)
      << " repeated_true_burst_fraction=" << (repeated_true_burst_fraction_ / denom)
      << " predicted_multi_active_fraction=" << (predicted_multi_active_fraction_ / denom)
      << " predictive_fraction=" << (predictive_fraction_ / denom)
      << " learning_fraction=" << (learning_fraction_ / denom)
      << "\n";

  if (new_true_bursts_ > 0) {
    out << label
        << " new_true_burst_causes"
        << " total=" << new_true_bursts_
        << " no_prev_prediction=" << new_burst_no_prev_prediction_
        << " prev_prediction_without_segment=" << new_burst_prev_prediction_without_segment_
        << " prev_segment_without_prediction=" << new_burst_prev_segment_without_prediction_
        << " had_prev_prediction_and_segment=" << new_burst_had_prev_prediction_and_segment_
        << "\n";
  }

  std::vector<std::pair<int, int>> counts;
  counts.reserve(true_burst_counts_.size());
  for (std::size_t col = 0; col < true_burst_counts_.size(); ++col) {
    if (true_burst_counts_[col] > 0) {
      counts.push_back({true_burst_counts_[col], static_cast<int>(col)});
    }
  }
  std::sort(counts.begin(), counts.end(), std::greater<>());
  const std::size_t limit = std::min<std::size_t>(counts.size(), 8);
  if (limit > 0) {
    out << label << " top_true_burst_columns=";
    for (std::size_t i = 0; i < limit; ++i) {
      if (i > 0) {
        out << ",";
      }
      out << counts[i].second << ":" << counts[i].first;
    }
    out << "\n";
  }

  std::vector<std::pair<int, int>> new_counts;
  new_counts.reserve(new_true_burst_counts_.size());
  for (std::size_t col = 0; col < new_true_burst_counts_.size(); ++col) {
    if (new_true_burst_counts_[col] > 0) {
      new_counts.push_back({new_true_burst_counts_[col], static_cast<int>(col)});
    }
  }
  std::sort(new_counts.begin(), new_counts.end(), std::greater<>());
  const std::size_t new_limit = std::min<std::size_t>(new_counts.size(), 8);
  if (new_limit > 0) {
    out << label << " top_new_true_burst_columns=";
    for (std::size_t i = 0; i < new_limit; ++i) {
      if (i > 0) {
        out << ",";
      }
      out << new_counts[i].second << ":" << new_counts[i].first;
    }
    out << "\n";
  }

  std::vector<std::pair<int, int>> pos_counts;
  pos_counts.reserve(position_new_true_burst_counts_.size());
  for (std::size_t pos = 0; pos < position_new_true_burst_counts_.size(); ++pos) {
    if (position_new_true_burst_counts_[pos] > 0) {
      pos_counts.push_back({position_new_true_burst_counts_[pos], static_cast<int>(pos)});
    }
  }
  std::sort(pos_counts.begin(), pos_counts.end(), std::greater<>());
  const std::size_t pos_limit = std::min<std::size_t>(pos_counts.size(), 10);
  if (pos_limit > 0) {
    out << label << " top_new_true_burst_positions=";
    for (std::size_t i = 0; i < pos_limit; ++i) {
      const int pos = pos_counts[i].second;
      if (i > 0) {
        out << ",";
      }
      out << pos << ":" << pos_counts[i].first;
      if (pos >= 0 && pos < static_cast<int>(position_samples_.size())) {
        out << "/" << position_samples_[static_cast<std::size_t>(pos)];
      }
    }
    out << "\n";
  }
}

}  // namespace chat_htm::diagnostics
