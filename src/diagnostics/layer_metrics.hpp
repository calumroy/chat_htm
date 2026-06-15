#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include <htm_gui/snapshot.hpp>

namespace chat_htm::diagnostics {

class LayerMetrics {
public:
  void add(const htm_gui::Snapshot& snap,
           const std::vector<int>& burst_cols_time,
           const std::vector<int>& predict_cells_time,
           const std::vector<int>& active_segments_time,
           int cells_per_column,
           int max_segments_per_cell,
           int input_position,
           int input_size);

  void print(std::ostream& out, const std::string& label, int layer_idx) const;

private:
  int samples_ = 0;
  double active_columns_ = 0.0;
  double multi_active_fraction_ = 0.0;
  double true_burst_fraction_ = 0.0;
  double new_true_burst_fraction_ = 0.0;
  double repeated_true_burst_fraction_ = 0.0;
  double predicted_multi_active_fraction_ = 0.0;
  double predictive_fraction_ = 0.0;
  double learning_fraction_ = 0.0;

  int new_true_bursts_ = 0;
  int new_burst_no_prev_prediction_ = 0;
  int new_burst_prev_prediction_without_segment_ = 0;
  int new_burst_prev_segment_without_prediction_ = 0;
  int new_burst_had_prev_prediction_and_segment_ = 0;

  std::vector<std::uint8_t> prev_true_bursting_;
  std::vector<int> true_burst_counts_;
  std::vector<int> new_true_burst_counts_;
  std::vector<int> position_samples_;
  std::vector<int> position_new_true_burst_counts_;
};

}  // namespace chat_htm::diagnostics
